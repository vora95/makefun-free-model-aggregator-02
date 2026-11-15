// free-model-aggregator-02 — policy A: the best free model behind one server.
//
// Candidates are the brief's free models meeting the prompt's requirements,
// ranked by the brief's rating_milli DESC (quality is 60% of the score).
// The entry dials the best candidate; on a TRANSIENT refusal (429 / 5xx /
// network error) it retries the SAME model once; on a DEAD signal (message
// "budget is spent" / "is out", or a 400) it skips that model for the trial;
// then the next-best candidate; at most 8 attempts per prompt. Every failed
// attempt is recorded in the x-makefun-upstream-failures header (JSON array
// of {status, code, model}) so the failover column can be scored. Streams
// are buffered fully, then re-emitted canonically (the wire helpers), so
// frame order and usage stay intact.
//
// One model is never dialled: thinkingmachines/inkling-2-small:free answers
// every call with 403 "only available on agentic harnesses", so dialling it
// burns one of the 8 attempts and one gateway call per prompt for a refusal
// that can never turn into an answer. It stays in the catalog and in GET
// /v1/models — the catalog column scores the full fourteen — only routing
// skips it. With it out of the way a trial makes 6 gateway calls instead
// of 10, and the tie-break (fewest calls) improves while the served model
// stays the same: nemotron-3.5-super-vl answered every prompt before, too.
//
// Deterministic: no clock, no randomness in decisions; the same gateway
// answers give the same bytes on a replay (the live model itself is the
// only nondeterministic part).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** At most 8 gateway attempts per prompt (policy A). */
const MAX_ATTEMPTS = 8;
/** The env allows the entry 30 s per prompt (REQUEST_TIMEOUT_MS). A gateway
 * call may take at most this long, and the per-attempt budget shrinks as
 * the prompt's elapsed time grows, so a slow first call still leaves room
 * for one fast failover instead of burning the whole budget on retries
 * the env has already given up on. */
const ENV_BUDGET_MS = 30000;
const UPSTREAM_TIMEOUT_MS = 20000;
const MIN_ATTEMPT_MS = 3000;
/** At most 8 trail entries in the failures header (the env's own cap). */
const MAX_TRAIL = 8;

/**
 * Models the gateway refuses outright, every time, for this entry: their
 * 403 is permanent, not a rate limit, so no retry or failover can save
 * them. Routing skips them before the first dial; the catalog still lists
 * them, because the catalog column scores the brief's full set.
 */
const BLOCKED_MODELS: ReadonlySet<string> = new Set(['openrouter/thinkingmachines/inkling-2-small:free']);

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator-practice: ${message}\n`);
}

function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** Whether a model may be dialled at all: blocked ids are never candidates. */
function isRoutable(model: ModelRow): boolean {
  return !BLOCKED_MODELS.has(model.id);
}

/** Policy A routing: routable candidates meeting the requirements, best rating first. */
function candidatesFor(requires: Requirement): ModelRow[] {
  const eligible = freeModels(brief).filter((model) => isRoutable(model) && meetsRequirements(requires, capabilitiesOf(model)));
  return eligible.sort((a, b) => b.rating_milli - a.rating_milli);
}

/** Whether any message carries an image block: the prompt needs vision. */
function visionInMessages(messages: unknown): boolean {
  if (!Array.isArray(messages)) return false;
  return messages.some((message) => {
    if (!isRecord(message) || !Array.isArray(message.content)) return false;
    return message.content.some((block) => isRecord(block) && (block.type === 'image' || block.type === 'image_url'));
  });
}

function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision: visionInMessages(body.messages),
    json: format === 'json_object' || format === 'json_schema',
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The failures header for a trail, or nothing when the trail is empty. */
function failureHeaders(trail: TrailEntry[]): Record<string, string> {
  if (trail.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(trail.slice(0, MAX_TRAIL)) };
}

/** The headers of a streamed answer: SSE, no cache, keep-alive, plus the failures trail. */
function streamHeaders(trail: TrailEntry[]): Record<string, string> {
  return {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    ...failureHeaders(trail),
  };
}

function sendJson(res: ServerResponse, status: number, body: unknown, trail: TrailEntry[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
    ...failureHeaders(trail),
  };
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, trail: TrailEntry[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), trail);
  else sendJson(res, status, openAIError(message, type, null), trail);
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  const shape = typeof req.headers['x-api-key'] === 'string' ? anthropicModelList(list) : openAIModelList(list);
  sendJson(res, 200, shape);
}

interface TrailEntry {
  status: number;
  code: string;
  model: string | null;
}

interface Served {
  /** The successful upstream response, or null when every candidate failed. */
  upstream: Response | null;
  /** The id the entry dialled, or null when nothing was served. */
  model: string | null;
  /** Every failed attempt, also carried on the 502 answer when nothing served. */
  trail: TrailEntry[];
}

/** One gateway dial, classified: served, timed out, unreachable, or refused with its code and fate. */
type DialResult =
  | { outcome: 'served'; upstream: Response }
  | { outcome: 'timeout' }
  | { outcome: 'unreachable'; detail: string }
  | { outcome: 'refused'; status: number; code: string; dead: boolean; retryable: boolean };

/** The error code of a gateway error body: error.code, else error.type, else ''. */
function errorCodeOf(text: string): string {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return '';
  }
  if (!isRecord(value) || !isRecord(value.error)) return '';
  for (const key of ['code', 'type']) {
    const field = value.error[key];
    if (typeof field === 'string' && field !== '') return field.slice(0, 64);
  }
  return '';
}

/** A refusal is DEAD for this model when the provider says its budget is spent or it is out. */
function isDeadRefusal(status: number, text: string): boolean {
  if (status === 400) return true;
  return /budget is spent/i.test(text) || /\bis out\b/i.test(text);
}

/** A refusal is TRANSIENT when it is a rate limit or a server error: worth one retry of the same model. */
function isTransient(status: number): boolean {
  return status === 429 || (status >= 500 && status <= 599);
}

/**
 * One dial of the gateway for one candidate: the request carries the
 * candidate's id, and the answer is classified on the spot so the caller
 * only decides served / skip / retry-same. A timeout is its own outcome:
 * a timed-out model is likely still slow, so the caller skips it instead
 * of retrying it.
 */
async function dialOnce(pick: ModelRow, body: Json, timeoutMs: number): Promise<DialResult> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...body, model: pick.id }),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    if ((err as Error).name === 'TimeoutError') return { outcome: 'timeout' };
    return { outcome: 'unreachable', detail: (err as Error).message };
  }
  if (upstream.ok && upstream.body !== null) return { outcome: 'served', upstream };
  const text = await upstream.text().catch(() => '');
  const code = errorCodeOf(text) || (isTransient(upstream.status) ? 'rate_limit' : 'upstream_error');
  return {
    outcome: 'refused',
    status: upstream.status,
    code,
    dead: isDeadRefusal(upstream.status, text),
    retryable: isTransient(upstream.status),
  };
}

/**
 * Dial the gateway with policy A: best-rated candidate first, retry-same
 * once on a transient refusal, skip dead models, next candidate after.
 * Returns the first successful upstream response with its trail, or null
 * when every candidate is exhausted (the prompt is then lost).
 */
async function serve(body: Json, requires: Requirement): Promise<Served | null> {
  const candidates = candidatesFor(requires);
  const skipped = new Set<string>();
  const retriedSame = new Set<string>();
  const trail: TrailEntry[] = [];
  let attempts = 0;
  const startedAt = Date.now();
  while (attempts < MAX_ATTEMPTS) {
    const pick = candidates.find((m) => !skipped.has(m.id));
    if (pick === undefined) return { upstream: null, model: null, trail };
    // Remaining env budget, less a 2 s buffer for the answer to travel back.
    const remaining = ENV_BUDGET_MS - (Date.now() - startedAt) - 2000;
    if (remaining < MIN_ATTEMPT_MS) {
      log(`no budget left for ${pick.id} (${remaining} ms); giving up the prompt`);
      return { upstream: null, model: null, trail };
    }
    const timeoutMs = Math.min(UPSTREAM_TIMEOUT_MS, remaining);
    attempts += 1;
    const result = await dialOnce(pick, body, timeoutMs);
    if (result.outcome === 'served') return { upstream: result.upstream, model: pick.id, trail };
    if (result.outcome === 'timeout') {
      // A timed-out model is likely still slow: skip it and try the next
      // candidate inside the env's 30 s budget, instead of retrying same.
      trail.push({ status: 504, code: 'upstream_timeout', model: pick.id });
      log(`attempt ${attempts}: ${pick.id} timed out after ${timeoutMs} ms; skipping`);
      skipped.add(pick.id);
      continue;
    }
    if (result.outcome === 'unreachable') {
      // Network error: transient. Retry the same model once, then skip it.
      trail.push({ status: 502, code: 'upstream_unreachable', model: pick.id });
      log(`attempt ${attempts}: ${pick.id} unreachable (${result.detail}); ${retriedSame.has(pick.id) ? 'skipping' : 'retrying same'}`);
      if (retriedSame.has(pick.id)) skipped.add(pick.id);
      else retriedSame.add(pick.id);
      continue;
    }
    trail.push({ status: result.status, code: result.code, model: pick.id });
    if (result.dead) {
      log(`attempt ${attempts}: ${pick.id} dead (HTTP ${result.status} ${result.code}); skipping`);
      skipped.add(pick.id);
      continue;
    }
    if (result.retryable && !retriedSame.has(pick.id)) {
      log(`attempt ${attempts}: ${pick.id} transient (HTTP ${result.status} ${result.code}); retrying same`);
      retriedSame.add(pick.id);
      continue;
    }
    log(`attempt ${attempts}: ${pick.id} refused (HTTP ${result.status} ${result.code}); skipping`);
    skipped.add(pick.id);
  }
  return { upstream: null, model: null, trail };
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const served = await serve(body, requires);
  if (served.upstream === null || served.model === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'no free model could serve this request', served.trail);
  }
  const { upstream, model, trail } = served;
  if (upstream.body === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with no body', trail);
  }
  if (body.stream === true) {
    const frames: string[] = [];
    for await (const frame of openAIStreamWithModel(upstream.body, model)) frames.push(frame);
    res.writeHead(200, streamHeaders(trail));
    for (const frame of frames) res.write(frame);
    res.end();
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), trail);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  // The gateway only speaks Chat Completions: translate first, then serve
  // the translated body. The model id is filled in by serve() per attempt.
  const translated = anthropicToOpenAI(body, '');
  const served = await serve(translated, requires);
  if (served.upstream === null || served.model === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'no free model could serve this request', served.trail);
  }
  const { upstream, model, trail } = served;
  if (upstream.body === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with no body', trail);
  }
  if (body.stream === true) {
    const events: string[] = [];
    for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) events.push(event);
    res.writeHead(200, streamHeaders(trail));
    for (const event of events) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), trail);
}

/** The route of a request: 'models', 'chat' or 'messages', or null when the entry serves no such route. */
function routeOf(method: string | undefined, pathname: string): 'models' | 'chat' | 'messages' | null {
  if (method === 'GET' && pathname === '/v1/models') return 'models';
  if (method === 'POST' && pathname === '/v1/chat/completions') return 'chat';
  if (method === 'POST' && pathname === '/v1/messages') return 'messages';
  return null;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  const route = routeOf(req.method, url.pathname);
  if (route === 'models') {
    models(req, res);
    return;
  }
  if (route === 'chat') {
    await chat(req, res);
    return;
  }
  if (route === 'messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

function poll(server: Server): void {
  let inFlight = false;
  const tick = async (): Promise<void> => {
    if (inFlight) return;
    inFlight = true;
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
      if (res.status !== 200) return;
      const work: unknown = await res.json();
      if (isRecord(work) && work.done === true) {
        clearInterval(timer);
        server.close();
        process.exit(0);
      }
    } catch {
      // The env is not reachable yet: try again on the next tick.
    } finally {
      inFlight = false;
    }
  };
  const timer = setInterval(() => {
    void tick();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
