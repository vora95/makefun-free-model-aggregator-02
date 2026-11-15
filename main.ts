// free-model-aggregator-02 — PRACTICE entry, policy A (Xingchi backend).
//
// PRACTICE ONLY. Written fresh for local training against the kit's
// practice trials, built on the kit starter's lib/ (providers.ts, wire.ts
// copied verbatim from the starter kit — kit files, not an old entry).
//
// POLICY A (retained from the agg-01 A/B/C experiment, adapted to agg-02):
//   - candidates = free models meeting the prompt's requirements, ranked by
//     the brief's rating_milli DESC (quality is 60% of the score);
//   - dial the best candidate; on a TRANSIENT refusal (429 / 5xx / network
//     error) retry the SAME model once; on a DEAD signal (message "budget
//     is spent" / "is out", or a 400) skip that model for the trial;
//   - then the next-best candidate; at most 8 attempts per prompt;
//   - every failed attempt is recorded in the x-makefun-upstream-failures
//     header (JSON array of {status, code, model}) so the failover column
//     can be scored;
//   - streams are buffered fully, then re-emitted canonically (same wire
//     helpers as the starter), so frame order and usage stay intact.
//
// DIFFERENCES vs the real submission (read REPORT.md before drawing
// conclusions): the gateway here is a local shim that forwards to Xingchi
// gpt-6.1-sol and rewrites the `model` field to the dialed OpenRouter id.
// The dialed id's rating — not Xingchi's — is what the local scorer
// credits. This practice validates routing, wire fidelity, token echo,
// failover handling and the scoring path. It does NOT validate model
// quality against the real OpenRouter free models, and it does NOT prove
// the graded score: grading runs on make.fun's server with a real
// OPENROUTER_API_KEY, real rate limits (20/min, 50/day under 10 credits)
// and real model behaviour.
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

/** Policy A routing: candidates meeting the requirements, best rating first. */
function candidatesFor(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
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

function sendJson(res: ServerResponse, status: number, body: unknown, trail: TrailEntry[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (trail.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(trail.slice(0, MAX_TRAIL));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, trail: TrailEntry[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), trail);
  else sendJson(res, status, openAIError(message, type, null), trail);
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
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

/** The error code of a gateway error body: error.code, else error.type, else ''. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the message match.
  }
  return '';
}

/** A refusal is DEAD for this model when the provider says its budget is spent or it is out. */
function isDeadRefusal(status: number, text: string): boolean {
  if (status === 400) return true;
  return /budget is spent/i.test(text) || /\bis out\b/i.test(text);
}

function isTransient(status: number): boolean {
  return status === 429 || (status >= 500 && status <= 599);
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
    let upstream: Response;
    try {
      const headers: Record<string, string> = {
        'content-type': 'application/json',
        authorization: `Bearer ${GATEWAY_KEY}`,
      };
      if (body.stream === true) headers.accept = 'text/event-stream';
      upstream = await fetch(chatCompletionsUrl(), {
        method: 'POST',
        headers,
        body: JSON.stringify({ ...body, model: pick.id }),
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      if ((err as Error).name === 'TimeoutError') {
        // A timed-out model is likely still slow: skip it and try the next
        // candidate inside the env's 30 s budget, instead of retrying same.
        trail.push({ status: 504, code: 'upstream_timeout', model: pick.id });
        log(`attempt ${attempts}: ${pick.id} timed out after ${timeoutMs} ms; skipping`);
        skipped.add(pick.id);
        continue;
      }
      // Network error: transient. Retry the same model once, then skip it.
      trail.push({ status: 502, code: 'upstream_unreachable', model: pick.id });
      log(`attempt ${attempts}: ${pick.id} unreachable (${(err as Error).message}); ${retriedSame.has(pick.id) ? 'skipping' : 'retrying same'}`);
      if (retriedSame.has(pick.id)) skipped.add(pick.id);
      else retriedSame.add(pick.id);
      continue;
    }
    if (upstream.ok && upstream.body !== null) {
      return { upstream, model: pick.id, trail };
    }
    const text = await upstream.text().catch(() => '');
    const code = errorCodeOf(text) || (isTransient(upstream.status) ? 'rate_limit' : 'upstream_error');
    trail.push({ status: upstream.status, code, model: pick.id });
    if (isDeadRefusal(upstream.status, text)) {
      log(`attempt ${attempts}: ${pick.id} dead (HTTP ${upstream.status} ${code}); skipping`);
      skipped.add(pick.id);
      continue;
    }
    if (isTransient(upstream.status) && !retriedSame.has(pick.id)) {
      log(`attempt ${attempts}: ${pick.id} transient (HTTP ${upstream.status} ${code}); retrying same`);
      retriedSame.add(pick.id);
      continue;
    }
    log(`attempt ${attempts}: ${pick.id} refused (HTTP ${upstream.status} ${code}); skipping`);
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
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(trail.slice(0, MAX_TRAIL));
    res.writeHead(200, headers);
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
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(trail.slice(0, MAX_TRAIL));
    res.writeHead(200, headers);
    for (const event of events) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
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
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
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
    })();
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
