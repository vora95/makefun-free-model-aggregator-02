// The brief's two documents, the free model list, and the catalog.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) is prose with two JSON documents in it, and the
// prose around them does not matter: this file finds each by its `schema` member, under this contest's
// prefix (`free-model-aggregator-02/`).
//
//   - the free model list (`free-model-aggregator-02/models/v1`): one row a model, keyed by the id the
//     gateway routes (`openrouter/<vendor>/<model>:free`), with the model's public rating in
//     milli-points, what it can do (tools, vision, JSON mode, context length) and its free flag. Only a
//     free model may serve a prompt; this baseline never routes a paid one.
//   - the six prompts (`free-model-aggregator-02/prompts/v1`): the prompts the house's client will
//     send, in order, each with its id, its api (`openai` or `anthropic`), whether it streams, its
//     weight (1 or 2) and the requirements a model must meet to serve it. The client sends the bodies;
//     this baseline routes each arrival by its row, which is why the prompt list is read before the
//     first request.
//
// A brief may spell the model list `models` or `ratings` and the prompts `prompts` or `manifest` (the
// sibling -01 contest's names), and one document may carry both; a document is found by the arrays it
// carries, never by a name alone. The brief's own prose is never parsed.
//
// Everything here is deterministic: no clock and no randomness, so `node main.ts --list` and the
// register body build the same document, byte for byte, on a live pass and on its replay.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this contest's brief writes names this prefix; the brief's JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches for them. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this baseline reads it. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/**
 * A cursor over the brief's text that finds JSON objects one `{` at a
 * time. Every `{` is tried as an opening brace, in order — nested objects
 * included — so a document buried inside another still turns up. A `{`
 * whose braces never balance, or whose text is not JSON, is skipped and the
 * scan moves on to the next one.
 */
class BraceScanner {
  private readonly text: string;
  private pos = 0;

  constructor(text: string) {
    this.text = text;
  }

  /** The next `{` at or after the cursor, or -1 when there is none. */
  private nextOpen(): number {
    const at = this.text.indexOf('{', this.pos);
    this.pos = at === -1 ? this.text.length : at;
    return at;
  }

  /**
   * The index of the `}` closing the object that opens at `start`, or -1.
   * String contents are skipped, and a backslash inside a string escapes
   * the character after it, so a brace inside a string never counts.
   */
  private matchingClose(start: number): number {
    let depth = 0;
    let inString = false;
    for (let i = start; i < this.text.length; i += 1) {
      const ch = this.text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) return i;
      }
    }
    return -1;
  }

  /** The next JSON object in the text, or null when the text holds no more. */
  nextObject(): Json | null {
    for (;;) {
      const start = this.nextOpen();
      if (start === -1) return null;
      this.pos = start + 1;
      const end = this.matchingClose(start);
      if (end === -1) continue;
      try {
        const value: unknown = JSON.parse(this.text.slice(start, end + 1));
        if (isRecord(value)) return value;
      } catch {
        // Not JSON on its own: look on.
      }
    }
  }
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const scanner = new BraceScanner(text);
  const out: Json[] = [];
  let doc: Json | null;
  while ((doc = scanner.nextObject()) !== null) {
    if (typeof doc.schema === 'string') out.push(doc);
  }
  return out;
}

/** A field as a string, or '' when it is missing or not one. */
function strField(row: Json, key: string): string {
  return typeof row[key] === 'string' ? (row[key] as string) : '';
}

/** A field as a finite number, or null when it is missing or not one. */
function numField(row: Json, key: string): number | null {
  const v = row[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** A field as a boolean, or null when it is missing or not one. */
function boolField(row: Json, key: string): boolean | null {
  const v = row[key];
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix that carries one of
 * `keys` as an array: the array itself. A brief that spells its lists in
 * the sibling's names still reads.
 */
function firstList(docs: readonly Json[], keys: readonly string[]): unknown[] | null {
  for (const doc of docs) {
    if (!strField(doc, 'schema').includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return doc[key] as unknown[];
    }
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it: a row that
 * gives the bare OpenRouter id (`vendor/model:free`) and names `openrouter` is routed as the gateway
 * spells it (`openrouter/vendor/model:free`). The brief lists free models, so a row is free unless it
 * says `free: false`; a paid row is never routed.
 */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = strField(v, 'id') !== '' ? strField(v, 'id') : strField(v, 'model');
  if (raw === '') return null;
  const named = strField(v, 'provider') !== '' ? strField(v, 'provider') : strField(v, 'wire');
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: strField(v, 'display_name') !== '' ? strField(v, 'display_name') : raw,
    rating_milli: numField(v, 'rating_milli') ?? 0,
    context_length: numField(v, 'context_length'),
    tools: boolField(v, 'tools'),
    vision: boolField(v, 'vision'),
    json: boolField(v, 'json'),
    free: v.free !== false,
  };
}

/** The requirements one prompt row names, as a Requirement: only the flags the row sets. */
function requirementsOf(v: Json): Requirement {
  const requires = isRecord(v.requires) ? v.requires : {};
  const out: Requirement = {};
  if (requires.tools === true) out.tools = true;
  if (requires.vision === true) out.vision = true;
  if (requires.json === true) out.json = true;
  const minContext = numField(requires, 'min_context');
  if (minContext !== null) out.min_context = minContext;
  return out;
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = strField(v, 'id');
  if (id === '') return null;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: requirementsOf(v),
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsList = firstList(docs, ['models', 'ratings']);
  if (modelsList === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models: ModelRow[] = [];
  for (const v of modelsList) {
    const m = modelOf(v);
    if (m !== null) models.push(m);
  }
  const promptsList = firstList(docs, ['prompts', 'manifest']);
  if (promptsList === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts: PromptRow[] = [];
  for (const v of promptsList) {
    const p = promptOf(v);
    if (p !== null) prompts.push(p);
  }
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/**
 * Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one.
 * Each requirement is one check; the first one the model fails decides the answer.
 */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  const checks: Array<[boolean, boolean]> = [
    [requires.tools === true, caps.tools === true],
    [requires.vision === true, caps.vision === true],
    [requires.json === true, caps.json === true],
  ];
  for (const [needed, met] of checks) {
    if (needed && !met) return false;
  }
  if (requires.min_context !== undefined) {
    if (caps.context_length === null || caps.context_length < requires.min_context) return false;
  }
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's trial-local address: the gateway's
 * origin differs between a live pass and its replay, and the catalog is part of the state export a
 * replay must reproduce byte for byte.
 *
 * This baseline lists every free model in the brief — exactly the fourteen dialable ids the catalog
 * column scores — so keep it that way: register this set and keep GET /v1/models (both auth headers)
 * exactly that set, since listing fewer loses the column as surely as listing more.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const models: CatalogModel[] = [];
  const seen = new Set<string>();
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
