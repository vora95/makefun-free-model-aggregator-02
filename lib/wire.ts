// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls the entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. The entry dials the grader's gateway, an
// OpenAI-compatible endpoint, so every prompt goes upstream as a Chat
// Completions request and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id)
//     to the gateway and relays the gateway's JSON or SSE answer, with every
//     `model` field set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system
//     prompt, image blocks, tool definitions, tool results and tool choice
//     included) and rewrites the gateway's answer into a Messages object, or
//     into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the
//     stop reason and the usage translated. Tool calls become tool_use blocks
//     and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really
//   answered, so a response's `model` must be the model this entry dialled.
//   OpenRouter's own answer names a free model without the gateway's
//   `openrouter/` prefix; these functions put the dialled id back on every
//   answer and every streamed chunk. They never make a model up: the id came
//   from the brief and went to the gateway in the request's `model`.
//
// THE STREAM RELAY
//   Both stream relays read the upstream body through one shared frame
//   splitter below: frames are the text between blank lines, as the SSE
//   grammar splits them, and a frame cut across reads is held until its end
//   arrives. The OpenAI relay rewrites each frame in place; the Anthropic
//   relay is a small state machine that turns frames into Messages events.
//
// Everything here is a pure function of its arguments, except the stream
// relays: no clock and no randomness, so the same body and the same gateway
// answer give the same bytes on a replay. The two model listings use fixed
// timestamps for the same reason: a `created` that moved between runs would
// be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  const data = models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider }));
  return { object: 'list', data };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  const data = models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT }));
  return { data, has_more: false };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const texts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  return texts.join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text === '' ? JSON.stringify(content) : text;
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value !== 'string' || value.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/** A finite number, or 0 for anything else. */
function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  switch (finish) {
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'refusal';
    default:
      return 'end_turn';
  }
}

/** The system prompt of an Anthropic body as one system message, or nothing when it is empty. */
function systemMessage(body: Json): Json[] {
  const system = textOf(body.system);
  return system === '' ? [] : [{ role: 'system', content: system }];
}

/** One Anthropic assistant turn as an OpenAI assistant message: text plus its tool calls. */
function assistantMessage(raw: Json): Json {
  const content = raw.content;
  if (typeof content === 'string') return { role: 'assistant', content };
  const toolCalls: Json[] = [];
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!isRecord(block) || block.type !== 'tool_use') continue;
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const text = textOf(content);
  const message: Json = { role: 'assistant', content: text === '' ? null : text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return message;
}

/**
 * One Anthropic user turn as OpenAI messages: tool_result blocks become
 * tool messages in place, the remaining blocks become one user message
 * with image_url parts, pushed after the tool messages.
 */
function userMessages(raw: Json): Json[] {
  const out: Json[] = [];
  const content = raw.content;
  if (typeof content === 'string') return [{ role: 'user', content }];
  if (!Array.isArray(content)) return out;
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'tool_result') {
      out.push({
        role: 'tool',
        tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        content: toolResultText(block.content),
      });
      continue;
    }
    const part = partOf(block);
    if (part !== null) parts.push(part);
  }
  if (parts.length > 0) out.push({ role: 'user', content: parts });
  return out;
}

/** The Anthropic `tools` array as OpenAI function tools; [] when it names none. */
function openAITools(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const tools: Json[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({
      type: 'function',
      function: {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    });
  }
  return tools;
}

/** The Anthropic `tool_choice` as an OpenAI one; undefined when it names none. */
function openAIToolChoice(body: Json): Json | string | undefined {
  const choice = body.tool_choice;
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'tool' && typeof choice.name === 'string') {
    return { type: 'function', function: { name: choice.name } };
  }
  return undefined;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [...systemMessage(body)];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') messages.push(assistantMessage(raw));
    else messages.push(...userMessages(raw));
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  const tools = openAITools(body);
  if (tools.length > 0) out.tools = tools;
  const choice = openAIToolChoice(body);
  if (choice !== undefined) out.tool_choice = choice;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The content blocks of a Chat Completions answer: text first, then one tool_use per call. */
function answerBlocks(message: Json): Json[] {
  const blocks: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') blocks.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    blocks.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
  return blocks;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content: answerBlocks(message),
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
export function openAIJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  const head = event === null ? '' : `event: ${event}\n`;
  return `${head}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * An upstream SSE body as complete frames: the text between blank lines,
 * as the SSE grammar splits them. A frame cut across reads is held until
 * its end arrives, and a body that ends without one is yielded as it is.
 * Both stream relays below read through this one splitter.
 */
async function* splitSseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield buffer;
  } finally {
    reader.releaseLock();
  }
}

/**
 * One SSE line with its data payload's `model` set to `model`, and its
 * blank line kept: comments, `event:` lines and payloads that are not JSON
 * objects pass through untouched. Chat Completions names the model on its
 * chunks; the client sees the id the entry dialled, never the provider's
 * spelling.
 *
 * OpenRouter repeats the finish chunk: after the chunk that names a
 * choice's finish_reason, the usage chunk names it again. A stream with two
 * finish chunks fails the env's stream_order check, so only the first
 * finish_reason per choice survives: later ones are nulled out, and the
 * chunk itself (its usage, its delta) is kept — the usage check reads the
 * usage chunk, and the stop reason is the first finish chunk's. `finished`
 * is the per-stream set of choice indexes that already finished.
 */
function rewriteDataLine(line: string, model: string, finished: Set<number>): string {
  const trimmed = line.trim();
  if (!trimmed.startsWith('data:')) return line;
  const payload = trimmed.slice(5).trim();
  if (payload === '' || payload === '[DONE]') return line;
  let chunk: unknown;
  try {
    chunk = JSON.parse(payload);
  } catch {
    return line;
  }
  if (!isRecord(chunk)) return line;
  const out: Json = { ...chunk, model };
  const choices = out.choices;
  if (Array.isArray(choices)) {
    out.choices = choices.map((raw, index) => {
      if (!isRecord(raw) || typeof raw.finish_reason !== 'string') return raw;
      if (finished.has(index)) return { ...raw, finish_reason: null };
      finished.add(index);
      return raw;
    });
  }
  return `data: ${JSON.stringify(out)}`;
}

/** One SSE frame with every data payload's `model` set to `model`. */
function rewriteFrame(frame: string, model: string, finished: Set<number>): string {
  const lines = frame.split('\n').map((line) => rewriteDataLine(line, model, finished));
  return `${lines.join('\n')}\n\n`;
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id the
 * entry dialled. OpenRouter's usage chunk repeats the finish_reason, so the relay dedupes it: only the first
 * finish_reason per choice is kept (see rewriteDataLine).
 */
export async function* openAIStreamWithModel(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const finished = new Set<number>();
  for await (const frame of splitSseFrames(upstream)) {
    yield rewriteFrame(frame, model, finished);
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence, as a
 * state machine over the upstream frames: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then
 * message_delta with the stop reason and usage, then message_stop. `model`
 * is the id the entry dialled, named back to the client.
 */
class AnthropicRelay {
  private readonly model: string;
  private started = false;
  private finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  private pendingStop: string | null = null;
  private textIndex = -1;
  private nextIndex = 0;
  private readonly toolIndexes = new Map<number, number>();
  private inputTokens = 0;
  private outputTokens = 0;

  constructor(model: string) {
    this.model = model;
  }

  /** The message_start event, once, carrying the usage known so far. */
  private openMessage(): string[] {
    if (this.started) return [];
    this.started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model: this.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: this.inputTokens, output_tokens: 0 },
        },
      }),
    ];
  }

  /** Stop events for every open block: the text block first, then tool blocks in index order. */
  private closeOpenBlocks(): string[] {
    const out: string[] = [];
    if (this.textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: this.textIndex }));
      this.textIndex = -1;
    }
    const indexes = [...this.toolIndexes.values()].sort((a, b) => a - b);
    for (const index of indexes) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    this.toolIndexes.clear();
    return out;
  }

  /** message_delta with the stop reason and usage, then message_stop, once. */
  private finishRelay(stopReason: string): string[] {
    if (this.finished) return [];
    this.finished = true;
    return [
      ...this.closeOpenBlocks(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: this.outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  }

  /** The text delta of one chunk as Anthropic events, opening the text block on first use. */
  private textEvents(text: string): string[] {
    const out: string[] = [];
    if (text === '') return out;
    if (this.textIndex === -1) {
      this.textIndex = this.nextIndex;
      this.nextIndex += 1;
      out.push(
        sseEvent('content_block_start', {
          type: 'content_block_start',
          index: this.textIndex,
          content_block: { type: 'text', text: '' },
        }),
      );
    }
    out.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: this.textIndex,
        delta: { type: 'text_delta', text },
      }),
    );
    return out;
  }

  /** One streamed tool call as Anthropic events: open its block on first sight, then its arguments. */
  private toolCallEvents(call: Json): string[] {
    const out: string[] = [];
    const upstreamIndex = typeof call.index === 'number' ? call.index : this.toolIndexes.size;
    let index = this.toolIndexes.get(upstreamIndex);
    if (index === undefined) {
      if (this.textIndex !== -1) {
        out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: this.textIndex }));
        this.textIndex = -1;
      }
      index = this.nextIndex;
      this.nextIndex += 1;
      this.toolIndexes.set(upstreamIndex, index);
      const fn = isRecord(call.function) ? call.function : {};
      out.push(
        sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        }),
      );
    }
    const fn = isRecord(call.function) ? call.function : {};
    if (typeof fn.arguments === 'string' && fn.arguments !== '') {
      out.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: fn.arguments },
        }),
      );
    }
    return out;
  }

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  private eventsOfFrame(frame: string): string[] {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...this.openMessage(), ...this.finishRelay(this.pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        this.inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        this.outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...this.openMessage());
      if (typeof delta.content === 'string') out.push(...this.textEvents(delta.content));
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        out.push(...this.toolCallEvents(call));
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        this.pendingStop = stopReasonOf(choice.finish_reason);
      }
    }
    return out;
  }

  /** The whole upstream body as Anthropic events, closing the message when the body ends. */
  async *run(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
    for await (const frame of splitSseFrames(upstream)) {
      yield* this.eventsOfFrame(frame);
    }
    if (!this.finished) {
      yield* this.openMessage();
      yield* this.finishRelay(this.pendingStop ?? 'end_turn');
    }
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the id the entry dialled, named back to the client.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  yield* new AnthropicRelay(model).run(upstream);
}
