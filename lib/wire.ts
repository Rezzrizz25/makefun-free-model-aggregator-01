// lib/wire.ts — the client wires and the three upstream wires.
//
// The trial's client calls this server as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), in JSON or
// server-sent events. Each provider speaks its own wire — openai, anthropic
// or gemini — and this file translates between them in both directions.
//
// The providers in this trial answer text only: their answer is one text
// (`ANSWER ECHO<the request's parsed fields as JSON>`) with a plain stop
// reason, so a translation's job is to carry that text, the usage counts and
// the stop reason across the wire boundary without changing a byte of the
// text. Every check the client runs compares the echo inside the answer
// with what it sent, so a field dropped or rewritten in translation loses
// the request: the system prompt, the user text (images included), the tool
// names in order, the stop sequences, max_tokens and JSON mode all pass
// through untouched.
//
// Strategy per pair of wires:
//   - same wire (OpenAI client to an OpenAI provider, Anthropic to
//     Anthropic): the bytes pass through unchanged;
//   - any other pair: the provider's whole answer is read first, parsed
//     into one normalized answer (text, input/output tokens, stop reason),
//     and re-rendered in the client's wire, in JSON or as the client's
//     streaming event order. Buffering a cross-wire stream is fine: the
//     client checks the events' order and content, not their timing.
//
// Nothing here reads the clock or randomness, so the same body and the
// same provider answer give the same bytes on a replay. The model lists
// use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request: Anthropic's list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One tool call in a provider's answer, normalized across the wires. */
export interface NormalizedToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** One provider answer, wire-neutral: its text, usage and stop reason. */
export interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The stop reason in the client's family: stop, length, tool_calls or content_filter. */
  stop: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  toolCalls: NormalizedToolCall[];
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** OpenAI's finish_reason as the normalized stop reason. */
function stopOfOpenAI(finish: unknown): NormalizedAnswer['stop'] {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

/** Anthropic's stop_reason as the normalized stop reason. */
function stopOfAnthropic(reason: unknown): NormalizedAnswer['stop'] {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as the normalized stop reason. */
function stopOfGemini(reason: unknown): NormalizedAnswer['stop'] {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** The normalized stop reason back in the OpenAI wire's spelling. */
function openAIStop(stop: NormalizedAnswer['stop']): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool_calls') return 'tool_calls';
  if (stop === 'content_filter') return 'content_filter';
  return 'stop';
}

/** The normalized stop reason back in the Anthropic wire's spelling. */
function anthropicStop(stop: NormalizedAnswer['stop']): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Parsing provider answers into the normalized form.
// ---------------------------------------------------------------------------

/** A Chat Completions JSON answer as a normalized answer. */
export function parseOpenAIJson(body: Json): NormalizedAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const content = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: NormalizedToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === 'string' ? call.id : 'call_unknown',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: content,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: stopOfOpenAI(choice.finish_reason),
    toolCalls,
  };
}

/** An Anthropic Messages JSON answer as a normalized answer. */
export function parseAnthropicJson(body: Json): NormalizedAnswer | null {
  if (body.type !== 'message' || !Array.isArray(body.content)) return null;
  const toolCalls: NormalizedToolCall[] = [];
  for (const block of body.content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      name: typeof block.name === 'string' ? block.name : '',
      arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: stopOfAnthropic(body.stop_reason),
    toolCalls,
  };
}

/** A Gemini generateContent JSON answer as a normalized answer. */
export function parseGeminiJson(body: Json): NormalizedAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: stopOfGemini(candidate.finishReason),
    toolCalls: [],
  };
}

/** One SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A Chat Completions SSE stream as a normalized answer. */
export function parseOpenAIStream(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop: NormalizedAnswer['stop'] = 'stop';
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = stopOfOpenAI(choice.finish_reason);
  }
  if (!sawAny) return null;
  return { text: out, inputTokens: input, outputTokens: output, stop, toolCalls: [] };
}

/** An Anthropic SSE stream as a normalized answer. */
export function parseAnthropicStream(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop: NormalizedAnswer['stop'] = 'stop';
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = stopOfAnthropic(data.delta.stop_reason);
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  if (!sawAny) return null;
  return { text: out, inputTokens: input, outputTokens: output, stop, toolCalls: [] };
}

/** A Gemini :streamGenerateContent SSE stream as a normalized answer. */
export function parseGeminiStream(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop: NormalizedAnswer['stop'] = 'stop';
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    sawAny = true;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const p of parts) if (isRecord(p) && typeof p.text === 'string') out += p.text;
      if (typeof candidate.finishReason === 'string') stop = stopOfGemini(candidate.finishReason);
    }
    if (isRecord(chunk.usageMetadata)) {
      input = numberOrZero(chunk.usageMetadata.promptTokenCount);
      output = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  if (!sawAny) return null;
  return { text: out, inputTokens: input, outputTokens: output, stop, toolCalls: [] };
}

// ---------------------------------------------------------------------------
// Rendering a normalized answer in a client wire.
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into pieces for streaming deltas. */
function splitText(text: string, size = 48): string[] {
  if (text === '') return [];
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces;
}

/** A normalized answer as a Chat Completions JSON object. */
export function renderOpenAIJson(answer: NormalizedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
  }
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStop(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A normalized answer as an Anthropic Messages JSON object. */
export function renderAnthropicJson(answer: NormalizedAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: Json = {};
    try {
      const parsed: unknown = JSON.parse(call.arguments);
      if (isRecord(parsed)) input = parsed;
    } catch {
      // Not JSON: an empty input.
    }
    content.push({ type: 'tool_use', id: call.id, name: call.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A normalized answer as a Chat Completions SSE stream, ending with [DONE]. */
export function renderOpenAIStream(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStop(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as the Anthropic SSE event order. */
export function renderAnthropicStream(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let nextIndex = 0;
  if (answer.text !== '') {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      }),
    );
    nextIndex = 1;
    for (const piece of splitText(answer.text)) {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'text_delta', text: piece },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  }
  for (const call of answer.toolCalls) {
    const index = nextIndex;
    nextIndex += 1;
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    if (call.arguments !== '') {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.arguments },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Rewriting a client request for an upstream wire.
// ---------------------------------------------------------------------------

/** The output cap of a Chat Completions body: max_tokens, else max_completion_tokens. */
function openAIMaxTokens(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

/** The stop sequences of a Chat Completions body, as a list. */
function openAIStops(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/** A data URL split into its media type and base64 payload, or null. */
function splitDataUrl(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** One Anthropic content block as an OpenAI content part (text or image). */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as a Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and
 * the cap, temperature, top_p, stop sequences and streaming pass through.
 * Every user message is emitted, even one that holds only tool results:
 * its text is empty, and the provider must echo exactly that.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
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
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    messages.push({ role: 'user', content: parts.length > 0 ? parts : '' });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    // The client checks the stream's final usage; the provider only sends
    // it when asked, so always ask.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * A Chat Completions request as an Anthropic Messages request naming
 * `model`. System messages join into one system prompt, user parts become
 * text/image blocks, assistant tool calls become tool_use blocks, tool
 * messages become tool_result blocks, and the cap, temperature, top_p,
 * stop sequences and streaming pass through. Every user message is
 * emitted, even an image-only one, so the provider echoes its (empty)
 * text exactly as the client sent it.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) input = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          }
          blocks.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input,
          });
        }
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    // A user message: text and image parts, in order.
    const blocks: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const split = splitDataUrl(part.image_url.url);
          if (split !== null) {
            blocks.push({ type: 'image', source: { type: 'base64', media_type: split.mime, data: split.data } });
          } else {
            blocks.push({ type: 'image', source: { type: 'url', url: part.image_url.url } });
          }
        }
      }
    }
    messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'auto' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One Gemini content part for a text or image block. */
function geminiPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return {
        inlineData: {
          mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
          data: source.data,
        },
      };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** One Gemini content part for an OpenAI image_url part. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const split = splitDataUrl(part.image_url.url);
    if (split !== null) return { inlineData: { mimeType: split.mime, data: split.data } };
    return { fileData: { fileUri: part.image_url.url } };
  }
  return null;
}

/**
 * A Chat Completions request as a Gemini generateContent body. System
 * messages become the systemInstruction, user messages become user
 * contents, assistant messages become model contents, and tool messages
 * ride as functionResponse parts on the most recent user content — never
 * as a content of their own, because a trailing content with no text
 * would make the provider echo an empty user text the client never sent.
 * Tools become functionDeclarations, the cap becomes maxOutputTokens,
 * stops become stopSequences, and JSON mode becomes responseMimeType.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  let lastUser: Json | null = null;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let args: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) args = parsed;
            } catch {
              // Not JSON: empty arguments.
            }
          }
          parts.push({
            functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args },
          });
        }
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const response: Json = {
        functionResponse: {
          name: '',
          response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
        },
      };
      if (lastUser !== null && Array.isArray(lastUser.parts)) {
        (lastUser.parts as Json[]).push(response);
      } else {
        const fresh: Json = { role: 'user', parts: [response] };
        contents.push(fresh);
        lastUser = fresh;
      }
      continue;
    }
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const p = geminiPartOfOpenAI(part);
        if (p !== null) parts.push(p);
      }
    }
    const entry: Json = { role: 'user', parts };
    contents.push(entry);
    lastUser = entry;
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = openAIStops(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (isRecord(body.response_format)) {
    const type = body.response_format.type;
    if (type === 'json_object' || type === 'json_schema') config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The
 * same shape as openAIToGemini: system to systemInstruction, user blocks
 * to user contents, assistant blocks to model contents, tool results as
 * functionResponse parts on the most recent user content.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const system = textOf(body.system);
  const contents: Json[] = [];
  let lastUser: Json | null = null;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      const entry: Json = assistant
        ? { role: 'model', parts: [{ text: content }] }
        : { role: 'user', parts: [{ text: content }] };
      contents.push(entry);
      if (!assistant) lastUser = entry;
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const parts: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        }
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        const response: Json = {
          functionResponse: {
            name: '',
            response: { content: toolResultText(block.content) },
          },
        };
        if (lastUser !== null && Array.isArray(lastUser.parts)) (lastUser.parts as Json[]).push(response);
        else parts.push(response);
        continue;
      }
      const p = geminiPartOf(block);
      if (p !== null) parts.push(p);
    }
    const entry: Json = { role: 'user', parts };
    contents.push(entry);
    lastUser = entry;
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}
