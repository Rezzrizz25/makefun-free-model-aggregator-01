// main.ts — one server, two client wires, ten simulated free providers.
//
// The best free AI model aggregator: for each of the trial's 20 requests
// it picks the highest-rated free model that meets the request's
// requirements, speaks that provider's wire (OpenAI, Anthropic or Gemini),
// and answers in the wire the client used, in JSON or server-sent events.
// When a provider refuses (its free-tier budget spent, its failure
// schedule, or its outage), the request is answered by the next-best free
// model that meets it instead.
//
// How a trial runs:
//   1. read the brief at $MAKEFUN_PROMPT_FILE (providers, quality table,
//      workload manifest) and probe every provider's catalog route;
//   2. listen on 0.0.0.0 at a free port and POST $MAKEFUN_ENV_URL/register
//      with protocol_version 1, the port and the one catalog document;
//   3. serve the client's requests: GET /v1/models (in the shape of the
//      auth header it got), POST /v1/chat/completions and POST
//      /v1/messages, each in JSON or streaming;
//   4. poll GET $MAKEFUN_ENV_URL/work and exit 0 the moment it says done.
//      A process still running when the wall ends scores zero, so nothing
//      waits on anything else.
// `node main.ts --list` prints the same catalog document and exits 0
// before any server starts; outside a trial it prints {"providers":[]}.
//
// Routing, in short. Candidates are the free models meeting the request,
// ranked by the quality table's rating (ties break by provider id, then
// model id, so the order is fixed). Each provider's per-trial budget is
// drawn inside a range the kit's public provider table publishes, so the
// planner treats the bottom of that range as a safe cap: a provider is
// only planned up to its cap, and the rest of its budget is a bonus the
// failover cascade can spend. A provider that refuses at or past its cap
// is treated as spent and drops to the back of the order; one that refuses
// before its cap (its failure schedule, or an outage starting) is skipped
// for the next request only, then tried again — a scheduled refusal is
// transient, a spent budget is not.
//
// Determinism: every choice is a function of the brief, the requests and
// the providers' answers so far. No wall clock (fixed timestamps in every
// reply) and no unseeded randomness, so a replayed trial leaves the same
// state export byte for byte.
//
// Node 22's standard library alone: no dependencies, nothing installed.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most provider attempts spent on one client request (plus one retry). */
const MAX_ATTEMPTS = 6;

/**
 * The bottom of each provider's per-trial free-tier budget range, as the
 * kit's public provider table draws it. Planning never spends past it, so
 * a planned call cannot be refused for a spent budget; anything above it
 * is headroom the failover cascade may use. An unknown provider gets the
 * smallest range's bottom, 4.
 */
const BUDGET_FLOOR: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** The index of the client request being served, for probation windows. */
let requestIndex = 0;

/** Per-provider call state, for planning and failover. */
interface ProviderState {
  /** Inference attempts made (successes and refusals both spend budget). */
  calls: number;
  /** True once a refusal arrived at or past the planning cap: spent. */
  spent: boolean;
  /** The request index this provider is skipped for, after an early refusal. */
  probatedFor: number;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, spent: false, probatedFor: -1 };
    providerState.set(providerId, state);
  }
  return state;
}

function budgetFloor(providerId: string): number {
  return BUDGET_FLOOR[providerId] ?? 4;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (ENV_URL === '') {
    catalog = { providers: [] };
    return;
  }
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The free models across the providers that answered, in catalog order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best rating first. An
 * OpenAI JSON-mode request may only go to a wire that can carry
 * response_format (OpenAI or Gemini): an Anthropic-wire provider echoes
 * JSON mode absent, which would lose the request.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', needsJson: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (needsJson && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/**
 * The attempt order for one request: candidates re-ranked by what the
 * trial has shown so far. Tier 0 is a provider inside its planning cap
 * that has not refused recently; tier 1 is one over its cap or skipped
 * for this request after an early refusal; tier 2 is one treated as
 * spent. Inside a tier the rating order stands. Providers already tried
 * for this request are left out by the caller.
 */
function orderCandidates(candidates: Candidate[], triedProviders: Set<string>): Candidate[] {
  const scored = candidates
    .filter((c) => !triedProviders.has(c.probe.provider.id))
    .map((c) => {
      const state = stateOf(c.probe.provider.id);
      let tier = 0;
      if (state.spent) tier = 2;
      else if (state.calls >= budgetFloor(c.probe.provider.id) || state.probatedFor === requestIndex) tier = 1;
      return { candidate: c, tier };
    });
  scored.sort((a, b) => {
    if (a.tier !== b.tier) return a.tier - b.tier;
    if (b.candidate.rating !== a.candidate.rating) return b.candidate.rating - a.candidate.rating;
    if (a.candidate.probe.provider.id !== b.candidate.probe.provider.id) {
      return a.candidate.probe.provider.id < b.candidate.probe.provider.id ? -1 : 1;
    }
    return a.candidate.model.id < b.candidate.model.id ? -1 : a.candidate.model.id > b.candidate.model.id ? 1 : 0;
  });
  return scored.map((s) => s.candidate);
}

/** Record one provider attempt, and what its refusal says about the provider. */
function recordAttempt(providerId: string, failed: boolean): void {
  const state = stateOf(providerId);
  state.calls += 1;
  if (!failed) return;
  if (state.calls >= budgetFloor(providerId)) {
    // Refused at or past the planning cap: the budget is plausibly spent.
    state.spent = true;
  } else {
    // Refused early: a scheduled refusal, or an outage starting. Skip
    // this provider for the next request only, then try it again.
    state.probatedFor = requestIndex + 1;
  }
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body, clientApi);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  let tools = false;
  if (clientApi === 'openai') {
    tools = Array.isArray(body.tools) && body.tools.length > 0;
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image_url') vision = true;
      }
    }
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    return { tools, vision, json: format === 'json_object' || format === 'json_schema' };
  }
  tools = Array.isArray(body.tools) && body.tools.length > 0;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && block.type === 'image') vision = true;
    }
  }
  return { tools, vision, json: false };
}

/** Whether an OpenAI client request asks for JSON mode. */
function needsJsonMode(body: Json, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi !== 'openai') return false;
  const type = isRecord(body.response_format) ? body.response_format.type : undefined;
  return type === 'json_object' || type === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
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

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ---------------------------------------------------------------------------
// Upstream calls: one request in a provider's wire, one answer back.
// ---------------------------------------------------------------------------

/** The upstream request body for a client body on a provider's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', providerWire: 'openai' | 'anthropic' | 'gemini', body: Json, model: string): Json {
  if (providerWire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (providerWire === 'anthropic') {
    if (clientApi === 'anthropic') return { ...body, model };
    return openAIToAnthropic(body, model);
  }
  return clientApi === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model);
}

/** The URL and headers for one provider call. */
function upstreamTarget(candidate: Candidate, stream: boolean): { url: string; headers: Record<string, string> } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, headers };
  }
  if (provider.wire === 'anthropic') {
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers };
  }
  const path = stream ? ':streamGenerateContent' : ':generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${path}${stream ? '?alt=sse' : ''}`;
  return { url, headers };
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  /** The provider's answer, buffered as text (cross-wire and JSON). */
  text: string | null;
  /** The live response, for a same-wire stream passed through untouched. */
  response: Response | null;
}

/** One inference call to a provider. A same-wire stream stays live for pass-through. */
async function callProvider(candidate: Candidate, body: Json, stream: boolean, passthrough: boolean): Promise<UpstreamResult> {
  const target = upstreamTarget(candidate, stream);
  let res: Response;
  try {
    res = await fetch(target.url, {
      method: 'POST',
      headers: target.headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, text: null, response: null };
  }
  if (!res.ok) {
    try {
      await res.arrayBuffer();
    } catch {
      // The error body is unreadable: the status is what matters.
    }
    return { ok: false, status: res.status, text: null, response: null };
  }
  if (passthrough && stream && res.body !== null) return { ok: true, status: res.status, text: null, response: res };
  const text = await res.text();
  return { ok: true, status: res.status, text, response: null };
}

/** Copy a live provider stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** A buffered provider answer as a normalized answer, or null when unparsable. */
function normalizeAnswer(providerWire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): NormalizedAnswer | null {
  if (providerWire === 'openai') {
    if (stream) return parseOpenAIStream(text);
    try {
      const value: unknown = JSON.parse(text);
      return isRecord(value) ? parseOpenAIJson(value) : null;
    } catch {
      return null;
    }
  }
  if (providerWire === 'anthropic') {
    if (stream) return parseAnthropicStream(text);
    try {
      const value: unknown = JSON.parse(text);
      return isRecord(value) ? parseAnthropicJson(value) : null;
    } catch {
      return null;
    }
  }
  if (stream) return parseGeminiStream(text);
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? parseGeminiJson(value) : null;
  } catch {
    return null;
  }
}

/**
 * Serve one client request: try the candidates in planned order, failing
 * over to the next-best free model when a provider refuses, and answer in
 * the client's wire. A same-wire answer passes through byte for byte; a
 * cross-wire one is normalized and re-rendered.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const stream = body.stream === true;
  requestIndex = manifestIndex;
  const requires = requirementsFor(body, clientApi);
  const candidates = candidatesFor(requires, clientApi, needsJsonMode(body, clientApi));
  if (candidates.length === 0) {
    const type = clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model';
    return sendError(res, clientApi, 503, type, 'no free model on the directory meets this request');
  }
  const triedProviders = new Set<string>();
  let attempts = 0;
  while (attempts < MAX_ATTEMPTS) {
    const ordered = orderCandidates(candidates, triedProviders);
    if (ordered.length === 0) break;
    const candidate = ordered[0];
    const providerId = candidate.probe.provider.id;
    triedProviders.add(providerId);
    attempts += 1;
    const providerWire = candidate.probe.provider.wire;
    const sameWire = providerWire === clientApi;
    const upBody = upstreamBody(clientApi, providerWire, body, candidate.model.id);
    let result: UpstreamResult;
    try {
      result = await callProvider(candidate, upBody, stream, sameWire);
    } catch {
      recordAttempt(providerId, true);
      continue;
    }
    if (!result.ok) {
      recordAttempt(providerId, true);
      continue;
    }
    recordAttempt(providerId, false);
    if (result.response !== null && result.response.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(result.response.body, res);
      return;
    }
    if (result.text === null) continue;
    if (sameWire && !stream) {
      // The provider's JSON is already a well-formed answer in this wire.
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(result.text) });
      res.end(result.text);
      return;
    }
    const answer = normalizeAnswer(providerWire, stream, result.text);
    if (answer === null) continue;
    if (clientApi === 'openai') {
      if (stream) {
        const text = renderOpenAIStream(answer, candidate.model.id);
        res.writeHead(200, {
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
          'content-length': Buffer.byteLength(text),
        });
        res.end(text);
      } else {
        sendJson(res, 200, renderOpenAIJson(answer, candidate.model.id));
      }
    } else if (stream) {
      const text = renderAnthropicStream(answer, candidate.model.id);
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        'content-length': Buffer.byteLength(text),
      });
      res.end(text);
    } else {
      sendJson(res, 200, renderAnthropicJson(answer, candidate.model.id));
    }
    return;
  }
  // Last resort: one more try at the best candidate, in case its refusal
  // was a scheduled one and the next call number answers.
  const best = candidates[0];
  if (best !== undefined) {
    const upBody = upstreamBody(clientApi, best.probe.provider.wire, body, best.model.id);
    try {
      const result = await callProvider(best, upBody, stream, false);
      if (result.ok && result.text !== null) {
        recordAttempt(best.probe.provider.id, false);
        const answer = normalizeAnswer(best.probe.provider.wire, stream, result.text);
        if (answer !== null) {
          if (clientApi === 'openai') {
            if (stream) {
              const text = renderOpenAIStream(answer, best.model.id);
              res.writeHead(200, {
                'content-type': 'text/event-stream',
                'cache-control': 'no-cache',
                connection: 'keep-alive',
                'content-length': Buffer.byteLength(text),
              });
              res.end(text);
            } else {
              sendJson(res, 200, renderOpenAIJson(answer, best.model.id));
            }
          } else if (stream) {
            const text = renderAnthropicStream(answer, best.model.id);
            res.writeHead(200, {
              'content-type': 'text/event-stream',
              'cache-control': 'no-cache',
              connection: 'keep-alive',
              'content-length': Buffer.byteLength(text),
            });
            res.end(text);
          } else {
            sendJson(res, 200, renderAnthropicJson(answer, best.model.id));
          }
          return;
        }
      } else {
        recordAttempt(best.probe.provider.id, true);
      }
    } catch {
      recordAttempt(best.probe.provider.id, true);
    }
  }
  sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free provider that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) {
      return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    }
    await serve(req, res, 'openai', body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) {
      return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    }
    await serve(req, res, 'anthropic', body);
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  }
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
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
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print
    // the one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
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
