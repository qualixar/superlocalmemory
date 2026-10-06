export type HeaderPair = readonly [string, string];
/** Supplied by gateway schema validation, never a wire frame or caller URL. */
export interface RelayCodecOptions { requestParamHeaders?: readonly string[]; }
interface FrameIdentity { v: 1; id: string; generation: number; }
export type RelayFrame =
  | (FrameIdentity & { kind: "request"; deadlineAt: number; headers: readonly HeaderPair[]; bodyBase64: string })
  | (FrameIdentity & { kind: "response"; status: number; headers: readonly HeaderPair[]; bodyBase64: string })
  | (FrameIdentity & { kind: "cancel" });
export type DecodeResult = { ok: true; frame: RelayFrame } | { ok: false; code: string };
export type EncodeResult = { ok: true; text: string } | { ok: false; code: string };
export const MAX_FRAME_BYTES = 8 * 1024 * 1024;
export const MAX_REQUEST_BYTES = 1024 * 1024;
export const MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const requestHeaders = new Set(["content-type", "accept", "mcp-protocol-version", "mcp-method", "mcp-name"]);
const responseHeaders = new Set(["content-type", "mcp-protocol-version", "retry-after"]);
const identityKeys = ["v", "kind", "id", "generation"];
function refused(code = "INVALID_FRAME"): { ok: false; code: string } { return { ok: false, code }; }
function integer(value: unknown, minimum: number): value is number { return typeof value === "number" && Number.isSafeInteger(value) && value >= minimum; }
function keysMatch(frame: Record<string, unknown>, expected: readonly string[]): boolean {
  const keys = Object.keys(frame);
  return keys.length === expected.length && expected.every(key => Object.hasOwn(frame, key));
}
function validHeaders(value: unknown, allowed: ReadonlySet<string>): value is HeaderPair[] {
  if (!Array.isArray(value) || value.length > 32) return false;
  const seen = new Set<string>();
  return value.every(pair => {
    if (!Array.isArray(pair) || pair.length !== 2 || typeof pair[0] !== "string" || typeof pair[1] !== "string") return false;
    const name = pair[0].toLowerCase();
    if (!allowed.has(name) || seen.has(name) || pair[1].length > 8192 || /[^\x20-\x7e]/.test(pair[1])) return false;
    seen.add(name);
    return true;
  });
}
function validBase64(value: unknown): value is string {
  if (typeof value !== "string" || value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) return false;
  if (value === "") return true;
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  // Canonical pad bits without decoding/copying a multi-megabyte body or using
  // a nested repeated regex, which can overflow the engine's regexp stack.
  if (value.endsWith("==")) return (alphabet.indexOf(value[value.length - 3]!) & 15) === 0;
  if (value.endsWith("=")) return (alphabet.indexOf(value[value.length - 2]!) & 3) === 0;
  return true;
}

/** Private wire contract v1 uses exact compact JSON.stringify encoding and padded
 * canonical base64. This is separate from public MCP JSON: original MCP bytes are
 * carried unchanged. Socket/auth/generation/deadline admission is a runtime concern.
 */
export function decodeRelayFrame(input: string | Uint8Array, options: RelayCodecOptions = {}): DecodeResult {
  if (options === null || typeof options !== "object") return refused();
  const params = options.requestParamHeaders ?? [];
  if (!Array.isArray(params) || params.length > 32 || params.some(name => typeof name !== "string" || !/^mcp-param-[a-z0-9_.-]{1,64}$/i.test(name))) return refused();
  const approvedRequestHeaders = new Set([...requestHeaders, ...params.map(name => name.toLowerCase())]);
  let text: string;
  if (typeof input === "string") {
    if (input.length > MAX_FRAME_BYTES || new TextEncoder().encode(input).byteLength > MAX_FRAME_BYTES) return refused("FRAME_TOO_LARGE");
    text = input;
  } else {
    if (!(input instanceof Uint8Array)) return refused();
    if (input.byteLength > MAX_FRAME_BYTES) return refused("FRAME_TOO_LARGE");
    try { text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(input); } catch { return refused("INVALID_UTF8"); }
  }
  let value: unknown;
  try { value = JSON.parse(text); } catch { return refused(); }
  if (value === null || typeof value !== "object" || Array.isArray(value)) return refused();
  // Reject duplicate keys and noncanonical representations rather than allowing
  // different parsers/routing stages to disagree about the same frame identity.
  const frame = value as Record<string, unknown>;
  if (frame.v !== 1 || typeof frame.id !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(frame.id) || !integer(frame.generation, 1)) return refused();
  if (frame.kind === "cancel") {
    if (!keysMatch(frame, identityKeys)) return refused();
    if (JSON.stringify(value) !== text) return refused("NON_CANONICAL_FRAME");
    return { ok: true, frame: Object.freeze({ v: 1, kind: "cancel", id: frame.id, generation: frame.generation }) };
  }
  if (frame.kind !== "request" && frame.kind !== "response") return refused();
  const request = frame.kind === "request";
  const specific = request ? "deadlineAt" : "status";
  if (!keysMatch(frame, [...identityKeys, specific, "headers", "bodyBase64"]) ||
      !integer(frame[specific], request ? 0 : 200) || (!request && (frame.status as number) > 599) ||
      !validHeaders(frame.headers, request ? approvedRequestHeaders : responseHeaders) || !validBase64(frame.bodyBase64)) return refused();
  // Known fields now have bounded shallow types, so canonical serialization
  // cannot recurse through attacker-controlled nested structures.
  if (JSON.stringify(value) !== text) return refused("NON_CANONICAL_FRAME");
  const decodedBytes = (frame.bodyBase64.length / 4) * 3 - (frame.bodyBase64.endsWith("==") ? 2 : frame.bodyBase64.endsWith("=") ? 1 : 0);
  if (decodedBytes > (request ? MAX_REQUEST_BYTES : MAX_RESPONSE_BYTES)) return refused("BODY_TOO_LARGE");
  const headers = Object.freeze(frame.headers.map(pair => Object.freeze([pair[0], pair[1]] as const)));
  const common = { v: 1 as const, id: frame.id, generation: frame.generation, headers, bodyBase64: frame.bodyBase64 };
  return request
    ? { ok: true, frame: Object.freeze({ ...common, kind: "request", deadlineAt: frame.deadlineAt as number }) }
    : { ok: true, frame: Object.freeze({ ...common, kind: "response", status: frame.status as number }) };
}

/** Encode only validated frames; never echo a parser error or credential value. */
export function encodeRelayFrame(frame: RelayFrame, options: RelayCodecOptions = {}): EncodeResult {
  let text: string;
  try { text = JSON.stringify(frame); } catch { return refused(); }
  if (typeof text !== "string") return refused();
  const validation = decodeRelayFrame(text, options);
  return validation.ok ? { ok: true, text } : validation;
}
