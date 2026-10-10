/** The gateway's half of the one-time upload link wire. Pure functions only: no I/O, no Worker types. */

/** Decoded bytes per relay request frame. A frame body may be 1 MiB; this leaves room and keeps one chunk to a quick round trip. */
export const CHUNK_BYTES = 700_000;
export const UPLOAD_HEADER = "x-slm-upload";
/** Refused before anything is relayed: no link allows more than this, whatever the laptop later says. */
export const HARD_MAX_BYTES = 1024 * 1024 * 1024;
const OPS = new Set(["info", "chunk", "finish"]);
const PATH = /^\/u\/([a-f0-9]{32})\/([A-Za-z0-9_-]{43})$/;
const TOKEN = /^[A-Za-z0-9_-]{43}$/;
const CODE = /^[a-z_]{1,40}$/;
const MESSAGE_CHARS = 300;

export type UploadOp = "info" | "chunk" | "finish";
export class UploadAbort extends Error {
  readonly code: string;
  constructor(code: string) { super(code); this.code = code; }
}
/** What the laptop said, rebuilt from known fields only. */
export interface UploadReply {
  ok: boolean; kind?: "image" | "document"; maxBytes?: number; received?: number;
  done?: boolean; message?: string; code?: string;
}

export function parseUploadPath(pathname: string): { connection: string; token: string } | null {
  const found = PATH.exec(pathname);
  return found ? { connection: found[1]!, token: found[2]! } : null;
}

export function uploadHeader(op: string, token: string, index: number, total: number): string {
  if (!OPS.has(op) || !TOKEN.test(token) || !Number.isSafeInteger(index) || index < 0 || index > 9_999_999_999 ||
      !Number.isSafeInteger(total) || total < 0 || total > 999_999_999_999) throw new Error("invalid_upload_header");
  return `${op} ${token} ${index} ${total}`;
}

function natural(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

/** Never trusts the shape: unknown keys are dropped, text is cut short, a bad code becomes "error". */
export function cleanReply(raw: unknown): UploadReply {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) return { ok: false, code: "error", message: "The reply from your computer was not understood." };
  const r = raw as Record<string, unknown>;
  const message = typeof r.message === "string" ? r.message.replace(/\s+/g, " ").slice(0, MESSAGE_CHARS) : undefined;
  if (r.ok !== true) {
    return { ok: false, code: typeof r.code === "string" && CODE.test(r.code) ? r.code : "error", message: message || "The file could not be saved." };
  }
  const kind = r.kind === "image" || r.kind === "document" ? r.kind : undefined;
  return { ok: true, kind, maxBytes: natural(r.max_bytes), received: natural(r.received), done: typeof r.done === "boolean" ? r.done : undefined, message };
}

const STATUS: Record<string, number> = { invalid_link: 404, expired: 410, used: 410, not_allowed: 403, too_large: 413, chunk_too_large: 413, daily_limit: 429, too_many_attempts: 429 };
export function statusFor(code: string): number { return STATUS[code] ?? 400; }

const PROBLEMS: Record<string, { status: number; message: string }> = {
  connector_offline: { status: 503, message: "Your computer is asleep or SuperLocalMemory is not running. Wake it, then reload this page." },
  connector_asleep: { status: 503, message: "Your computer is asleep or SuperLocalMemory is not running. Wake it, then reload this page." },
  connector_unavailable: { status: 503, message: "Your computer is not connected right now. Open SuperLocalMemory on it, then reload this page." },
  connector_closed: { status: 503, message: "The connection to your computer dropped. Reload this page and try again." },
  connector_replaced: { status: 503, message: "The connection to your computer was restarted. Reload this page and try again." },
  relay_timeout: { status: 504, message: "Your computer took too long to answer. Try again in a moment." },
  origin_timeout: { status: 504, message: "Your computer took too long to answer. Try again in a moment." },
  relay_busy: { status: 429, message: "Your computer is busy with other requests. Try again in a moment." },
  connection_revoked: { status: 403, message: "This computer is no longer connected to this app." },
};
export function relayProblem(code: string): { status: number; message: string } {
  return PROBLEMS[code] ?? { status: 502, message: "Your computer could not be reached. Try again in a moment." };
}

export function toBase64(bytes: Uint8Array): string {
  const pieces: string[] = [];
  for (let i = 0; i < bytes.length; i += 16384) pieces.push(String.fromCharCode(...bytes.subarray(i, i + 16384)));
  return btoa(pieces.join(""));
}

/** Re-cuts a body into pieces of exactly `size` bytes (the last may be shorter). Stops with `size_mismatch` if more than `limit` bytes arrive. */
export async function* chunksOf(body: ReadableStream<Uint8Array>, size: number, limit: number): AsyncGenerator<Uint8Array> {
  const reader = body.getReader();
  let buffer = new Uint8Array(size), fill = 0, total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.length;
      if (total > limit) throw new UploadAbort("size_mismatch");
      for (let offset = 0; offset < value.length;) {
        const take = Math.min(size - fill, value.length - offset);
        buffer.set(value.subarray(offset, offset + take), fill);
        fill += take; offset += take;
        if (fill === size) { yield buffer; buffer = new Uint8Array(size); fill = 0; }
      }
    }
    if (fill > 0) yield buffer.subarray(0, fill);
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}
