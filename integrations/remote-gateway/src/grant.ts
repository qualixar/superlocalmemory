import type { Scope } from "./contracts.ts";

/** Carried on frames from the relay to the laptop only; the laptop checks it before trusting any caller identity. */
export const GRANT_HEADER = "x-slm-grant";
export const GRANT_SCOPE_ORDER = ["slm:read", "slm:write", "slm:session", "slm:mesh", "slm:media"] as const;
export interface GrantInput { aid: string; ver: number; app: string; scp: readonly Scope[]; fv: boolean; }
export interface GrantFrame { cid: string; fid: string; gen: number; dl: number; }
/** A grant key as kept in storage: only ever wrapped under the deployment's secret. */
export interface StoredGrantKey { v: 1; version: number; iv: string; ct: string; }

const encoder = new TextEncoder();
export function toBase64Url(bytes: Uint8Array): string {
  let binary = ""; for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}
export function fromBase64Url(text: string): Uint8Array<ArrayBuffer> {
  if (typeof text !== "string" || !/^[A-Za-z0-9_-]*$/.test(text) || text.length % 4 === 1) throw new Error("invalid_base64url");
  const binary = atob(text.replaceAll("-", "+").replaceAll("_", "/"));
  return Uint8Array.from(binary, c => c.charCodeAt(0));
}
function hexBytes(hex: string): Uint8Array<ArrayBuffer> {
  if (typeof hex !== "string" || !/^[a-fA-F0-9]{64}$/.test(hex)) throw new Error("invalid_wrap_key");
  return new Uint8Array(hex.match(/../g)!.map(x => parseInt(x, 16)));
}
export async function importGrantKey(raw: Uint8Array): Promise<CryptoKey> {
  if (!(raw instanceof Uint8Array) || raw.byteLength !== 32) throw new Error("invalid_grant_key");
  return crypto.subtle.importKey("raw", raw as Uint8Array<ArrayBuffer>, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
}
/** Compact JSON in a fixed key order and scopes in canonical order, so the laptop can recompute the exact bytes. */
export async function signGrant(key: CryptoKey, kid: number, input: GrantInput, frame: GrantFrame): Promise<string> {
  const scp = GRANT_SCOPE_ORDER.filter(scope => input.scp.includes(scope));
  const payload = toBase64Url(encoder.encode(JSON.stringify({ v: 1, kid, cid: frame.cid, aid: input.aid, ver: input.ver, app: input.app, scp, fv: input.fv, fid: frame.fid, gen: frame.gen, dl: frame.dl })));
  const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, encoder.encode("slm-grant-v1." + payload)));
  return "v1." + payload + "." + toBase64Url(mac);
}
async function wrapper(wrapHex: string, usage: "encrypt" | "decrypt"): Promise<CryptoKey> {
  return crypto.subtle.importKey("raw", hexBytes(wrapHex), "AES-GCM", false, [usage]);
}
const aad = (version: number) => encoder.encode("slm-grant-key-v1." + version);
/** AES-256-GCM under the deployment secret; the version is authenticated so a record cannot be moved to another version. */
export async function wrapGrantKey(raw: Uint8Array, wrapHex: string, version: number): Promise<StoredGrantKey> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ct = await crypto.subtle.encrypt({ name: "AES-GCM", iv, additionalData: aad(version) }, await wrapper(wrapHex, "encrypt"), raw as Uint8Array<ArrayBuffer>);
  return { v: 1, version, iv: toBase64Url(iv), ct: toBase64Url(new Uint8Array(ct)) };
}
export async function unwrapGrantKey(stored: StoredGrantKey, wrapHex: string): Promise<Uint8Array> {
  if (!stored || stored.v !== 1 || !Number.isSafeInteger(stored.version)) throw new Error("invalid_grant_record");
  const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: fromBase64Url(stored.iv), additionalData: aad(stored.version) }, await wrapper(wrapHex, "decrypt"), fromBase64Url(stored.ct));
  return new Uint8Array(plain);
}
