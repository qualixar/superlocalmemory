export type HeaderPair = readonly [string, string];
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
// RED scaffold: wire decoding/encoding disabled until contract assertions pass.
export function decodeRelayFrame(input: string | Uint8Array): DecodeResult { return { ok: false, code: "NOT_IMPLEMENTED" }; }
export function encodeRelayFrame(frame: RelayFrame): EncodeResult { return { ok: false, code: "NOT_IMPLEMENTED" }; }
