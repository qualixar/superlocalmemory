import { DurableObject } from "cloudflare:workers";
import type { RelayFrame, RelayCodecOptions } from "./relay-protocol.ts";
export interface RelayBinding {
  ownerId: string; connectionId: string; installationId: string; profileId: string;
  deviceDigest: string; deviceExpiresAt: number;
}
export class RelayDO extends DurableObject {
  // Private namespace/RPC scaffold; no public production Worker supplied.
  async configureBinding(_binding: RelayBinding): Promise<void> {}
  async revoke(): Promise<void> {}
  async forward(_frame: RelayFrame, _options: RelayCodecOptions = {}): Promise<Response> { return new Response('unimplemented', {status:503}); }
  async fetch(_request: Request): Promise<Response> { return new Response('unimplemented', {status:503}); }
}
