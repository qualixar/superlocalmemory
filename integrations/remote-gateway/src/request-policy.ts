import type { AuthorizationGrant, ConnectionGrant, PolicyResult, RequestEnvelope, VerifiedActor } from "./contracts.ts";
// RED scaffold: reject all traffic until the admission contract is implemented.
export function authorizeRequest(actor: VerifiedActor, authorization: AuthorizationGrant | null, connection: ConnectionGrant | null, resource: string, request: RequestEnvelope): PolicyResult {
  return { allowed: false, code: "NOT_IMPLEMENTED", httpStatus: 503 };
}
