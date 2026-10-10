// Gateway domain contracts. Not token validation or a deployed gateway.
export type Scope = "slm:read" | "slm:write" | "slm:session" | "slm:mesh" | "slm:media";
export type Era = "legacy" | "modern-2026-07-28";
// Server-derived properties persisted in the authorization provider grant.
export interface AuthProps {
  ownerId: string;
  authorizationId: string;
  connectionId: string;
}
export interface VerifiedActor {
  ownerId: string;
  authorizationId: string;
  connectionId: string;
  clientId: string;
  audience: string;
  credentialKind: "oauth" | "service";
  scopes: readonly Scope[];
}
// Immutable consent ceiling; changing a connection cannot enlarge this grant.
export interface AuthorizationGrant {
  authorizationId: string;
  audience: string;
  ownerId: string;
  clientId: string;
  connectionId: string;
  consentedTools: readonly string[];
  consentedScopes: readonly Scope[];
  consentedCorrection: boolean;
  consentedSharedRead: boolean;
  consentedGlobalRead: boolean;
  // Bound once from validated provider callback; only for OAuth cleanup.
  providerGrantRef?: { userId: string; grantId: string };
  authorizationVersion: number;
  revokedAt: string | null;
}
export type OriginTransport =
  | {kind:'relay';installationId:string;profileId:string}
  | {kind:'https';url:string;credentialRef:string;exactAgentPath:string};
export interface ConnectionGrant {
  connectionId: string;
  ownerId: string;
  installationId: string;
  profileId: string;
  origin: Readonly<OriginTransport>;
  allowedTools: readonly string[];
  allowCorrection: boolean;
  allowSharedRead: boolean;
  allowGlobalRead: boolean;
  policyVersion: number;
  revokedAt: string | null;
}
export interface RequestEnvelope {
  era: Era;
  rpcMethod: string;
  toolName?: string;
  rpcId?: string | number | null;
  arguments?: Readonly<Record<string, unknown>>;
  originalBody: Uint8Array;
}
export interface EffectiveGrant {
  connection: ConnectionGrant;
  authorization: AuthorizationGrant;
  allowedTools: readonly string[];
  allowedScopes: readonly Scope[];
  allowCorrection: boolean;
  allowSharedRead: boolean;
  allowGlobalRead: boolean;
}
export type PolicyResult =
  | { allowed: true; grant: EffectiveGrant }
  | { allowed: false; code: string; httpStatus: number };
export interface GrantRegistry {
  resolve(actor: VerifiedActor): Promise<EffectiveGrant>;
  revokeAuthorization(ownerId: string, authorizationId: string, expectedVersion: number): Promise<number>;
  revoke(ownerId: string, connectionId: string, expectedVersion: number): Promise<number>;
}
export interface GatewayAudit {
  eventId: string;
  timestamp: string;
  connectionId: string;
  authorizationId: string;
  method: string;
  tool?: string;
  policyDecision: "allow" | "deny";
  outcome: "success" | "denied" | "unavailable" | "timeout" | "error";
  durationMs: number;
}
// Never add request/response bodies or credential values to GatewayAudit.

export interface IssuedTokenReference {
  tokenDigest: string;
  tokenKind: "access" | "refresh";
  authorizationId: string;
  ownerId: string;
  clientId: string;
  connectionId: string;
  audience: string;
  validUntil: string;
}
