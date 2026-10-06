import type { AuthorizationGrant, ConnectionGrant, EffectiveGrant, PolicyResult, RequestEnvelope, Scope, VerifiedActor } from "./contracts.ts";

const scopes: readonly Scope[] = ["slm:read", "slm:write", "slm:session"];
const toolScopes = new Map<string, Scope>([
  ["recall", "slm:read"], ["search", "slm:read"], ["fetch", "slm:read"], ["get_status", "slm:read"],
  ["remember", "slm:write"], ["session_init", "slm:session"], ["close_session", "slm:session"],
  ["report_feedback", "slm:session"], ["report_outcome", "slm:session"],
]);
const methods = new Map<string, ReadonlySet<string>>([
  ["legacy", new Set(["initialize", "notifications/initialized", "ping", "tools/list", "tools/call"])],
  ["modern-2026-07-28", new Set(["server/discover", "tools/list", "tools/call"])],
]);
function denied(code: string, httpStatus = 403): PolicyResult { return { allowed: false, code, httpStatus }; }
function nonempty(value: string): boolean { return typeof value === "string" && value.trim().length > 0; }

/** Domain policy only. Caller must validate tokens, schemas, wire-era metadata and
 * obtain current registry snapshots before calling; never construct actor from RPC arguments.
 * Origin argument binding, response filtering and atomic registry admission remain separate gates.
 */
export function authorizeRequest(actor: VerifiedActor, authorization: AuthorizationGrant | null, connection: ConnectionGrant | null, resource: string, request: RequestEnvelope): PolicyResult {
  if (![actor.ownerId, actor.clientId, actor.authorizationId, actor.connectionId, resource].every(nonempty) ||
      actor.audience !== resource || (authorization !== null && authorization.audience !== resource)) {
    return denied("INVALID_PRINCIPAL", 401);
  }
  if (!authorization || !connection || actor.ownerId !== authorization.ownerId || actor.ownerId !== connection.ownerId ||
      actor.clientId !== authorization.clientId || actor.authorizationId !== authorization.authorizationId ||
      actor.connectionId !== authorization.connectionId || actor.connectionId !== connection.connectionId) {
    return denied("BINDING_MISMATCH");
  }
  if (authorization.revokedAt !== null || connection.revokedAt !== null) return denied("REVOKED");
  if (!methods.get(request.era)?.has(request.rpcMethod)) return denied("METHOD_DENIED");
  const allowedScopes = scopes.filter(scope => actor.scopes.includes(scope) && authorization.consentedScopes.includes(scope));
  const allowedTools = [...new Set(authorization.consentedTools)].filter(tool => {
    const scope = toolScopes.get(tool);
    return scope !== undefined && allowedScopes.includes(scope) && connection.allowedTools.includes(tool);
  });
  const grant: EffectiveGrant = {
    connection: Object.freeze({ ...connection, allowedTools: Object.freeze([...connection.allowedTools]) }),
    authorization: Object.freeze({ ...authorization, consentedTools: Object.freeze([...authorization.consentedTools]), consentedScopes: Object.freeze([...authorization.consentedScopes]),
      ...(authorization.providerGrantRef ? { providerGrantRef: Object.freeze({ ...authorization.providerGrantRef }) } : {}) }),
    allowedScopes: Object.freeze(allowedScopes), allowedTools: Object.freeze(allowedTools),
    allowCorrection: authorization.consentedCorrection && connection.allowCorrection,
    allowSharedRead: authorization.consentedSharedRead && connection.allowSharedRead,
    allowGlobalRead: authorization.consentedGlobalRead && connection.allowGlobalRead,
  };
  if (request.rpcMethod === "tools/call") {
    const scope = request.toolName === undefined ? undefined : toolScopes.get(request.toolName);
    if (scope === undefined || !authorization.consentedTools.includes(request.toolName!) || !connection.allowedTools.includes(request.toolName!)) return denied("TOOL_DENIED");
    if (!allowedScopes.includes(scope)) return denied("INSUFFICIENT_SCOPE");
    const args = request.arguments ?? {};
    if (Object.hasOwn(args, "profile_id") && (typeof args.profile_id !== "string" || (args.profile_id !== "" && args.profile_id !== connection.profileId))) return denied("PROFILE_DENIED");
    for (const [key, allowed] of [["include_shared", grant.allowSharedRead], ["include_global", grant.allowGlobalRead]] as const) {
      if (Object.hasOwn(args, key) && (typeof args[key] !== "boolean" || (args[key] === true && !allowed))) return denied("SHARING_DENIED");
    }
    if (request.toolName === "remember") {
      if (Object.hasOwn(args, "scope") && args.scope !== "personal") return denied("SHARING_DENIED");
      if (Object.hasOwn(args, "replaces") && args.replaces !== null && args.replaces !== "" && !grant.allowCorrection) return denied("CORRECTION_DENIED");
    }
  }
  return { allowed: true, grant: Object.freeze(grant) };
}
