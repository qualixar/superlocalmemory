import { GRANT_SCOPE_ORDER } from "./grant.ts";
import type { AuthorizationGrant, ConnectionGrant, EffectiveGrant, PolicyResult, RequestEnvelope, Scope, VerifiedActor } from "./contracts.ts";

/** Every tool the gateway can expose and ALL the scopes it needs. The registry, the policy and the consent page read this one table. */
export const TOOL_SCOPES: ReadonlyMap<string, readonly Scope[]> = new Map<string, readonly Scope[]>([
  ["recall", ["slm:read"]], ["search", ["slm:read"]], ["fetch", ["slm:read"]], ["get_status", ["slm:read"]],
  ["remember", ["slm:write"]], ["session_init", ["slm:session"]], ["close_session", ["slm:session"]],
  ["report_feedback", ["slm:session"]], ["report_outcome", ["slm:session"]],
  ["mesh_peers", ["slm:mesh"]], ["mesh_send", ["slm:mesh"]], ["mesh_inbox", ["slm:mesh"]], ["mesh_wait", ["slm:mesh"]], ["mesh_state", ["slm:mesh"]],
  ["get_media", ["slm:media"]], ["media_status", ["slm:media"]],
  ["remember_media", ["slm:write", "slm:media"]], ["remember_document", ["slm:write", "slm:media"]],
  ["media_upload_link", ["slm:write", "slm:media"]],
]);
/** Mesh and media tools are limited by the laptop (its remote key must opt in), not by the connection's tool list: that list is fixed
 * when the computer is linked and cannot grow later without a conflict. The consent and the token scopes are still checked here. */
export const LAPTOP_GATED_TOOLS: ReadonlySet<string> = new Set([...TOOL_SCOPES].filter(([, needed]) => needed.some(scope => scope === "slm:mesh" || scope === "slm:media")).map(([tool]) => tool));
/** The tools a consent holding exactly these scopes may use, in table order. */
export function toolsForScopes(held: readonly string[]): string[] {
  return [...TOOL_SCOPES].filter(([, needed]) => needed.every(scope => held.includes(scope))).map(([tool]) => tool);
}
const methods = new Map<string, ReadonlySet<string>>([
  ["legacy", new Set(["initialize", "notifications/initialized", "ping", "tools/list", "tools/call"])],
  ["modern-2026-07-28", new Set(["server/discover", "tools/list", "tools/call"])],
]);
function denied(code: string, httpStatus = 403): PolicyResult { return { allowed: false, code, httpStatus }; }
function listed(connection: ConnectionGrant, tool: string): boolean { return LAPTOP_GATED_TOOLS.has(tool) || connection.allowedTools.includes(tool); }
/** Argument limits for tools whose arguments the laptop must not be trusted to bound alone. */
function argumentsRefused(tool: string, args: Readonly<Record<string, unknown>>): boolean {
  if (tool === "mesh_state") return (Object.hasOwn(args, "action") && args.action !== "get") || typeof args.key !== "string" || args.key.length < 1 || args.key.length > 256;
  if (tool === "mesh_wait") return Object.hasOwn(args, "timeout_s") && !(Number.isSafeInteger(args.timeout_s) && (args.timeout_s as number) >= 1 && (args.timeout_s as number) <= 20);
  if (tool === "remember_media" || tool === "remember_document") return args.path !== undefined && args.path !== null && args.path !== "";
  return false;
}
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
  const allowedScopes = GRANT_SCOPE_ORDER.filter(scope => actor.scopes.includes(scope) && authorization.consentedScopes.includes(scope));
  const allowedTools = [...new Set(authorization.consentedTools)].filter(tool => {
    const needed = TOOL_SCOPES.get(tool);
    return needed !== undefined && needed.every(scope => allowedScopes.includes(scope)) && listed(connection, tool);
  });
  const grant: EffectiveGrant = {
    connection: Object.freeze({ ...connection, origin:Object.freeze({...connection.origin}), allowedTools: Object.freeze([...connection.allowedTools]) }),
    authorization: Object.freeze({ ...authorization, consentedTools: Object.freeze([...authorization.consentedTools]), consentedScopes: Object.freeze([...authorization.consentedScopes]),
      ...(authorization.providerGrantRef ? { providerGrantRef: Object.freeze({ ...authorization.providerGrantRef }) } : {}) }),
    allowedScopes: Object.freeze(allowedScopes), allowedTools: Object.freeze(allowedTools),
    allowCorrection: authorization.consentedCorrection && connection.allowCorrection,
    allowSharedRead: authorization.consentedSharedRead && connection.allowSharedRead,
    allowGlobalRead: authorization.consentedGlobalRead && connection.allowGlobalRead,
  };
  if (request.rpcMethod === "tools/call") {
    const needed = request.toolName === undefined ? undefined : TOOL_SCOPES.get(request.toolName);
    if (needed === undefined || !authorization.consentedTools.includes(request.toolName!) || !listed(connection, request.toolName!)) return denied("TOOL_DENIED");
    if (!needed.every(scope => allowedScopes.includes(scope))) return denied("INSUFFICIENT_SCOPE");
    const args = request.arguments ?? {};
    if (Object.hasOwn(args, "profile_id") && (typeof args.profile_id !== "string" || (args.profile_id !== "" && args.profile_id !== connection.profileId))) return denied("PROFILE_DENIED");
    for (const [key, allowed] of [["include_shared", grant.allowSharedRead], ["include_global", grant.allowGlobalRead]] as const) {
      if (Object.hasOwn(args, key) && (typeof args[key] !== "boolean" || (args[key] === true && !allowed))) return denied("SHARING_DENIED");
    }
    if (argumentsRefused(request.toolName!, args)) return denied("ARGUMENT_DENIED");
    if (request.toolName === "remember") {
      if (Object.hasOwn(args, "scope") && args.scope !== "personal") return denied("SHARING_DENIED");
      if (Object.hasOwn(args, "replaces") && args.replaces !== null && args.replaces !== "" && !grant.allowCorrection) return denied("CORRECTION_DENIED");
    }
  }
  return { allowed: true, grant: Object.freeze(grant) };
}
