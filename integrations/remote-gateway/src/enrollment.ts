export type EnrollmentStatus = "requested" | "authorized" | "provisioning" | "awaiting_connector" | "verifying" | "ready" | "failed_retryable" | "failed_terminal" | "disconnecting" | "cleanup_pending" | "revoked";
export type EnrollmentEvent = "authorize" | "start_provision" | "provisioned" | "connector_arrived" | "verified" | "retryable_failure" | "terminal_failure" | "disconnect" | "cleanup_failed" | "cleanup_complete";
export interface EnrollmentSeed {
  remoteOptIn: boolean;
  ownerId: string;
  installationId: string;
  enrollmentId: string;
  profileId: string;
  idempotencyKey: string;
  payloadDigest: string;
  expiresAt: number;
}
export interface EnrollmentRecord extends Omit<EnrollmentSeed, "remoteOptIn"> {
  status: EnrollmentStatus;
  version: number;
  connectorGeneration: number;
}
export interface EnrollmentContext {
  ownerId: string;
  installationId: string;
  expectedVersion: number;
  connectorGeneration: number;
  now: number;
}
export type EnrollmentResult = { ok: true; enrollment: EnrollmentRecord } | { ok: false; code: string };
const transitions = new Map<EnrollmentStatus, Readonly<Partial<Record<EnrollmentEvent, EnrollmentStatus>>>>([
  ["requested", { authorize: "authorized", terminal_failure: "failed_terminal", disconnect: "disconnecting" }],
  ["authorized", { start_provision: "provisioning", terminal_failure: "failed_terminal", disconnect: "disconnecting" }],
  ["provisioning", { provisioned: "awaiting_connector", retryable_failure: "failed_retryable", terminal_failure: "failed_terminal", disconnect: "disconnecting" }],
  ["awaiting_connector", { connector_arrived: "verifying", retryable_failure: "failed_retryable", terminal_failure: "failed_terminal", disconnect: "disconnecting" }],
  ["verifying", { verified: "ready", retryable_failure: "failed_retryable", terminal_failure: "failed_terminal", disconnect: "disconnecting" }],
  ["ready", { disconnect: "disconnecting" }],
  ["failed_retryable", { start_provision: "provisioning", terminal_failure: "failed_terminal", disconnect: "disconnecting" }],
  ["failed_terminal", { disconnect: "disconnecting" }],
  ["disconnecting", { cleanup_failed: "cleanup_pending", cleanup_complete: "revoked" }],
  ["cleanup_pending", { cleanup_complete: "revoked" }],
  ["revoked", {}],
]);
function identifier(value: string): boolean { return typeof value === "string" && value.length > 0 && value.length <= 256 && value.trim() === value; }
function timestamp(value: number): boolean { return Number.isSafeInteger(value) && value >= 0; }
function validSeed(seed: Omit<EnrollmentSeed, "remoteOptIn">): boolean {
  return [seed.ownerId, seed.installationId, seed.enrollmentId, seed.profileId, seed.idempotencyKey].every(identifier) &&
    typeof seed.payloadDigest === "string" && /^[a-f0-9]{64}$/.test(seed.payloadDigest) && timestamp(seed.expiresAt);
}
function denied(code: string): EnrollmentResult { return { ok: false, code }; }

/** Inputs are server-generated identity/digest metadata after authenticated UI opt-in.
 * No provider/credential side effects. The future registry must persist this result
 * with compare-and-swap, a resource journal and validated callback evidence.
 */
export function createEnrollment(seed: EnrollmentSeed, verifiedOwnerId: string, now: number): EnrollmentResult {
  if (seed.remoteOptIn !== true) return denied("OPT_IN_REQUIRED");
  if (seed.ownerId !== verifiedOwnerId) return denied("OWNER_MISMATCH");
  if (!validSeed(seed) || !timestamp(now) || seed.expiresAt <= now) return denied("INVALID_ENROLLMENT");
  // Pick known fields rather than copying an input that could contain credentials.
  return { ok: true, enrollment: Object.freeze({ ownerId: seed.ownerId, installationId: seed.installationId, enrollmentId: seed.enrollmentId,
    profileId: seed.profileId, idempotencyKey: seed.idempotencyKey, payloadDigest: seed.payloadDigest, expiresAt: seed.expiresAt,
    status: "requested", version: 1, connectorGeneration: 1 }) };
}

/** Events must originate from validated control-plane operations, not caller-supplied
 * UI status fields. A transition is not proof of connector identity or live provisioning.
 */
export function transitionEnrollment(record: EnrollmentRecord, event: EnrollmentEvent, context: EnrollmentContext): EnrollmentResult {
  if (!validSeed(record) || !timestamp(context.now) || !Number.isSafeInteger(record.version) || record.version < 1 || record.version >= Number.MAX_SAFE_INTEGER ||
      !Number.isSafeInteger(record.connectorGeneration) || record.connectorGeneration < 1 || record.connectorGeneration >= Number.MAX_SAFE_INTEGER) return denied("INVALID_ENROLLMENT");
  if (record.ownerId !== context.ownerId || record.installationId !== context.installationId) return denied("BINDING_MISMATCH");
  if (record.version !== context.expectedVersion) return denied("VERSION_CONFLICT");
  if (record.connectorGeneration !== context.connectorGeneration) return denied("GENERATION_CONFLICT");
  const table = transitions.get(record.status);
  const status = table && Object.hasOwn(table, event) ? table[event] : undefined;
  if (status === undefined) return denied("INVALID_TRANSITION");
  const cleanup = event === "disconnect" || event === "cleanup_failed" || event === "cleanup_complete";
  if (!cleanup && context.now >= record.expiresAt) return denied("ENROLLMENT_EXPIRED");
  return { ok: true, enrollment: Object.freeze({ ownerId: record.ownerId, installationId: record.installationId, enrollmentId: record.enrollmentId,
    profileId: record.profileId, idempotencyKey: record.idempotencyKey, payloadDigest: record.payloadDigest, expiresAt: record.expiresAt,
    status, version: record.version + 1, connectorGeneration: record.connectorGeneration + (event === "disconnect" ? 1 : 0) }) };
}
