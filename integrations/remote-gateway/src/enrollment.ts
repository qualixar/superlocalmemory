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
// RED scaffold: no enrollment or transition until the contract is implemented.
export function createEnrollment(seed: EnrollmentSeed, verifiedOwnerId: string, now: number): EnrollmentResult {
  return { ok: false, code: "NOT_IMPLEMENTED" };
}
export function transitionEnrollment(record: EnrollmentRecord, event: EnrollmentEvent, context: EnrollmentContext): EnrollmentResult {
  return { ok: false, code: "NOT_IMPLEMENTED" };
}
