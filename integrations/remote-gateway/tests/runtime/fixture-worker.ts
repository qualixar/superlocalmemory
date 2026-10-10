// Test-only entrypoint. Never deploy; production OAuth/control plane is separate.
import { RelayDO } from '../../src/relay-do.ts';
import { RegistryDO } from '../../src/registry-do.ts';
export { RelayDO, RegistryDO };
/** A relay deployed without the key-wrapping secret. */
export class RelayNoWrapKeyDO extends RelayDO { constructor(ctx: DurableObjectState, env: Cloudflare.Env) { super(ctx, { ...env, GRANT_WRAP_KEY: '' } as Cloudflare.Env); } }
/** A registry with a generous daily allowance and a tiny mesh poll budget, so the limits can be reached in a test. */
export class RegistryTunedDO extends RegistryDO { constructor(ctx: DurableObjectState, env: Record<string, unknown>) { super(ctx, { ...env, DAILY_TOOL_CALL_LIMIT: '1000', MESH_POLL_DAILY_LIMIT: '2' }); } }
export { TokenIndexDO } from '../../src/token-index-do.ts';
export { BootstrapDO } from '../../src/bootstrap-do.ts';
export { OwnerIndexDO } from '../../src/owner-index-do.ts';
export default { fetch() { return new Response('runtime fixture only', {status:404}); } };

export { DeviceIndexDO } from '../../src/device-index-do.ts';
