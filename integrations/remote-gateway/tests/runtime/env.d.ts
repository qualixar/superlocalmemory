import type { RelayDO } from '../../src/relay-do.ts';
import type { RegistryDO } from '../../src/registry-do.ts';
import type { TokenIndexDO } from '../../src/token-index-do.ts';
import type { BootstrapDO } from '../../src/bootstrap-do.ts';
import type { OwnerIndexDO } from '../../src/owner-index-do.ts';
import type {DeviceIndexDO} from '../../src/device-index-do.ts';
declare global {
  namespace Cloudflare { interface Env { OAUTH_KV: KVNamespace; DEVICES:DurableObjectNamespace<DeviceIndexDO>; RELAYS: DurableObjectNamespace<RelayDO>; REGISTRIES: DurableObjectNamespace<RegistryDO>; TOKEN_INDEX:DurableObjectNamespace<TokenIndexDO>; BOOTSTRAPS:DurableObjectNamespace<BootstrapDO>; OWNERS:DurableObjectNamespace<OwnerIndexDO>; } }
}
