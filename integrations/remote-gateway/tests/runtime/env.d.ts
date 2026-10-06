import type { RelayDO } from '../../src/relay-do.ts';
declare global {
  namespace Cloudflare { interface Env { RELAYS: DurableObjectNamespace<RelayDO>; } }
}
