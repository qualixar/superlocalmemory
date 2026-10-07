// Test-only entrypoint. Never deploy; production OAuth/control plane is separate.
export { RelayDO } from '../../src/relay-do.ts';
export { RegistryDO } from '../../src/registry-do.ts';
export { TokenIndexDO } from '../../src/token-index-do.ts';
export { BootstrapDO } from '../../src/bootstrap-do.ts';
export { OwnerIndexDO } from '../../src/owner-index-do.ts';
export default { fetch() { return new Response('runtime fixture only', {status:404}); } };
