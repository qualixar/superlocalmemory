import { defineConfig } from 'vitest/config';
import { cloudflareTest } from '@cloudflare/vitest-pool-workers';
export default defineConfig({
  plugins: [cloudflareTest({ main: './tests/runtime/fixture-worker.ts', remoteBindings: false,
    additionalExports: { RelayDO: 'DurableObject', RegistryDO: 'DurableObject', TokenIndexDO: 'DurableObject', BootstrapDO:'DurableObject', OwnerIndexDO:'DurableObject' },
    miniflare: { compatibilityDate: '2026-07-01', compatibilityFlags: ['nodejs_compat'], cf: false, kvNamespaces: ["OAUTH_KV"],
      durableObjects: { RELAYS: { className: 'RelayDO', useSQLite: true }, REGISTRIES: { className: 'RegistryDO', useSQLite: true }, TOKEN_INDEX: { className:'TokenIndexDO',useSQLite:true }, BOOTSTRAPS:{className:'BootstrapDO',useSQLite:true}, OWNERS:{className:'OwnerIndexDO',useSQLite:true} } } })],
  test: { include: ['tests/runtime/*.test.ts'], testTimeout: 10000 }
});
