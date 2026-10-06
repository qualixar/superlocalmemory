import { defineConfig } from 'vitest/config';
import { cloudflareTest } from '@cloudflare/vitest-pool-workers';
export default defineConfig({
  plugins: [cloudflareTest({ main: './tests/runtime/fixture-worker.ts', remoteBindings: false,
    additionalExports: { RelayDO: 'DurableObject' },
    miniflare: { compatibilityDate: '2026-07-01', compatibilityFlags: ['nodejs_compat'], cf: false,
      durableObjects: { RELAYS: { className: 'RelayDO', useSQLite: true } } } })],
  test: { include: ['tests/runtime/*.test.ts'], testTimeout: 10000 }
});
