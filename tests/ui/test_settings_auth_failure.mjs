import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';

const root = new URL('../..', import.meta.url);
const settingsUrl = new URL(
  'src/superlocalmemory/ui/js/od-settings.js',
  root,
);

async function loadAuthRequest(fetchImpl) {
  const source = await readFile(settingsUrl, 'utf8');
  const marker = '  /* ── Helpers';
  const prefix = source.slice(0, source.indexOf(marker));
  const context = {
    fetch: fetchImpl,
    window: {},
    Promise,
    Error,
  };
  vm.runInNewContext(
    `${prefix}\n globalThis.authRequestForTest = authRequest;\n}());`,
    context,
  );
  return context.authRequestForTest;
}

function response(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  };
}

describe('settings authenticated mutations', function () {
  it('rejects a non-2xx mutation so callers cannot render Saved', async function () {
    const fetches = [];
    const authRequest = await loadAuthRequest(async function (url) {
      fetches.push(url);
      if (url === '/internal/token') return response(200, { token: 'token-one' });
      return response(500, { error: 'write rejected' });
    });

    await assert.rejects(
      authRequest('/api/v3/mesh/config', 'PUT', { enabled: false }),
      /write rejected/,
    );
    assert.equal(fetches.filter((url) => url === '/internal/token').length, 1);
  });

  it('refreshes the install token once after an authorization rejection', async function () {
    let tokenReads = 0;
    const mutationTokens = [];
    const authRequest = await loadAuthRequest(async function (url, options = {}) {
      if (url === '/internal/token') {
        tokenReads += 1;
        return response(200, { token: `token-${tokenReads}` });
      }
      mutationTokens.push(options.headers['X-Install-Token']);
      return response(mutationTokens.length === 1 ? 403 : 200, { ok: true });
    });

    const result = await authRequest(
      '/api/v3/trust/config',
      'PUT',
      { use_trust_weighting: true },
    );

    assert.equal(result.status, 200);
    assert.equal(tokenReads, 2);
    assert.deepEqual(mutationTokens, ['token-1', 'token-2']);
  });

  it('refreshes once when the token endpoint returns an empty token', async function () {
    let tokenReads = 0;
    const authRequest = await loadAuthRequest(async function (url) {
      if (url === '/internal/token') {
        tokenReads += 1;
        return response(200, { token: tokenReads === 1 ? '' : 'fresh-token' });
      }
      return response(200, { ok: true });
    });

    const result = await authRequest('/api/v3/forgetting/config', 'PUT', {
      enabled: true,
    });

    assert.equal(result.status, 200);
    assert.equal(tokenReads, 2);
  });
});
