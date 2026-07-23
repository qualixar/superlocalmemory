import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';

const root = new URL('../..', import.meta.url);
const backupUrl = new URL('src/superlocalmemory/ui/js/od-backup.js', root);

async function loadAuthMutation(fetchImpl) {
  const source = await readFile(backupUrl, 'utf8');
  const marker = '  /* ── Tiny utilities';
  const prefix = source.slice(0, source.indexOf(marker));
  const context = { fetch: fetchImpl, window: {}, Promise, Error };
  vm.runInNewContext(
    `${prefix}\n globalThis.authMutationForTest = authMutation;\n}());`,
    context,
  );
  return context.authMutationForTest;
}

function response(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

describe('backup authenticated mutations', function () {
  it('rejects non-2xx responses', async function () {
    const authMutation = await loadAuthMutation(async function (url) {
      if (url === '/internal/token') return response(200, { token: 'token-one' });
      return response(500, { detail: 'backup rejected' });
    });

    await assert.rejects(
      authMutation('/api/backup/create', 'POST'),
      /backup rejected/,
    );
  });

  it('refreshes a stale token once after an authorization rejection', async function () {
    let tokenReads = 0;
    const mutationTokens = [];
    const authMutation = await loadAuthMutation(async function (url, options = {}) {
      if (url === '/internal/token') {
        tokenReads += 1;
        return response(200, { token: `token-${tokenReads}` });
      }
      mutationTokens.push(options.headers['X-Install-Token']);
      return response(mutationTokens.length === 1 ? 403 : 200, { success: true });
    });

    const result = await authMutation('/api/backup/sync', 'POST');
    assert.equal(result.status, 200);
    assert.equal(tokenReads, 2);
    assert.deepEqual(mutationTokens, ['token-1', 'token-2']);
  });

  it('refreshes once after an empty token response', async function () {
    let tokenReads = 0;
    const authMutation = await loadAuthMutation(async function (url) {
      if (url === '/internal/token') {
        tokenReads += 1;
        return response(200, { token: tokenReads === 1 ? '' : 'fresh-token' });
      }
      return response(200, { success: true });
    });

    const result = await authMutation('/api/backup/create', 'POST');
    assert.equal(result.status, 200);
    assert.equal(tokenReads, 2);
  });
});
