import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const root = new URL('../..', import.meta.url);

async function source(relativePath) {
  return readFile(new URL(relativePath, root), 'utf8');
}

describe('control-plane dashboard contracts', function () {
  it('posts skill evolution configuration with the install credential', async function () {
    const skills = await source('src/superlocalmemory/ui/js/od-skills.js');
    assert.match(skills, /\/api\/evolution\/config/);
    assert.match(skills, /X-Install-Token/);
  });

  it('does not claim cloud backup connectivity before status is loaded', async function () {
    const backup = await source('src/superlocalmemory/ui/js/od-backup.js');
    const hero = backup.slice(backup.indexOf('function buildHero'), backup.indexOf('function buildKPIs'));
    assert.doesNotMatch(hero, /Personal cloud connected|> Synced/);
  });

  it('authenticates every backup mutation with the install credential', async function () {
    const backup = await source('src/superlocalmemory/ui/js/od-backup.js');
    assert.match(backup, /function authMutation/);
    assert.match(backup, /X-Install-Token/);
    assert.match(backup, /tokenPromise = null/);
    assert.match(backup, /if \(!response\.ok\)/);
    for (const endpoint of ['/api/backup/configure', '/api/backup/create', '/api/backup/sync']) {
      assert.match(backup, new RegExp(endpoint.replaceAll('/', '\\/')));
    }
  });

  it('does not make encryption, restore, or unwitnessed sync claims', async function () {
    const backup = await source('src/superlocalmemory/ui/js/od-backup.js');
    assert.doesNotMatch(backup, /AES-256|encrypted with your local key|enable encrypted cloud|every snapshot is restorable/);
    assert.doesNotMatch(backup, /restoreBtn|>Restore</);
    assert.match(backup, /Plain SQLite/);
    assert.match(backup, /last_sync_status/);
  });

  it('represents manual scheduling and destination sync truthfully', async function () {
    const backup = await source('src/superlocalmemory/ui/js/od-backup.js');
    assert.match(backup, /enabled:\s*false/);
    assert.match(backup, /Sync all/);
    assert.match(backup, /if \(forceReload\)[\s\S]*Destination status refreshed/);
    assert.match(backup, /failed \? 'danger' : allSuccessful \? 'ok'/);
    assert.match(backup, /!d\.enabled[\s\S]*'Manual only'/);
    assert.match(backup, /Local storage used/);
    assert.doesNotMatch(backup, /Cloud storage used|Connected successfully/);
    assert.doesNotMatch(backup, /destination_id:\s*destId/);
  });

  it('exports through an authenticated POST instead of a state-changing navigation', async function () {
    const backup = await source('src/superlocalmemory/ui/js/od-backup.js');
    const legacySettings = await source('src/superlocalmemory/ui/js/settings.js');
    assert.match(backup, /authMutation\('\/api\/backup\/export', 'POST'\)/);
    assert.doesNotMatch(backup, /window\.location\.href\s*=\s*'\/api\/backup\/export'/);
    assert.match(legacySettings, /fetch\('\/api\/backup\/export', \{ method: 'POST' \}\)/);
    assert.doesNotMatch(legacySettings, /window\.location\.href\s*=\s*'\/api\/backup\/export'/);
  });

  it('renders GitHub callback data through textContent rather than HTML concatenation', async function () {
    const backup = await source('src/superlocalmemory/server/routes/backup.py');
    const patForm = backup.slice(backup.indexOf('async function doConnect()'), backup.indexOf('@router.get("/api/backup/oauth/github/callback")'));
    assert.match(patForm, /textContent/);
    assert.doesNotMatch(patForm, /document\.body\.innerHTML\s*=.*data\./s);
    assert.doesNotMatch(patForm, /innerHTML\s*=.*data\.detail/s);
  });
});
