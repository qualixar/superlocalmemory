import { readdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const testDir = join(root, 'tests', 'ui');
const files = readdirSync(testDir)
    .filter(name => /^test_.*\.mjs$/.test(name))
    .sort()
    .map(name => relative(root, join(testDir, name)));

if (files.length === 0) {
    throw new Error('No UI tests found under tests/ui');
}

const result = spawnSync(process.execPath, ['--test', ...files], {
    cwd: root,
    stdio: 'inherit',
});

if (result.error) throw result.error;
process.exit(result.status ?? 1);
