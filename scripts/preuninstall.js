#!/usr/bin/env node
/**
 * SuperLocalMemory NPM pre-uninstall notice.
 *
 * npm owns and removes the package directory, including its private Python
 * runtime. Durable SLM data has separate ownership and is never inspected or
 * modified by an application-code uninstall.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

console.log('SuperLocalMemory application code is being removed.');
console.log('Memory data is preserved. No database, profile, backup, or configuration was inspected or changed.');
