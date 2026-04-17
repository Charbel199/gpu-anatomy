#!/usr/bin/env node
// Checks data/*.yaml for internal consistency. Runs as prebuild.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import yaml from 'js-yaml';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '..', 'data');
const REPO_ROOT = path.resolve(__dirname, '..', '..');

const errors = [];
const err = (msg) => errors.push(msg);

function load(name) {
  const p = path.join(DATA_DIR, name);
  if (!fs.existsSync(p)) { err(`missing data file: ${name}`); return []; }
  return yaml.load(fs.readFileSync(p, 'utf8')) ?? [];
}

const parts = load('parts.yaml');
const views = load('views.yaml');
const flows = load('flows.yaml');

const VALID_CATEGORIES = [
  'scheduling', 'compute', 'memory', 'interconnect', 'fixed-function', 'special',
];
const VALID_LATENCY = ['fast', 'mid', 'slow'];

const partIds = new Set();
for (const p of parts) {
  if (!p.id) { err(`part missing id: ${JSON.stringify(p).slice(0, 80)}`); continue; }
  if (partIds.has(p.id)) err(`duplicate part id: ${p.id}`);
  partIds.add(p.id);
  if (typeof p.number !== 'number') err(`part ${p.id}: number must be a number`);
  if (!p.name) err(`part ${p.id}: name is required`);
  if (!p.short) err(`part ${p.id}: short is required`);
  if (!Array.isArray(p.views) || p.views.length === 0) err(`part ${p.id}: views must be a non-empty array`);
  if (!VALID_CATEGORIES.includes(p.category)) err(`part ${p.id}: invalid category ${p.category}`);
  if (p.latency !== undefined && !VALID_LATENCY.includes(p.latency)) {
    err(`part ${p.id}: invalid latency ${p.latency} (expected fast|mid|slow)`);
  }
}

for (const p of parts) {
  for (const r of p.related ?? []) {
    if (!partIds.has(r)) err(`part ${p.id}: related id "${r}" does not exist`);
  }
}

const viewIds = new Set();
for (const v of views) {
  if (!v.id) { err(`view missing id`); continue; }
  if (viewIds.has(v.id)) err(`duplicate view id: ${v.id}`);
  viewIds.add(v.id);
  if (!Array.isArray(v.rows)) { err(`view ${v.id}: rows must be an array`); continue; }
  for (const row of v.rows) {
    if (!Array.isArray(row)) { err(`view ${v.id}: each row must be an array`); continue; }
    for (const id of row) {
      if (!partIds.has(id)) err(`view ${v.id}: row references unknown part "${id}"`);
    }
  }
}

for (const p of parts) {
  for (const vid of p.views ?? []) {
    if (!viewIds.has(vid)) err(`part ${p.id}: views contains unknown view "${vid}"`);
  }
  if (p.parent && !partIds.has(p.parent) && !viewIds.has(p.parent)) {
    err(`part ${p.id}: parent "${p.parent}" is neither a part nor a view`);
  }
}

const flowIds = new Set();
for (const f of flows) {
  if (!f.id) { err(`flow missing id`); continue; }
  if (flowIds.has(f.id)) err(`duplicate flow id: ${f.id}`);
  flowIds.add(f.id);
  if (!viewIds.has(f.view)) err(`flow ${f.id}: unknown view "${f.view}"`);
  if (!Array.isArray(f.steps) || f.steps.length === 0) err(`flow ${f.id}: steps must be non-empty`);
  for (const s of f.steps ?? []) {
    if (!partIds.has(s.from)) err(`flow ${f.id}: step.from "${s.from}" unknown`);
    if (!partIds.has(s.to))   err(`flow ${f.id}: step.to "${s.to}" unknown`);
  }
}

// Missing module folders are only a warning (they may be planned but not written yet).
for (const p of parts) {
  for (const mod of p.modules ?? []) {
    if (!fs.existsSync(path.join(REPO_ROOT, mod))) {
      console.warn(`[validate] ${p.id}: module folder "${mod}" not found (planned?)`);
    }
  }
}

if (errors.length) {
  console.error('\nSchema validation failed:\n');
  for (const e of errors) console.error('  ✗ ' + e);
  console.error(`\n${errors.length} error(s).\n`);
  process.exit(1);
}

console.log(`[validate] ${parts.length} parts, ${views.length} views, ${flows.length} flows - OK`);
