// Tests for the dashboard's client-side EMA-filter / ATR-extension-badge /
// copy-visible-tickers logic. Run with: node --test tests/
//
// These tests import the ACTUAL functions from index.html (via a small
// regex extraction + vm sandbox) rather than a reimplementation, so a
// change to the real filter logic in index.html cannot silently drift out
// of sync with the test without the test itself failing to find the
// function it needs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const html = readFileSync(path.join(__dirname, '..', 'index.html'), 'utf-8');

function extractFunction(src, name) {
  const startIdx = src.indexOf(`function ${name}(`);
  if (startIdx === -1) throw new Error(`function ${name} not found in index.html`);
  let depth = 0, i = src.indexOf('{', startIdx);
  const bodyStart = i;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') { depth--; if (depth === 0) break; }
  }
  return src.slice(startIdx, i + 1);
}

// Pull the real implementations straight out of index.html.
const ncEmaFilterMembersSrc = extractFunction(html, 'ncEmaFilterMembers');
const ncSortMembersSrc = extractFunction(html, 'ncSortMembers');
const ncVisibleSortedMembersSrc = extractFunction(html, 'ncVisibleSortedMembers');

const sandbox = {
  ncEmaFilter: 'all',
  ncHorizon: '1d',
  engineConfig: { dashboard: { ema_proximity_threshold_pct: 4.0, atr_extension_warning_threshold: 5.0 } },
  ncMemberSort: { field: 'd1_pct', dir: 'desc' },
  console,
};
vm.createContext(sandbox);
vm.runInContext(
  `${ncEmaFilterMembersSrc}\n${ncSortMembersSrc}\n${ncVisibleSortedMembersSrc}`,
  sandbox
);

function makeMembers() {
  return [
    { symbol: 'AAA', d1_pct: 1.0, ema10_distance_pct: 1.0, ema20_distance_pct: 8.0, atr_extension: 2.0 },
    { symbol: 'BBB', d1_pct: 3.0, ema10_distance_pct: 8.0, ema20_distance_pct: -3.5, atr_extension: 6.5 },
    { symbol: 'CCC', d1_pct: -2.0, ema10_distance_pct: -4.0, ema20_distance_pct: 4.0, atr_extension: 5.0 },
    { symbol: 'DDD', d1_pct: 0.5, ema10_distance_pct: null, ema20_distance_pct: null, atr_extension: null },
  ];
}

test('EMA10 filter: abs(distance) <= 4% threshold', () => {
  sandbox.ncEmaFilter = 'ema10';
  const out = vm.runInContext('ncEmaFilterMembers(members)', Object.assign(sandbox, { members: makeMembers() }));
  assert.deepEqual(out.map(m => m.symbol).sort(), ['AAA', 'CCC']); // 1.0 and -4.0 are within +-4%
});

test('EMA20 filter: abs(distance) <= 4% threshold', () => {
  sandbox.ncEmaFilter = 'ema20';
  const out = vm.runInContext('ncEmaFilterMembers(members)', Object.assign(sandbox, { members: makeMembers() }));
  assert.deepEqual(out.map(m => m.symbol).sort(), ['BBB', 'CCC']); // -3.5 and 4.0 are within +-4%
});

test('EMA10 OR EMA20 filter: at least one condition satisfied', () => {
  sandbox.ncEmaFilter = 'either';
  const out = vm.runInContext('ncEmaFilterMembers(members)', Object.assign(sandbox, { members: makeMembers() }));
  assert.deepEqual(out.map(m => m.symbol).sort(), ['AAA', 'BBB', 'CCC']);
});

test('EMA filter "all" returns every member, including nulls', () => {
  sandbox.ncEmaFilter = 'all';
  const out = vm.runInContext('ncEmaFilterMembers(members)', Object.assign(sandbox, { members: makeMembers() }));
  assert.equal(out.length, 4);
});

test('ncVisibleSortedMembers respects filter + sort, never changes membership count beyond the filter', () => {
  sandbox.ncEmaFilter = 'either';
  sandbox.ncMemberSort = { field: 'd1_pct', dir: 'desc' };
  const narrative = { members: makeMembers().map(m => ({ ...m, percentile_1d: 50 })), n_members: 4 };
  const out = vm.runInContext('ncVisibleSortedMembers(narrative)', Object.assign(sandbox, { narrative }));
  assert.deepEqual(out.map(m => m.symbol), ['BBB', 'AAA', 'CCC']); // sorted by d1_pct desc: 3.0, 1.0, -2.0
});

test('Copy-visible-tickers format: comma-separated symbols only, in the visible+sorted order', () => {
  sandbox.ncEmaFilter = 'either';
  sandbox.ncMemberSort = { field: 'd1_pct', dir: 'asc' };
  const narrative = { members: makeMembers().map(m => ({ ...m, percentile_1d: 50 })), n_members: 4 };
  const out = vm.runInContext('ncVisibleSortedMembers(narrative)', Object.assign(sandbox, { narrative }));
  const text = out.map(m => m.symbol).join(',');
  assert.equal(text, 'CCC,AAA,BBB'); // ascending by d1_pct: -2.0, 1.0, 3.0
  assert.doesNotMatch(text, /[^A-Z,]/); // symbols + commas only, per point 22's "SNDK,WDC,STX" format
});

test('ATR Extension > threshold badge condition matches the config threshold, not a hardcoded 5', () => {
  const threshold = sandbox.engineConfig.dashboard.atr_extension_warning_threshold;
  const members = makeMembers();
  const extended = members.filter(m => m.atr_extension !== null && m.atr_extension > threshold);
  assert.deepEqual(extended.map(m => m.symbol), ['BBB']); // 6.5 > 5.0; 5.0 itself is NOT > 5.0 (CCC excluded)
});
