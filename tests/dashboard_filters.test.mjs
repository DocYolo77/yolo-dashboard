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
const ncNearEma10Src = extractFunction(html, 'ncNearEma10');
const ncNearEma20Src = extractFunction(html, 'ncNearEma20');
const ncIsAtrExtendedSrc = extractFunction(html, 'ncIsAtrExtended');
const ncEmaFilterMembersSrc = extractFunction(html, 'ncEmaFilterMembers');
const ncSortMembersSrc = extractFunction(html, 'ncSortMembers');
const ncMemberOpportunityStateSrc = extractFunction(html, 'ncMemberOpportunityState');
const ncStateFilterMembersSrc = extractFunction(html, 'ncStateFilterMembers');
const ncVisibleSortedMembersSrc = extractFunction(html, 'ncVisibleSortedMembers');
const ncMomentumTickerListSrc = extractFunction(html, 'ncMomentumTickerList');
const oppTabFilterSrc = extractFunction(html, 'oppTabFilter');
const oppApplyFiltersSrc = extractFunction(html, 'oppApplyFilters');
const oppSortItemsSrc = extractFunction(html, 'oppSortItems');
const oppVisibleSortedItemsSrc = extractFunction(html, 'oppVisibleSortedItems');

const sandbox = {
  ncEmaFilter: 'all',
  ncStateFilter: 'all',
  ncHorizon: '1d',
  engineConfig: { dashboard: { ema_proximity_threshold_pct: 4.0, atr_extension_warning_threshold: 5.0 } },
  ncMemberSort: { field: 'd1_pct', dir: 'desc' },
  dashboardState: null,
  oppTab: 'all',
  oppSortState: { field: 'leadership_score', dir: 'desc' },
  // Minimal DOM stub for the Opportunities filter inputs — oppApplyFilters
  // reads these directly via getElementById, same as the real page.
  document: {
    _elements: {},
    getElementById(id) {
      if (!this._elements[id]) this._elements[id] = { value: '', checked: false };
      return this._elements[id];
    },
  },
  console,
};
vm.createContext(sandbox);
vm.runInContext(
  `${ncNearEma10Src}\n${ncNearEma20Src}\n${ncIsAtrExtendedSrc}\n${ncEmaFilterMembersSrc}\n${ncSortMembersSrc}\n` +
  `${ncMemberOpportunityStateSrc}\n${ncStateFilterMembersSrc}\n${ncVisibleSortedMembersSrc}\n${ncMomentumTickerListSrc}\n` +
  `${oppTabFilterSrc}\n${oppApplyFiltersSrc}\n${oppSortItemsSrc}\n${oppVisibleSortedItemsSrc}`,
  sandbox
);

function setOppFilterInputs(overrides) {
  const defaults = {
    oppFilterNarrative: '', oppFilterStructuralRs: '', oppFilterRs: '', oppFilterThrust: '', oppFilterAtr: '', oppFilterCap: '',
  };
  const values = Object.assign({}, defaults, overrides);
  Object.entries(values).forEach(([id, value]) => { sandbox.document._elements[id] = { value }; });
  sandbox.document._elements.oppFilterNearEma = { checked: !!(overrides && overrides.oppFilterNearEma) };
}
setOppFilterInputs({});

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
  const extended = members.filter(m => vm.runInContext('ncIsAtrExtended(m, t)', Object.assign(sandbox, { m, t: threshold })));
  assert.deepEqual(extended.map(m => m.symbol), ['BBB']); // 6.5 > 5.0; 5.0 itself is NOT > 5.0 (CCC excluded)
});

test('ncMomentumTickerList: near-EMA + not-ATR-extended across all narratives, deduplicated', () => {
  const data = {
    narratives: [
      {
        id: 'n1',
        members: [
          { symbol: 'NEAR_OK', ema10_distance_pct: 1.0, ema20_distance_pct: 9.0, atr_extension: 2.0 },   // near EMA10, not extended -> included
          { symbol: 'NEAR_EXT', ema10_distance_pct: 0.5, ema20_distance_pct: 9.0, atr_extension: 7.0 },  // near EMA10 but ATR-extended -> excluded
          { symbol: 'FAR', ema10_distance_pct: 9.0, ema20_distance_pct: 9.0, atr_extension: 1.0 },       // not near either EMA -> excluded
        ],
      },
      {
        id: 'n2',
        // NEAR_OK also belongs here (multi-membership) -> must appear only once in the output.
        members: [
          { symbol: 'NEAR_OK', ema10_distance_pct: 1.0, ema20_distance_pct: 9.0, atr_extension: 2.0 },
          { symbol: 'NEAR_OK2', ema10_distance_pct: 9.0, ema20_distance_pct: -2.0, atr_extension: null }, // near EMA20, ATR unknown -> not disqualifying -> included
        ],
      },
    ],
  };
  const list = vm.runInContext(
    'ncMomentumTickerList(data, ema, atr)',
    Object.assign(sandbox, { data, ema: 4.0, atr: 5.0 })
  );
  // Array.from(): the Set/Array built inside vm.runInContext belong to the
  // sandbox's own realm (new Set()/Array.from() there resolve to the
  // sandbox's constructors), so deepEqual against a host array literal
  // needs re-materializing into the host realm first, same as the other
  // tests' `.map()` calls implicitly do for their host-created inputs.
  assert.deepEqual(Array.from(list), ['NEAR_OK', 'NEAR_OK2']); // sorted, deduplicated, NEAR_EXT and FAR excluded
});

// ── Narrative Detail View: EMA-Filter + State-Filter kombinierbar (Punkt 32) ──

function makeStateFilterFixture() {
  const members = [
    { symbol: 'AAA', ema10_distance_pct: 1.0, ema20_distance_pct: 1.0 },  // near EMA
    { symbol: 'BBB', ema10_distance_pct: 1.0, ema20_distance_pct: 1.0 },  // near EMA
    { symbol: 'CCC', ema10_distance_pct: 9.0, ema20_distance_pct: 9.0 },  // NOT near EMA
  ];
  const narrative = { id: 'n1', members: members.map(m => ({ ...m, percentile_1d: 50 })), n_members: 3 };
  const dashboardState = {
    opportunities: {
      items: [
        { symbol: 'AAA', narratives: ['n1'], quality_state: 'leader', constructive_reset_narratives: ['n1'], laggard_narratives: [], extended: false },
        { symbol: 'BBB', narratives: ['n1'], quality_state: 'neutral', constructive_reset_narratives: [], laggard_narratives: ['n1'], extended: true },
        { symbol: 'CCC', narratives: ['n1'], quality_state: 'leader', constructive_reset_narratives: ['n1'], laggard_narratives: [], extended: false },
      ],
    },
  };
  return { narrative, dashboardState };
}

test('ncVisibleSortedMembers: State-Filter allein (Leader) matcht ueber alle Mitglieder', () => {
  const { narrative, dashboardState } = makeStateFilterFixture();
  sandbox.ncEmaFilter = 'all';
  sandbox.ncStateFilter = 'leader';
  sandbox.dashboardState = dashboardState;
  const out = vm.runInContext('ncVisibleSortedMembers(narrative)', Object.assign(sandbox, { narrative }));
  assert.deepEqual(Array.from(out).map(m => m.symbol).sort(), ['AAA', 'CCC']); // BBB is neutral
});

test('ncVisibleSortedMembers: EMA-Filter + State-Filter kombiniert (Near EMA + Constructive Reset)', () => {
  const { narrative, dashboardState } = makeStateFilterFixture();
  sandbox.ncEmaFilter = 'either';       // AAA, BBB near EMA; CCC excluded
  sandbox.ncStateFilter = 'constructive_reset';  // AAA, CCC have the reset; BBB does not
  sandbox.dashboardState = dashboardState;
  const out = vm.runInContext('ncVisibleSortedMembers(narrative)', Object.assign(sandbox, { narrative }));
  // Intersection of both filters: only AAA satisfies both.
  assert.deepEqual(Array.from(out).map(m => m.symbol), ['AAA']);
  sandbox.ncEmaFilter = 'all';
  sandbox.ncStateFilter = 'all';
  sandbox.dashboardState = null;
});

// ── Opportunities: Tabs, Filter, Sortierung, Copy (Punkt 30-31) ────────

function makeOppItems() {
  return [
    { symbol: 'MU', narratives: ['memory'], quality_state: 'fresh_leader', near_emas: true, extended: false,
      constructive_reset_narratives: ['memory'], laggard_narratives: [], structural_rs: 95, trend_strength: 82,
      leadership_score: 92, rs_1w: 90, rs_1m: 88,
      thrust_percentile_1d: 90, thrust_percentile_1w: 85, ema10_distance_pct: 1.0, ema20_distance_pct: 2.0,
      atr_extension: 3.0, w1_pct: 8.0, m1_pct: 20.0, market_cap: 120e9 },
    { symbol: 'VRT', narratives: ['ai_infra'], quality_state: 'leader', near_emas: false, extended: true,
      constructive_reset_narratives: [], laggard_narratives: [], structural_rs: 90, trend_strength: 78,
      leadership_score: 88, rs_1w: 85, rs_1m: 80,
      thrust_percentile_1d: 60, thrust_percentile_1w: 55, ema10_distance_pct: 12.0, ema20_distance_pct: 15.0,
      atr_extension: 6.5, w1_pct: 15.0, m1_pct: 30.0, market_cap: 40e9 },
    { symbol: 'SNDK', narratives: ['memory'], quality_state: 'neutral', near_emas: true, extended: false,
      constructive_reset_narratives: [], laggard_narratives: ['memory'], structural_rs: 45, trend_strength: 30,
      leadership_score: 40, rs_1w: 35, rs_1m: 30,
      thrust_percentile_1d: 20, thrust_percentile_1w: 25, ema10_distance_pct: -1.5, ema20_distance_pct: -1.0,
      atr_extension: 1.0, w1_pct: -3.0, m1_pct: -5.0, market_cap: 8e9 },
    // Isolated from the memory/near-EMA/RS1W/ATR/cap fixtures above on
    // purpose, so it only shows up where a test specifically means to
    // exercise it (the 'recent' tab and the Structural RS filter below) —
    // every OTHER existing filter test's expected list stays unchanged.
    { symbol: 'WDC', narratives: ['ai_infra'], quality_state: 'recent_leader', near_emas: false, extended: false,
      constructive_reset_narratives: [], laggard_narratives: [], structural_rs: 88, trend_strength: 60,
      leadership_score: 70, rs_1w: 60, rs_1m: 65,
      thrust_percentile_1d: 50, thrust_percentile_1w: 45, ema10_distance_pct: 9.0, ema20_distance_pct: 9.0,
      atr_extension: 8.0, w1_pct: -1.0, m1_pct: 5.0, market_cap: 15e9 },
  ];
}

test('oppTabFilter: all/fresh/leaders/recent/reset/extended/laggards', () => {
  const items = makeOppItems();
  const bySymbol = tab => vm.runInContext('items.filter(it => oppTabFilter(it, tab))', Object.assign(sandbox, { items, tab })).map(it => it.symbol);
  assert.deepEqual(bySymbol('all'), ['MU', 'VRT', 'SNDK', 'WDC']);
  assert.deepEqual(bySymbol('fresh'), ['MU']);
  assert.deepEqual(bySymbol('leaders'), ['MU', 'VRT']); // fresh_leader counts as leader-like too, recent_leader does NOT
  assert.deepEqual(bySymbol('recent'), ['WDC']);
  assert.deepEqual(bySymbol('reset'), ['MU']);
  assert.deepEqual(bySymbol('extended'), ['VRT']);
  assert.deepEqual(bySymbol('laggards'), ['SNDK']);
});

test('oppApplyFilters: Narrative-Filter', () => {
  setOppFilterInputs({ oppFilterNarrative: 'memory' });
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol).sort(), ['MU', 'SNDK']);
  setOppFilterInputs({});
});

test('oppApplyFilters: Min Structural RS', () => {
  setOppFilterInputs({ oppFilterStructuralRs: '87' });
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol).sort(), ['MU', 'VRT', 'WDC']); // 95, 90, 88 pass; SNDK's 45 excluded
  setOppFilterInputs({});
});

test('oppApplyFilters: Min RS1W', () => {
  setOppFilterInputs({ oppFilterRs: '80' });
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol).sort(), ['MU', 'VRT']);
  setOppFilterInputs({});
});

test('oppApplyFilters: Max ATR Extension', () => {
  setOppFilterInputs({ oppFilterAtr: '5' });
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol).sort(), ['MU', 'SNDK']); // VRT's 6.5 excluded
  setOppFilterInputs({});
});

test('oppApplyFilters: Min Market Cap ($Mrd)', () => {
  setOppFilterInputs({ oppFilterCap: '50' }); // $50bn minimum
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol), ['MU']); // only 120bn qualifies
  setOppFilterInputs({});
});

test('oppApplyFilters: Near EMAs checkbox', () => {
  setOppFilterInputs({ oppFilterNearEma: true });
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol).sort(), ['MU', 'SNDK']);
  setOppFilterInputs({});
});

test('oppSortItems: numeric field descending/ascending', () => {
  const items = makeOppItems();
  sandbox.oppSortState = { field: 'leadership_score', dir: 'desc' };
  const desc = vm.runInContext('oppSortItems(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(desc).map(it => it.symbol), ['MU', 'VRT', 'WDC', 'SNDK']);
  sandbox.oppSortState = { field: 'leadership_score', dir: 'asc' };
  const asc = vm.runInContext('oppSortItems(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(asc).map(it => it.symbol), ['SNDK', 'WDC', 'VRT', 'MU']);
});

test('oppSortItems: sorts by structural_rs (the V1.1 default sort field)', () => {
  const items = makeOppItems();
  sandbox.oppSortState = { field: 'structural_rs', dir: 'desc' };
  const desc = vm.runInContext('oppSortItems(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(desc).map(it => it.symbol), ['MU', 'VRT', 'WDC', 'SNDK']); // 95, 90, 88, 45
});

test('oppVisibleSortedItems: combines tab + filters + sort into one visible/copyable list', () => {
  sandbox.dashboardState = { opportunities: { items: makeOppItems() } };
  sandbox.oppTab = 'all';
  sandbox.oppSortState = { field: 'symbol', dir: 'asc' };
  setOppFilterInputs({ oppFilterNearEma: true }); // MU + SNDK only
  const visible = vm.runInContext('oppVisibleSortedItems()', sandbox);
  const symbols = Array.from(visible).map(it => it.symbol);
  assert.deepEqual(symbols, ['MU', 'SNDK']); // near-EMA filter applied, alphabetically sorted
  // No duplicates even if a stock were double-counted upstream (Copy-Funktion dedupliziert zusaetzlich).
  assert.deepEqual([...new Set(symbols)], symbols);
  setOppFilterInputs({});
  sandbox.dashboardState = null;
});

test('oppVisibleSortedItems: empty dashboardState degrades gracefully to empty list', () => {
  sandbox.dashboardState = null;
  const visible = vm.runInContext('oppVisibleSortedItems()', sandbox);
  assert.deepEqual(Array.from(visible), []);
});
