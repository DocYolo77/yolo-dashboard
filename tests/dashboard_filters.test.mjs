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

// Simple single-statement const extractor (e.g. `const TC_W = 300, TC_H = 140;`
// or `const TC_COLORS = {...};`) -- sufficient for the flat literal consts
// the ticker hover-chart functions below depend on.
function extractConst(src, name) {
  const startIdx = src.indexOf(`const ${name} =`);
  if (startIdx === -1) throw new Error(`const ${name} not found in index.html`);
  const endIdx = src.indexOf(';', startIdx);
  return src.slice(startIdx, endIdx + 1);
}

// Same idea as extractConst but for `let NAME = ...;` module-level state
// (e.g. oppTableOpen/oppRenderLimit — the Opportunities collapse/pagination
// state, spec point 4).
function extractLet(src, name) {
  const startIdx = src.indexOf(`let ${name} =`);
  if (startIdx === -1) throw new Error(`let ${name} not found in index.html`);
  const endIdx = src.indexOf(';', startIdx);
  return src.slice(startIdx, endIdx + 1);
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
// V6 point 7/11-13: RSP-based Narrative Strength/Thrust headline metrics.
const ncScoreValueSrc = extractFunction(html, 'ncScoreValue');
const ncFmtScoreSrc = extractFunction(html, 'ncFmtScore');
const oppTabFilterSrc = extractFunction(html, 'oppTabFilter');
const oppApplyFiltersSrc = extractFunction(html, 'oppApplyFilters');
const oppSortItemsSrc = extractFunction(html, 'oppSortItems');
const oppVisibleSortedItemsSrc = extractFunction(html, 'oppVisibleSortedItems');
const ncJumpToOpportunitiesSrc = extractFunction(html, 'ncJumpToOpportunities');
// Spec point 4: Opportunities collapse/windowed-rendering state + actions.
const oppTableOpenLetSrc = extractLet(html, 'oppTableOpen');
const oppRenderLimitLetSrc = extractLet(html, 'oppRenderLimit');
const oppPageSizeConstSrc = extractConst(html, 'OPP_RENDER_PAGE_SIZE');
const oppToggleTableSrc = extractFunction(html, 'oppToggleTable');
const oppShowMoreSrc = extractFunction(html, 'oppShowMore');
const tcPolylineSegmentsSrc = extractFunction(html, 'tcPolylineSegments');
const tickerChartSvgSrc = extractFunction(html, 'tickerChartSvg');
const tickerChartTooltipHtmlSrc = extractFunction(html, 'tickerChartTooltipHtml');
const positionTickerChartTooltipSrc = extractFunction(html, 'positionTickerChartTooltip');
const tcColorsConstSrc = extractConst(html, 'TC_COLORS');
const tcDimsConstSrc = extractConst(html, 'TC_W'); // combined statement also declares TC_H

const sandbox = {
  ncEmaFilter: 'all',
  ncStateFilter: 'all',
  ncHorizon: '1d',
  engineConfig: { dashboard: { ema_proximity_threshold_pct: 4.0, atr_extension_warning_threshold: 5.0 } },
  ncMemberSort: { field: 'd1_pct', dir: 'desc' },
  dashboardState: null,
  tickerCharts: null,
  window: { innerWidth: 1200, innerHeight: 800 },
  oppTab: 'all',
  oppSortState: { field: 'leadership_score', dir: 'desc' },
  // oppRender is a spy here (not the real, DOM-heavy implementation) — this
  // sandbox only cares whether ncJumpToOpportunities calls it, not what it
  // renders (oppApplyFilters/oppTabFilter above are the ones tested against
  // the real logic).
  oppRenderCallCount: 0,
  oppRender() { sandbox.oppRenderCallCount++; },
  // Minimal DOM stub for the Opportunities filter inputs — oppApplyFilters
  // reads these directly via getElementById, same as the real page.
  // scrollIntoView is a no-op spy so ncJumpToOpportunities's section-scroll
  // call doesn't throw in a DOM-less sandbox.
  document: {
    _elements: {},
    getElementById(id) {
      if (!this._elements[id]) this._elements[id] = { value: '', checked: false, scrollIntoView() {} };
      return this._elements[id];
    },
  },
  console,
};
vm.createContext(sandbox);
vm.runInContext(
  `${ncNearEma10Src}\n${ncNearEma20Src}\n${ncIsAtrExtendedSrc}\n${ncEmaFilterMembersSrc}\n${ncSortMembersSrc}\n` +
  `${ncMemberOpportunityStateSrc}\n${ncStateFilterMembersSrc}\n${ncVisibleSortedMembersSrc}\n${ncMomentumTickerListSrc}\n` +
  `${ncScoreValueSrc}\n${ncFmtScoreSrc}\n` +
  `${oppTabFilterSrc}\n${oppApplyFiltersSrc}\n${oppSortItemsSrc}\n${oppVisibleSortedItemsSrc}\n${ncJumpToOpportunitiesSrc}\n` +
  `${tcColorsConstSrc}\n${tcDimsConstSrc}\n${tcPolylineSegmentsSrc}\n${tickerChartSvgSrc}\n${tickerChartTooltipHtmlSrc}\n` +
  `${positionTickerChartTooltipSrc}\n` +
  `${oppTableOpenLetSrc}\n${oppRenderLimitLetSrc}\n${oppPageSizeConstSrc}\n${oppToggleTableSrc}\n${oppShowMoreSrc}`,
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

test('ncMomentumTickerList: copies ONLY items with backend near_emas===true, no frontend recomputation', () => {
  // Fixed rule: the Momentum-Copy button trusts the backend's near_emas
  // flag (Opportunity Engine, calc_near_emas) exclusively -- it must never
  // recompute EMA-distance/ATR-extension thresholds itself. Sourced from
  // dashboardState.opportunities.items (whole eligible universe), not
  // narrativesRaw.narratives[].members.
  const state = {
    opportunities: {
      items: [
        { symbol: 'FLAGGED_TRUE', near_emas: true, ema10_distance_pct: 9.0, atr_extension: 99.0 },  // near_emas wins even though the OLD frontend heuristic would have excluded it
        { symbol: 'FLAGGED_FALSE', near_emas: false, ema10_distance_pct: 0.1, atr_extension: 0.1 }, // near_emas wins even though the OLD frontend heuristic would have included it
        { symbol: 'ALSO_TRUE', near_emas: true },
        { symbol: 'MISSING_FLAG' }, // near_emas undefined -> not === true -> excluded
      ],
    },
  };
  const list = vm.runInContext('ncMomentumTickerList(state)', Object.assign(sandbox, { state }));
  assert.deepEqual(Array.from(list), ['ALSO_TRUE', 'FLAGGED_TRUE']); // sorted, exactly the near_emas=true rows
});

test('ncMomentumTickerList: returns empty list gracefully when opportunities data is unavailable', () => {
  const list1 = vm.runInContext('ncMomentumTickerList(state)', Object.assign(sandbox, { state: null }));
  assert.deepEqual(Array.from(list1), []);
  const list2 = vm.runInContext('ncMomentumTickerList(state)', Object.assign(sandbox, { state: {} }));
  assert.deepEqual(Array.from(list2), []);
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

test('oppApplyFilters: Narrative-Filter "__NONE__" zeigt nur Ticker ohne aktives Narrative', () => {
  // Quality Patch point 26: an explicit "Ohne Narrative" filter option lets
  // users isolate stocks whose only classification is undersized/absent
  // (narratives.length === 0, since build_narratives.py only ever emits
  // ACTIVE narrative memberships into this array).
  setOppFilterInputs({ oppFilterNarrative: '__NONE__' });
  const items = makeOppItems().concat([
    { symbol: 'ZZZ', narratives: [], quality_state: 'neutral', near_emas: false, extended: false,
      constructive_reset_narratives: [], laggard_narratives: [], structural_rs: 50, trend_strength: 40,
      rs_1w: 40, rs_1m: 40, thrust_percentile_1d: 40, thrust_percentile_1w: 40,
      atr_extension: 2.0, w1_pct: 0.5, m1_pct: 1.0, market_cap: 2e9 },
  ]);
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol), ['ZZZ']);
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

// ── Full-Universe: Narrative filter combinable with State tabs (Punkt 13/L) ──

test('Narrative filter (memory) + State tab (leaders) combine to a narrower set than either alone', () => {
  sandbox.dashboardState = { opportunities: { items: makeOppItems() } };
  sandbox.oppTab = 'leaders';
  setOppFilterInputs({ oppFilterNarrative: 'memory' });
  const visible = vm.runInContext('oppVisibleSortedItems()', sandbox);
  // makeOppItems: MU is memory+fresh_leader (leader-like) -> passes both.
  // VRT is ai_infra+leader -> fails the narrative filter.
  // SNDK is memory+neutral -> fails the state tab.
  assert.deepEqual(Array.from(visible).map(it => it.symbol), ['MU']);
  setOppFilterInputs({});
  sandbox.oppTab = 'all';
  sandbox.dashboardState = null;
});

test('Narrative filter = ALL (empty selection) + a state tab behaves like the tab alone', () => {
  sandbox.dashboardState = { opportunities: { items: makeOppItems() } };
  sandbox.oppTab = 'recent';
  setOppFilterInputs({ oppFilterNarrative: '' });
  const visible = vm.runInContext('oppVisibleSortedItems()', sandbox);
  assert.deepEqual(Array.from(visible).map(it => it.symbol), ['WDC']);
  sandbox.oppTab = 'all';
  sandbox.dashboardState = null;
});

// ── Full-Universe: Narrative Card -> Opportunities jump (Punkt 12/M) ──

test('ncJumpToOpportunities sets the narrative filter and triggers a re-render', () => {
  sandbox.oppRenderCallCount = 0;
  sandbox.document._elements.oppFilterNarrative = { value: '', checked: false, scrollIntoView() {} };
  vm.runInContext(`ncJumpToOpportunities('ai_infra')`, sandbox);
  assert.equal(sandbox.document._elements.oppFilterNarrative.value, 'ai_infra');
  assert.equal(sandbox.oppRenderCallCount, 1);
});

test('ncJumpToOpportunities stops the click from also toggling the narrative card (event.stopPropagation)', () => {
  let stopPropagationCalled = false;
  const fakeEvent = { stopPropagation: () => { stopPropagationCalled = true; } };
  vm.runInContext('ncJumpToOpportunities', sandbox)('semiconductors', fakeEvent);
  assert.equal(stopPropagationCalled, true);
});

test('ncJumpToOpportunities does not throw when the event is omitted (called without a click)', () => {
  assert.doesNotThrow(() => {
    vm.runInContext('ncJumpToOpportunities', sandbox)('semiconductors');
  });
});

// ══════════════════════════════════════════════
// TICKER HOVER MINI-CHART (pure functions)
// ══════════════════════════════════════════════

test('tcPolylineSegments: breaks into separate runs at null gaps, drops single-point runs', () => {
  const values = [1, 2, null, 5, 6, 7, null, null, 9];
  const xFor = i => i * 10;
  const yFor = v => 100 - v;
  const segments = vm.runInContext('tcPolylineSegments(values, xFor, yFor)', Object.assign(sandbox, { values, xFor, yFor }));
  const materialized = Array.from(segments).map(seg => Array.from(seg));
  // run [1,2] -> 2 points kept; lone [5,6,7] run of 3 kept; lone trailing [9] (length 1) dropped
  assert.deepEqual(materialized, [
    ['0,99', '10,98'],
    ['30,95', '40,94', '50,93'],
  ]);
});

test('tcPolylineSegments: all-null series produces zero segments (no crash)', () => {
  const values = [null, null, null];
  const xFor = i => i, yFor = v => v;
  const segments = vm.runInContext('tcPolylineSegments(values, xFor, yFor)', Object.assign(sandbox, { values, xFor, yFor }));
  assert.deepEqual(Array.from(segments), []);
});

function makeChart(overrides) {
  const n = 10;
  const base = Array.from({ length: n }, (_, i) => 100 + i);
  return Object.assign({
    dates: Array.from({ length: n }, (_, i) => `2026-01-${i + 1}`),
    o: base.slice(), h: base.map(v => v + 1), l: base.map(v => v - 1), c: base.slice(),
    ema10: base.slice(), ema20: base.slice(), sma50: Array(n).fill(null), sma200: Array(n).fill(null),
  }, overrides || {});
}

test('tickerChartSvg: renders one candle rect per day with usable close data', () => {
  const chart = makeChart();
  const svg = vm.runInContext('tickerChartSvg(chart)', Object.assign(sandbox, { chart }));
  const rectCount = (svg.match(/<rect/g) || []).length;
  assert.equal(rectCount, 10);
  assert.match(svg, /<svg/);
});

test('tickerChartSvg: skips a day with no close value instead of drawing a broken candle', () => {
  const chart = makeChart({ c: [100, 101, null, 103, 104, 105, 106, 107, 108, 109] });
  const svg = vm.runInContext('tickerChartSvg(chart)', Object.assign(sandbox, { chart }));
  const rectCount = (svg.match(/<rect/g) || []).length;
  assert.equal(rectCount, 9); // 10 days minus the one null close
});

test('tickerChartSvg: shows a fallback message instead of an empty/broken chart when there is too little data', () => {
  const chart = makeChart({ dates: ['2026-01-01'], o: [100], h: [101], l: [99], c: [100], ema10: [100], ema20: [100], sma50: [null], sma200: [null] });
  const svg = vm.runInContext('tickerChartSvg(chart)', Object.assign(sandbox, { chart }));
  assert.match(svg, /tc-empty/);
  assert.doesNotMatch(svg, /<svg/);
});

test('tickerChartTooltipHtml: shows "no data" state when the ticker has no chart entry', () => {
  sandbox.tickerCharts = { tickers: {} };
  const html = vm.runInContext(`tickerChartTooltipHtml('GHOST')`, sandbox);
  assert.match(html, /GHOST/);
  assert.match(html, /Keine Chart-Daten/);
});

test('tickerChartTooltipHtml: renders symbol, last close price and full MA legend when data exists', () => {
  sandbox.tickerCharts = { tickers: { AAPL: makeChart() } };
  const html = vm.runInContext(`tickerChartTooltipHtml('AAPL')`, sandbox);
  assert.match(html, /AAPL/);
  assert.match(html, /\$109\.00/); // last close of the synthetic series (100..109)
  ['EMA10', 'EMA20', 'SMA50', 'SMA200'].forEach(label => assert.match(html, new RegExp(label)));
});

test('positionTickerChartTooltip: places the tooltip to the right of the anchor by default', () => {
  const el = { style: {} };
  const anchorRect = { left: 100, right: 130, top: 200, bottom: 216 };
  vm.runInContext('positionTickerChartTooltip(el, anchorRect)', Object.assign(sandbox, { el, anchorRect }));
  assert.equal(el.style.left, '138px'); // anchorRect.right (130) + margin (8)
  assert.equal(el.style.top, '200px');
});

test('positionTickerChartTooltip: flips to the left when the tooltip would overflow the right viewport edge', () => {
  sandbox.window = { innerWidth: 400, innerHeight: 800 };
  const el = { style: {} };
  const anchorRect = { left: 350, right: 380, top: 100, bottom: 116 };
  vm.runInContext('positionTickerChartTooltip(el, anchorRect)', Object.assign(sandbox, { el, anchorRect }));
  assert.equal(el.style.left, '22px'); // anchorRect.left (350) - tooltipW (320) - margin (8)
  sandbox.window = { innerWidth: 1200, innerHeight: 800 }; // restore for later tests
});

test('positionTickerChartTooltip: clamps to the bottom viewport edge instead of overflowing', () => {
  sandbox.window = { innerWidth: 1200, innerHeight: 300 };
  const el = { style: {} };
  const anchorRect = { left: 100, right: 130, top: 280, bottom: 296 };
  vm.runInContext('positionTickerChartTooltip(el, anchorRect)', Object.assign(sandbox, { el, anchorRect }));
  assert.equal(el.style.top, '92px'); // window.innerHeight (300) - tooltipH (200) - margin (8)
  sandbox.window = { innerWidth: 1200, innerHeight: 800 }; // restore
});

// ── Spec point 4: Opportunities table collapse + windowed rendering ──
// oppRender() itself is DOM-heavy (builds the whole table); these tests
// target the two small, real actions the collapse/pagination UX is built
// from (oppToggleTable/oppShowMore), spying on oppRender the same way the
// ncJumpToOpportunities tests above already do, to verify the STATE
// transition and the exact opts each action passes through -- without
// re-simulating the full DOM render.

test('oppToggleTable flips oppTableOpen and triggers a re-render (defaults: collapsed)', () => {
  vm.runInContext('oppTableOpen = false', sandbox);
  sandbox.oppRenderCallCount = 0;
  vm.runInContext('oppToggleTable()', sandbox);
  assert.equal(vm.runInContext('oppTableOpen', sandbox), true);
  assert.equal(sandbox.oppRenderCallCount, 1);

  vm.runInContext('oppToggleTable()', sandbox);
  assert.equal(vm.runInContext('oppTableOpen', sandbox), false); // toggles back
  assert.equal(sandbox.oppRenderCallCount, 2);
});

test('oppShowMore increases oppRenderLimit by exactly the page size', () => {
  vm.runInContext('oppRenderLimit = 50', sandbox);
  vm.runInContext('oppShowMore()', sandbox);
  assert.equal(vm.runInContext('oppRenderLimit', sandbox), 100);
  vm.runInContext('oppShowMore()', sandbox);
  assert.equal(vm.runInContext('oppRenderLimit', sandbox), 150);
});

test('oppShowMore calls oppRender with { keepLimit: true } so the window survives the re-render', () => {
  let receivedOpts = 'NOT_CALLED';
  sandbox.oppRender = (opts) => { receivedOpts = opts; };
  vm.runInContext('oppShowMore()', sandbox);
  // Property-level check rather than assert.deepEqual: the opts object
  // literal is constructed INSIDE the vm sandbox's separate realm, so a
  // strict deep-equal against an outer-realm {keepLimit:true} object can
  // spuriously fail on [[Prototype]] identity even though the data matches.
  assert.equal(receivedOpts.keepLimit, true);
  assert.deepEqual(Object.keys(receivedOpts), ['keepLimit']);
});

test('oppToggleTable calls oppRender with no args (so the render limit resets to the first page)', () => {
  let receivedOpts = 'NOT_CALLED';
  sandbox.oppRender = (opts) => { receivedOpts = opts; };
  vm.runInContext('oppToggleTable()', sandbox);
  assert.equal(receivedOpts, undefined);
});

test('copy-visible-tickers keeps reading from the full filtered DATA STATE regardless of the collapse/window state', () => {
  // Spec point 4's core guarantee: oppVisibleSortedItems() (what
  // oppCopyVisibleTickers reads) must be entirely independent of
  // oppTableOpen/oppRenderLimit -- collapsing the table or only having
  // rendered the first page must never change what gets copied.
  sandbox.dashboardState = {
    opportunities: {
      items: Array.from({ length: 120 }, (_, i) => ({ symbol: `T${i}`, quality_state: 'neutral', narratives: [], constructive_reset_narratives: [], laggard_narratives: [] })),
    },
  };
  sandbox.oppTab = 'all';
  setOppFilterInputs({});

  vm.runInContext('oppTableOpen = false', sandbox);
  vm.runInContext('oppRenderLimit = 50', sandbox);
  const collapsedVisible = vm.runInContext('oppVisibleSortedItems()', sandbox);

  vm.runInContext('oppTableOpen = true', sandbox);
  vm.runInContext('oppRenderLimit = 50', sandbox); // only "page 1" rendered
  const openFirstPageVisible = vm.runInContext('oppVisibleSortedItems()', sandbox);

  assert.equal(collapsedVisible.length, 120); // NOT limited to 0 just because collapsed
  assert.equal(openFirstPageVisible.length, 120); // NOT limited to 50 just because only 50 rows are on-screen
  assert.deepEqual(collapsedVisible.map(it => it.symbol), openFirstPageVisible.map(it => it.symbol));
});

// ── Spec point 7/11-13: RSP-based Narrative Strength/Thrust headline ──

test('ncScoreValue reads strength_rsp[window] for the "strength" score, fully separate per window', () => {
  const narrative = { strength_rsp: { '1w': 80.0, '1m': 70.0, '3m': 65.1, '6m': 61.9 }, thrust_rsp: 78.0 };
  const out = vm.runInContext('ncScoreValue(narrative, "strength", window)', Object.assign(sandbox, { narrative, window: '1w' }));
  assert.equal(out, 80.0);
  const out3m = vm.runInContext('ncScoreValue(narrative, "strength", window)', Object.assign(sandbox, { narrative, window: '3m' }));
  assert.equal(out3m, 65.1);
});

test('ncScoreValue "thrust" ignores the strengthWindow argument entirely (spec point 11: no tab affects Thrust)', () => {
  const narrative = { strength_rsp: { '1w': 80.0, '1m': 70.0, '3m': 65.1, '6m': 61.9 }, thrust_rsp: 78.0 };
  const forWindow1w = vm.runInContext('ncScoreValue(narrative, "thrust", window)', Object.assign(sandbox, { narrative, window: '1w' }));
  const forWindow6m = vm.runInContext('ncScoreValue(narrative, "thrust", window)', Object.assign(sandbox, { narrative, window: '6m' }));
  assert.equal(forWindow1w, 78.0);
  assert.equal(forWindow6m, 78.0);
  assert.equal(forWindow1w, forWindow6m);
});

test('ncScoreValue returns null (never fabricated) when strength_rsp/thrust_rsp are missing', () => {
  const narrative = { name: 'No RSP data yet' };
  assert.equal(vm.runInContext('ncScoreValue(narrative, "strength", window)', Object.assign(sandbox, { narrative, window: '1w' })), null);
  assert.equal(vm.runInContext('ncScoreValue(narrative, "thrust", window)', Object.assign(sandbox, { narrative, window: '1w' })), null);
});

test('ncFmtScore formats Strength as a plain 0-100-style number (percentile rank, not a %-return)', () => {
  const out = vm.runInContext('ncFmtScore(v, "strength")', Object.assign(sandbox, { v: 62.45 }));
  assert.equal(out, '62.5'); // rounds to 1 decimal, no leading + / no % suffix
});

test('ncFmtScore formats Thrust as a signed float, never clamped, matching the worked example from the spec', () => {
  const out = vm.runInContext('ncFmtScore(v, "thrust")', Object.assign(sandbox, { v: 78.0 }));
  assert.equal(out, '+78.00');
  const negative = vm.runInContext('ncFmtScore(v, "thrust")', Object.assign(sandbox, { v: -10.0 }));
  assert.equal(negative, '-10.00');
  const over100 = vm.runInContext('ncFmtScore(v, "thrust")', Object.assign(sandbox, { v: 110.0 }));
  assert.equal(over100, '+110.00'); // NOT clamped to 100
});

test('ncFmtScore renders missing values as "—" for both Strength and Thrust', () => {
  assert.equal(vm.runInContext('ncFmtScore(v, "strength")', Object.assign(sandbox, { v: null })), '—');
  assert.equal(vm.runInContext('ncFmtScore(v, "thrust")', Object.assign(sandbox, { v: undefined })), '—');
});
