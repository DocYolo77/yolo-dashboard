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
const ncEmaFilterMembersSrc = extractFunction(html, 'ncEmaFilterMembers');
// V6.1 (Narrative Ranking & UI Bugfix Patch) point 17: member-table sort is
// now derived from the Ranking-mode toggle, replacing the old free
// per-column ncSortMembers/ncMemberSort. State-Filter (ncStateFilter/
// ncMemberOpportunityState/ncStateFilterMembers) was removed together with
// the Opportunities-summary/jump-link -- the narrative view no longer reads
// Opportunity-engine state at all (point 10).
const ncSortMembersForRankingSrc = extractFunction(html, 'ncSortMembersForRanking');
const ncVisibleSortedMembersSrc = extractFunction(html, 'ncVisibleSortedMembers');
const ncMomentumTickerListSrc = extractFunction(html, 'ncMomentumTickerList');
const ncToggleDetailSrc = extractFunction(html, 'ncToggleDetail');
const ncToggleExpandAllSrc = extractFunction(html, 'ncToggleExpandAll');
// V6.1 point 6-8: cross-sectional Narrative RS/Thrust headline metrics.
const ncScoreValueSrc = extractFunction(html, 'ncScoreValue');
const ncFmtScoreSrc = extractFunction(html, 'ncFmtScore');
const memberColumnsConstSrc = extractConst(html, 'MEMBER_COLUMNS');
// V6 point 29A: QQQ breadth chart NaN-safety / empty-state fallback.
const qqqFiniteSeriesSrc = extractFunction(html, 'qqqFiniteSeries');
const qqqHasEnoughHistorySrc = extractFunction(html, 'qqqHasEnoughHistory');
const qqqEmptyStateHtmlSrc = extractFunction(html, 'qqqEmptyStateHtml');
// V6 point 29B: RSP Benchmark chart (cumulative return, client-side timeframe slicing).
const rsBenchmarkWindowSrc = extractFunction(html, 'rsBenchmarkWindow');
const rsBenchmarkCumulativeReturnSrc = extractFunction(html, 'rsBenchmarkCumulativeReturn');
const rsBenchmarkDateLabelSrc = extractFunction(html, 'rsBenchmarkDateLabel');
// V6 point 19-27: Screener (06) — TradingView export + filename.
const screenerTradingViewTxtContentSrc = extractFunction(html, 'screenerTradingViewTxtContent');
const screenerFilenameSlugSrc = extractFunction(html, 'screenerFilenameSlug');
const oppTabFilterSrc = extractFunction(html, 'oppTabFilter');
const oppApplyFiltersSrc = extractFunction(html, 'oppApplyFilters');
const oppSortItemsSrc = extractFunction(html, 'oppSortItems');
const oppVisibleSortedItemsSrc = extractFunction(html, 'oppVisibleSortedItems');
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
  ncRankingMode: 'rs',
  ncStrengthWindow: '1w',
  ncExpandedId: null,
  ncNarrativesExpanded: false,
  engineConfig: { dashboard: { ema_proximity_threshold_pct: 4.0, atr_extension_warning_threshold: 5.0 } },
  dashboardState: null,
  tickerCharts: null,
  window: { innerWidth: 1200, innerHeight: 800 },
  oppTab: 'all',
  oppSortState: { field: 'leadership_score', dir: 'desc' },
  // renderNarrativeGrid/oppRender are spies here (not the real, DOM-heavy
  // implementations) — ncToggleDetail/ncToggleExpandAll/oppToggleTable/
  // oppShowMore just need to be observed CALLING them, the actual rendering
  // is exercised indirectly through the pure functions it's built from
  // (ncVisibleSortedMembers, oppApplyFilters/oppTabFilter/oppSortItems etc.).
  renderNarrativeGridCallCount: 0,
  renderNarrativeGrid() { sandbox.renderNarrativeGridCallCount++; },
  oppRenderCallCount: 0,
  oppRender() { sandbox.oppRenderCallCount++; },
  // Minimal DOM stub for the Opportunities filter inputs — oppApplyFilters
  // reads these directly via getElementById, same as the real page.
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
  `${ncNearEma10Src}\n${ncNearEma20Src}\n${ncEmaFilterMembersSrc}\n${ncSortMembersForRankingSrc}\n` +
  `${ncVisibleSortedMembersSrc}\n${ncMomentumTickerListSrc}\n${ncToggleDetailSrc}\n${ncToggleExpandAllSrc}\n` +
  `${ncScoreValueSrc}\n${ncFmtScoreSrc}\n${memberColumnsConstSrc}\n` +
  `${qqqFiniteSeriesSrc}\n${qqqHasEnoughHistorySrc}\n${qqqEmptyStateHtmlSrc}\n` +
  `${rsBenchmarkWindowSrc}\n${rsBenchmarkCumulativeReturnSrc}\n${rsBenchmarkDateLabelSrc}\n` +
  `${screenerTradingViewTxtContentSrc}\n${screenerFilenameSlugSrc}\n` +
  `${oppTabFilterSrc}\n${oppApplyFiltersSrc}\n${oppSortItemsSrc}\n${oppVisibleSortedItemsSrc}\n` +
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

// V6.1 point 17: member-table sort is derived from the Ranking-mode toggle
// (never a free per-column click-sort anymore) -- RS mode sorts by Stock RS
// of the currently selected RS-Zeitfenster (percentile_<window>, exposed as
// `rs` by ncVisibleSortedMembers), tie-broken by ticker ascending.
function makeRankedMembers() {
  return [
    { symbol: 'AAA', percentile_1w: 40, stock_thrust_rs: 90, ema10_distance_pct: 1.0, ema20_distance_pct: 8.0 },
    { symbol: 'BBB', percentile_1w: 90, stock_thrust_rs: 10, ema10_distance_pct: 8.0, ema20_distance_pct: -3.5 },
    { symbol: 'CCC', percentile_1w: 90, stock_thrust_rs: 50, ema10_distance_pct: -4.0, ema20_distance_pct: 4.0 },
    { symbol: 'DDD', percentile_1w: null, stock_thrust_rs: null, ema10_distance_pct: null, ema20_distance_pct: null },
  ];
}

test('ncVisibleSortedMembers RS mode: Stock RS (selected window) desc, ticker asc tie-break', () => {
  sandbox.ncEmaFilter = 'all';
  sandbox.ncRankingMode = 'rs';
  sandbox.ncStrengthWindow = '1w';
  const narrative = { members: makeRankedMembers(), n_members: 4 };
  const out = vm.runInContext('ncVisibleSortedMembers(narrative)', Object.assign(sandbox, { narrative }));
  // BBB and CCC tie at RS=90 -> ticker asc (BBB before CCC); AAA=40; DDD=null sorts last.
  assert.deepEqual(out.map(m => m.symbol), ['BBB', 'CCC', 'AAA', 'DDD']);
});

test('ncVisibleSortedMembers THRUST mode: Stock Thrust desc, then Stock RS desc, then ticker asc', () => {
  sandbox.ncEmaFilter = 'all';
  sandbox.ncRankingMode = 'thrust';
  sandbox.ncStrengthWindow = '1w';
  const narrative = { members: makeRankedMembers(), n_members: 4 };
  const out = vm.runInContext('ncVisibleSortedMembers(narrative)', Object.assign(sandbox, { narrative }));
  assert.deepEqual(out.map(m => m.symbol), ['AAA', 'CCC', 'BBB', 'DDD']); // thrust 90, 50, 10, null
});

test('ncVisibleSortedMembers Stock RS follows the RS-Zeitfenster tab (percentile_<window>)', () => {
  sandbox.ncEmaFilter = 'all';
  sandbox.ncRankingMode = 'rs';
  const members = [
    { symbol: 'AAA', percentile_1w: 90, percentile_1m: 10, stock_thrust_rs: 0 },
    { symbol: 'BBB', percentile_1w: 10, percentile_1m: 90, stock_thrust_rs: 0 },
  ];
  sandbox.ncStrengthWindow = '1w';
  const out1w = vm.runInContext('ncVisibleSortedMembers(narrative)', Object.assign(sandbox, { narrative: { members, n_members: 2 } }));
  assert.deepEqual(out1w.map(m => m.symbol), ['AAA', 'BBB']);
  sandbox.ncStrengthWindow = '1m';
  const out1m = vm.runInContext('ncVisibleSortedMembers(narrative)', Object.assign(sandbox, { narrative: { members, n_members: 2 } }));
  assert.deepEqual(out1m.map(m => m.symbol), ['BBB', 'AAA']); // reversed -- proves the tab actually switches the field
});

test('Copy-visible-tickers format: comma-separated symbols only, in the visible+sorted order', () => {
  sandbox.ncEmaFilter = 'either'; // AAA(1.0/8.0), BBB(8.0/-3.5), CCC(-4.0/4.0) all qualify; DDD excluded (nulls)
  sandbox.ncRankingMode = 'rs';
  sandbox.ncStrengthWindow = '1w';
  const narrative = { members: makeRankedMembers(), n_members: 4 };
  const out = vm.runInContext('ncVisibleSortedMembers(narrative)', Object.assign(sandbox, { narrative }));
  const text = out.map(m => m.symbol).join(',');
  assert.equal(text, 'BBB,CCC,AAA'); // RS desc with tie-break, DDD filtered out by the EMA filter
  assert.doesNotMatch(text, /[^A-Z,]/); // symbols + commas only, per point 22's "SNDK,WDC,STX" format
  sandbox.ncEmaFilter = 'all';
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

// ── V6.1 point 13: at most one Member-Detailtabelle open at a time ──

test('ncToggleDetail opens a narrative, re-clicking the same one closes it', () => {
  sandbox.ncExpandedId = null;
  sandbox.renderNarrativeGridCallCount = 0;
  vm.runInContext(`ncToggleDetail('n1')`, sandbox);
  assert.equal(vm.runInContext('ncExpandedId', sandbox), 'n1');
  assert.equal(sandbox.renderNarrativeGridCallCount, 1);

  vm.runInContext(`ncToggleDetail('n1')`, sandbox);
  assert.equal(vm.runInContext('ncExpandedId', sandbox), null); // re-click closes
  assert.equal(sandbox.renderNarrativeGridCallCount, 2);
});

test('ncToggleDetail clicking a different narrative closes the previous and opens the new one', () => {
  sandbox.ncExpandedId = null;
  vm.runInContext(`ncToggleDetail('n1')`, sandbox);
  assert.equal(vm.runInContext('ncExpandedId', sandbox), 'n1');
  vm.runInContext(`ncToggleDetail('n2')`, sandbox);
  assert.equal(vm.runInContext('ncExpandedId', sandbox), 'n2'); // never both open at once
  vm.runInContext('ncExpandedId = null', sandbox);
});

test('ncToggleDetail resets the EMA filter when opening/closing a detail view', () => {
  sandbox.ncEmaFilter = 'ema10';
  vm.runInContext(`ncToggleDetail('n1')`, sandbox);
  assert.equal(vm.runInContext('ncEmaFilter', sandbox), 'all');
  sandbox.ncEmaFilter = 'all';
  vm.runInContext('ncExpandedId = null', sandbox);
});

// ── V6.1 point 12: Top 10 default, expand/collapse all ──

test('ncToggleExpandAll flips ncNarrativesExpanded and triggers a re-render', () => {
  vm.runInContext('ncNarrativesExpanded = false', sandbox);
  sandbox.renderNarrativeGridCallCount = 0;
  vm.runInContext('ncToggleExpandAll()', sandbox);
  assert.equal(vm.runInContext('ncNarrativesExpanded', sandbox), true);
  assert.equal(sandbox.renderNarrativeGridCallCount, 1);

  vm.runInContext('ncToggleExpandAll()', sandbox);
  assert.equal(vm.runInContext('ncNarrativesExpanded', sandbox), false);
  assert.equal(sandbox.renderNarrativeGridCallCount, 2);
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

// ── V6.1 point 6-8: cross-sectional Narrative RS/Thrust headline metrics ──

test('ncScoreValue reads narrative_rs[window] for the "rs" score, fully separate per window', () => {
  const narrative = { narrative_rs: { '1w': 80.0, '1m': 70.0, '3m': 65.1, '6m': 61.9 }, thrust_rsp: 78.0 };
  const out = vm.runInContext('ncScoreValue(narrative, "rs", window)', Object.assign(sandbox, { narrative, window: '1w' }));
  assert.equal(out, 80.0);
  const out3m = vm.runInContext('ncScoreValue(narrative, "rs", window)', Object.assign(sandbox, { narrative, window: '3m' }));
  assert.equal(out3m, 65.1);
});

test('ncScoreValue "thrust" ignores the strengthWindow argument entirely (point 2: no tab affects Thrust)', () => {
  const narrative = { narrative_rs: { '1w': 80.0, '1m': 70.0, '3m': 65.1, '6m': 61.9 }, thrust_rsp: 78.0 };
  const forWindow1w = vm.runInContext('ncScoreValue(narrative, "thrust", window)', Object.assign(sandbox, { narrative, window: '1w' }));
  const forWindow6m = vm.runInContext('ncScoreValue(narrative, "thrust", window)', Object.assign(sandbox, { narrative, window: '6m' }));
  assert.equal(forWindow1w, 78.0);
  assert.equal(forWindow6m, 78.0);
  assert.equal(forWindow1w, forWindow6m);
});

test('ncScoreValue returns null (never fabricated) when narrative_rs/thrust_rsp are missing', () => {
  const narrative = { name: 'No RSP data yet' };
  assert.equal(vm.runInContext('ncScoreValue(narrative, "rs", window)', Object.assign(sandbox, { narrative, window: '1w' })), null);
  assert.equal(vm.runInContext('ncScoreValue(narrative, "thrust", window)', Object.assign(sandbox, { narrative, window: '1w' })), null);
});

test('ncFmtScore formats RS as a plain 0-100-style number (percentile rank, not a %-return)', () => {
  const out = vm.runInContext('ncFmtScore(v, "rs")', Object.assign(sandbox, { v: 62.45 }));
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

test('ncFmtScore renders missing values as "—" for both RS and Thrust', () => {
  assert.equal(vm.runInContext('ncFmtScore(v, "rs")', Object.assign(sandbox, { v: null })), '—');
  assert.equal(vm.runInContext('ncFmtScore(v, "thrust")', Object.assign(sandbox, { v: undefined })), '—');
});

// ── V6.1 point 14/25-26: exact Member-Detailtabelle column set, static HTML checks ──

test('MEMBER_COLUMNS matches the spec\'s exact 6-column set (Ticker/Kurs/Veränderung abs./Veränderung %/RS/Thrust)', () => {
  const cols = vm.runInContext('MEMBER_COLUMNS', sandbox);
  assert.deepEqual(Array.from(cols).map(c => c.key), ['symbol', 'price', 'change_abs', 'd1_pct', 'rs', 'stock_thrust_rs']);
});

test('index.html: "Zeithorizont (Mitglieder)" toggle is fully removed (point 2)', () => {
  assert.doesNotMatch(html, /Zeithorizont \(Mitglieder\)/);
  assert.doesNotMatch(html, /id="horizonToggle"/);
});

test('index.html: RS-Zeitfenster and Ranking controls are present (point 2-3)', () => {
  assert.match(html, /RS-Zeitfenster/);
  assert.match(html, /id="strengthWindowToggle"/);
  assert.match(html, /id="rankingModeToggle"/);
  assert.match(html, /data-ranking="rs"/);
  assert.match(html, /data-ranking="thrust"/);
});

test('index.html: no colored full-surface narrative cards remain (point 1/11)', () => {
  assert.doesNotMatch(html, /class="narrative-card/);
  assert.doesNotMatch(html, /nc-card-scores/);
});

test('index.html: Structural Score/Lifecycle/Momentum-Modifier/Opportunities-summary removed from the narrative view (point 10)', () => {
  assert.doesNotMatch(html, /nc-lifecycle-badge/);
  assert.doesNotMatch(html, /nc-modifier-badge/);
  assert.doesNotMatch(html, /nc-opp-summary/);
  assert.doesNotMatch(html, /ncJumpToOpportunities/);
});

test('index.html: narrative table header is exactly # | Narrative | RS | Thrust (point 11)', () => {
  assert.match(html, /<th>#<\/th><th>Narrative<\/th><th>RS<\/th><th>Thrust<\/th>/);
});

// ── Spec point 29A: QQQ breadth chart NaN-safety / explicit fallback text ──

test('qqqFiniteSeries drops null/undefined/NaN/Infinity, keeps valid numbers', () => {
  const out = vm.runInContext('qqqFiniteSeries(arr)', Object.assign(sandbox, {
    arr: [1.5, null, 2.5, undefined, NaN, Infinity, -Infinity, 3.5, 0],
  }));
  assert.deepEqual(out, [1.5, 2.5, 3.5, 0]);
});

test('qqqFiniteSeries returns empty array for a missing/null input (never throws)', () => {
  // Array.from(...) re-materializes the vm-realm array's contents as a
  // host-realm array -- vm.runInContext's `return []` literal is built in
  // the SEPARATE vm realm, so a strict deepEqual against a host-realm []
  // can spuriously fail on [[Prototype]] identity even with identical
  // (empty) contents.
  assert.deepEqual(Array.from(vm.runInContext('qqqFiniteSeries(arr)', Object.assign(sandbox, { arr: null }))), []);
  assert.deepEqual(Array.from(vm.runInContext('qqqFiniteSeries(arr)', Object.assign(sandbox, { arr: undefined }))), []);
});

test('qqqHasEnoughHistory requires >=2 valid points by default, ignoring NaN/null noise', () => {
  assert.equal(vm.runInContext('qqqHasEnoughHistory(arr)', Object.assign(sandbox, { arr: [1, NaN, null] })), false);
  assert.equal(vm.runInContext('qqqHasEnoughHistory(arr)', Object.assign(sandbox, { arr: [1, 2, NaN] })), true);
  assert.equal(vm.runInContext('qqqHasEnoughHistory(arr)', Object.assign(sandbox, { arr: null })), false);
});

test('qqqEmptyStateHtml renders the explicit fallback message, not a blank string', () => {
  const html = vm.runInContext('qqqEmptyStateHtml(msg)', Object.assign(sandbox, { msg: 'Keine historischen Daten verfügbar.' }));
  assert.match(html, /Keine historischen Daten verfügbar\./);
  assert.match(html, /class="qqq-chart-empty"/);
});

// ── Spec point 29B: RSP Benchmark chart window/cumulative-return logic ──

test('rsBenchmarkWindow takes the last N sessions, real trading days not calendar days', () => {
  const dates = ['2026-01-01', '2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07'];
  const close = [100, 101, 102, 103, 104];
  const out = vm.runInContext('rsBenchmarkWindow(dates, close, 3)', Object.assign(sandbox, { dates, close }));
  assert.deepEqual(Array.from(out.dates), ['2026-01-05', '2026-01-06', '2026-01-07']);
  assert.deepEqual(Array.from(out.close), [102, 103, 104]);
});

test('rsBenchmarkWindow drops non-finite closes before windowing (never passed to the renderer)', () => {
  const dates = ['2026-01-01', '2026-01-02', '2026-01-05'];
  const close = [100, null, 102];
  const out = vm.runInContext('rsBenchmarkWindow(dates, close, 5)', Object.assign(sandbox, { dates, close }));
  assert.deepEqual(Array.from(out.close), [100, 102]);
});

test('rsBenchmarkWindow returns null when fewer than 2 valid points survive', () => {
  const dates = ['2026-01-01'];
  const close = [100];
  const out = vm.runInContext('rsBenchmarkWindow(dates, close, 5)', Object.assign(sandbox, { dates, close }));
  assert.equal(out, null);
});

test('rsBenchmarkCumulativeReturn starts at exactly 0 and matches (close/start-1)*100', () => {
  const out = vm.runInContext('rsBenchmarkCumulativeReturn(closeWindow)', Object.assign(sandbox, { closeWindow: [100, 102, 99, 105] }));
  assert.equal(out[0], 0);
  assert.equal(out[1], pctClose(102, 100));
  assert.equal(out[3], pctClose(105, 100));

  function pctClose(v, start) { return (v / start - 1) * 100; }
});

test('rsBenchmarkDateLabel formats ISO dates as DD.MM. (matches the existing Benchmark date-label convention)', () => {
  const out = vm.runInContext('rsBenchmarkDateLabel(iso)', Object.assign(sandbox, { iso: '2026-03-07' }));
  assert.equal(out, '07.03.');
});

// ── Spec point 26: Screener TradingView export — known MIC -> prefix, ──
// unknown MIC never silently mis-prefixed, exact filename format ──

test('screenerTradingViewTxtContent joins known tradingview_symbol values with commas', () => {
  const tickers = [
    { symbol: 'AAPL', tradingview_symbol: 'NASDAQ:AAPL' },
    { symbol: 'DELL', tradingview_symbol: 'NYSE:DELL' },
  ];
  const out = vm.runInContext('screenerTradingViewTxtContent(tickers)', Object.assign(sandbox, { tickers }));
  assert.equal(out, 'NASDAQ:AAPL,NYSE:DELL');
});

test('screenerTradingViewTxtContent excludes tickers with an unmapped MIC (null tradingview_symbol) instead of mis-prefixing them', () => {
  const tickers = [
    { symbol: 'AAPL', tradingview_symbol: 'NASDAQ:AAPL' },
    { symbol: 'UNKNOWN', tradingview_symbol: null },
    { symbol: 'DELL', tradingview_symbol: 'NYSE:DELL' },
  ];
  const out = vm.runInContext('screenerTradingViewTxtContent(tickers)', Object.assign(sandbox, { tickers }));
  assert.equal(out, 'NASDAQ:AAPL,NYSE:DELL');
  assert.doesNotMatch(out, /UNKNOWN/);
});

test('screenerTradingViewTxtContent only includes tickers from the current screener hits (whatever list it is given)', () => {
  const tickers = [{ symbol: 'ONE', tradingview_symbol: 'NYSE:ONE' }];
  const out = vm.runInContext('screenerTradingViewTxtContent(tickers)', Object.assign(sandbox, { tickers }));
  assert.equal(out, 'NYSE:ONE');
});

test('screenerFilenameSlug converts the preset id to the spec\'s exact dash-separated filename format', () => {
  const out = vm.runInContext('screenerFilenameSlug(presetId)', Object.assign(sandbox, { presetId: 'weekly_strength' }));
  assert.equal(out, 'weekly-strength');
});

test('TradingView TXT filename matches spec point 26\'s exact example shape', () => {
  const slug = vm.runInContext('screenerFilenameSlug(presetId)', Object.assign(sandbox, { presetId: 'weekly_strength' }));
  const filename = `${slug}-2026-03-07-tradingview.txt`;
  assert.equal(filename, 'weekly-strength-2026-03-07-tradingview.txt');
});
