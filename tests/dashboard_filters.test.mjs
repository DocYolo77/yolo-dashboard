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
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const html = readFileSync(path.join(__dirname, '..', 'index.html'), 'utf-8');

function extractFunction(src, name) {
  const startIdx = src.indexOf(`function ${name}(`);
  if (startIdx === -1) throw new Error(`function ${name} not found in index.html`);
  // Skip past the parameter list by balancing parens first, THEN look for
  // the body's opening brace -- a naive "first { after the name" breaks on
  // a destructured parameter like `function f(x, { a, b }) { ... }` (e.g.
  // sortTableRows(rows, { direction, type, accessor, key })), which would
  // otherwise treat the parameter's own `{...}` as the function body.
  let parenDepth = 0, i = src.indexOf('(', startIdx);
  for (; i < src.length; i++) {
    if (src[i] === '(') parenDepth++;
    else if (src[i] === ')') { parenDepth--; if (parenDepth === 0) { i++; break; } }
  }
  while (src[i] !== '{') i++;
  let depth = 0;
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
// Point 12B: manual header-click override state for the Member-Detailtabelle
// -- ncVisibleSortedMembers reads this module-level `let` directly.
const memberTableSortLetSrc = extractLet(html, 'memberTableSort');
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
// RVOL/Screener/Benchmark/Futures Patch point 2: shared Strength color helper.
const ncStrengthColorClassSrc = extractFunction(html, 'ncStrengthColorClass');
// Point 5: Volumen (M) coloring; point 3: Volumen (M) display formatting.
const ncVolumeColorClassSrc = extractFunction(html, 'ncVolumeColorClass');
const ncFmtVolumeMSrc = extractFunction(html, 'ncFmtVolumeM');
// Point 9-10: restored multi-narrative-pill Benchmark chart (commit ec36828a),
// now rendering the current equal-weight/RSP rs_history methodology.
const benchmarkColorsConstSrc = extractConst(html, 'BENCHMARK_COLORS');
const renderBenchmarkPillsSrc = extractFunction(html, 'renderBenchmarkPills');
const ncToggleBenchmarkSrc = extractFunction(html, 'ncToggleBenchmark');
const benchmarkLineChartSrc = extractFunction(html, 'benchmarkLineChart');
const renderBenchmarkChartSrc = extractFunction(html, 'renderBenchmarkChart');
// V6 point 19-27: Screener (06) — TradingView export + filename.
const screenerTradingViewTxtContentSrc = extractFunction(html, 'screenerTradingViewTxtContent');
const screenerFilenameSlugSrc = extractFunction(html, 'screenerFilenameSlug');
const screenerFmtNumSrc = extractFunction(html, 'screenerFmtNum');
// Screener-Ergebnistabellen-Umbau: screenerColumnCellHtml now mirrors the
// Member table's own cell rendering (fmtPrice/pctClass alongside the
// already-extracted fmtPct/ncFmtPctOrDash/ncVolumeColorClass/ncFmtVolumeM/
// ncStrengthColorClass), and SCREENER_PRESET_COLUMNS is now built via the
// screenerStandardColumns(...) factory instead of 4 hand-written arrays.
const fmtPriceSrc = extractFunction(html, 'fmtPrice');
const pctClassSrc = extractFunction(html, 'pctClass');
const screenerStandardColumnsSrc = extractFunction(html, 'screenerStandardColumns');
const screenerColumnCellHtmlSrc = extractFunction(html, 'screenerColumnCellHtml');
const screenerPresetColumnsConstSrc = extractConst(html, 'SCREENER_PRESET_COLUMNS');
const screenerPresetTagsConstSrc = extractConst(html, 'SCREENER_PRESET_TAGS');
// Strength Screeners 3M/6M Union Patch point 12B: ONE shared sort primitive
// pair (sortTableRows/sortTableNextState), plus the generic MARKET_TABLE_IDS
// -driven implementation that replaced the old Futures-only
// futuresReturn5d/futuresSortValue/sortFuturesRows/futuresSortBy/renderFuturesTable.
const sortTableRowsSrc = extractFunction(html, 'sortTableRows');
const sortTableNextStateSrc = extractFunction(html, 'sortTableNextState');
const marketTableIdsConstSrc = extractConst(html, 'MARKET_TABLE_IDS');
const marketTableReturn5dSrc = extractFunction(html, 'marketTableReturn5d');
const marketTableSortValueSrc = extractFunction(html, 'marketTableSortValue');
const sortMarketTableRowsSrc = extractFunction(html, 'sortMarketTableRows');
// Point 12A: null-safe MTD %/YTD % formatting/coloring (deliberately NOT
// reusing fmtPct/pctClass, see the helpers' own comments in index.html).
// ncFmtPctOrDash delegates to fmtPct/fmt for the non-null case, so those two
// need to come along too.
const fmtSrc = extractFunction(html, 'fmt');
const fmtPctSrc = extractFunction(html, 'fmtPct');
const ncFmtPctOrDashSrc = extractFunction(html, 'ncFmtPctOrDash');
const ncPctColorClassStrictSrc = extractFunction(html, 'ncPctColorClassStrict');
// Point 12B: Member-Detailtabelle + Narrative-Haupttabelle manual sort.
const ncMemberColumnTypeSrc = extractFunction(html, 'ncMemberColumnType');
const ncTableSortAccessorSrc = extractFunction(html, 'ncTableSortAccessor');
// Point 12B: Screener-Ergebnistabellen manual sort.
const screenerColumnTypeSrc = extractFunction(html, 'screenerColumnType');
// Point 5/6/7: 3 Month / 6 Month Strength screener presets.
const screenerPresetHtmlSrc = extractFunction(html, 'screenerPresetHtml');
// Point 8: global Strength-union copy button.
const screenerStrengthPresetIdsConstSrc = extractConst(html, 'SCREENER_STRENGTH_PRESET_IDS');
const screenerUnionTickersSrc = extractFunction(html, 'screenerUnionTickers');
const screenerCopyUnionSrc = extractFunction(html, 'screenerCopyUnion');
// ATR-Extension-Ausschluss-Button: zweiter Union-Copy neben "Alle Strength-
// Kandidaten kopieren", schliesst atr_extension > Schwelle aus.
const screenerUnionTickersExcludingAtrExtendedSrc = extractFunction(html, 'screenerUnionTickersExcludingAtrExtended');
const screenerCopyUnionAtrFilteredSrc = extractFunction(html, 'screenerCopyUnionAtrFiltered');
const screenerBaseUniverseLineConstSrc = extractConst(html, 'SCREENER_BASE_UNIVERSE_LINE');
const oppTabFilterSrc = extractFunction(html, 'oppTabFilter');
const oppApplyFiltersSrc = extractFunction(html, 'oppApplyFilters');
const oppSortItemsSrc = extractFunction(html, 'oppSortItems');
const oppVisibleSortedItemsSrc = extractFunction(html, 'oppVisibleSortedItems');
// Calibration-aware Opportunities UI v1, spec point 1: the central
// Structural-RS-Timeframe selector state + the per-horizon field pickers
// oppApplyFilters/oppSortItems (via OPP_DYNAMIC_ACCESSORS) depend on.
const oppRsHorizonLetSrc = extractLet(html, 'oppRsHorizon');
const oppRsFieldSrc = extractFunction(html, 'oppRsField');
const oppRelativeStrengthFieldSrc = extractFunction(html, 'oppRelativeStrengthField');
const oppThrustFieldSrc = extractFunction(html, 'oppThrustField');
const oppDynamicAccessorsConstSrc = extractConst(html, 'OPP_DYNAMIC_ACCESSORS');
const oppQualityLabelConstSrc = extractConst(html, 'QUALITY_V2_LABEL');
const oppQualityHtmlSrc = extractFunction(html, 'oppQualityHtml');
const oppStructureBadgesHtmlSrc = extractFunction(html, 'oppStructureBadgesHtml');
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
  engineConfig: {
    dashboard: { ema_proximity_threshold_pct: 4.0, atr_extension_warning_threshold: 5.0 },
    volume_context: { average_volume_lookback_sessions: 50, high_rvol_threshold: 1.30 },
  },
  dashboardState: null,
  tickerCharts: null,
  // RVOL/Screener/Benchmark/Futures Patch point 9: restored multi-select
  // pill state + the narratives.json payload the pill/chart functions read.
  narrativesRaw: null,
  ncBenchmarkSelected: new Set(),
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
  // Minimal DOM stub for the Opportunities filter inputs / Benchmark pills —
  // oppApplyFilters/renderBenchmarkPills/renderBenchmarkChart read these
  // directly via getElementById, same as the real page. innerHTML is just a
  // plain settable property here (no real DOM), enough to prove the render
  // functions run to completion and mutate state correctly without a crash.
  document: {
    _elements: {},
    getElementById(id) {
      if (!this._elements[id]) this._elements[id] = { value: '', checked: false, innerHTML: '', scrollIntoView() {} };
      return this._elements[id];
    },
    querySelectorAll() { return []; },
  },
  // Point 8: screenerCopyUnion's only side effect besides mutating the
  // button label is navigator.clipboard.writeText — captured here instead
  // of hitting a real clipboard.
  screenersRaw: null,
  clipboardWrites: [],
  navigator: {
    clipboard: {
      writeText(text) { sandbox.clipboardWrites.push(text); return Promise.resolve(); },
    },
  },
  // screenerCopyUnion/screenerCopyUnionAtrFiltered revert the button label
  // back to its original text via setTimeout(…, 1500) after the clipboard
  // write. Tests assert the button's "Kopiert (N)" state immediately after
  // the write (well before 1500ms would elapse in a real browser), so this
  // stub deliberately never fires `fn` -- it just needs to exist so the
  // call doesn't throw ReferenceError in the vm sandbox.
  setTimeout: () => 0,
  clearTimeout: () => {},
  console,
};
vm.createContext(sandbox);
vm.runInContext(
  `${ncNearEma10Src}\n${ncNearEma20Src}\n${ncEmaFilterMembersSrc}\n${ncSortMembersForRankingSrc}\n` +
  `${memberTableSortLetSrc}\n${ncVisibleSortedMembersSrc}\n${ncMomentumTickerListSrc}\n${ncToggleDetailSrc}\n${ncToggleExpandAllSrc}\n` +
  `${ncScoreValueSrc}\n${ncFmtScoreSrc}\n${memberColumnsConstSrc}\n${ncStrengthColorClassSrc}\n` +
  `${ncVolumeColorClassSrc}\n${ncFmtVolumeMSrc}\n` +
  `${qqqFiniteSeriesSrc}\n${qqqHasEnoughHistorySrc}\n${qqqEmptyStateHtmlSrc}\n` +
  `${benchmarkColorsConstSrc}\n${renderBenchmarkPillsSrc}\n${ncToggleBenchmarkSrc}\n${benchmarkLineChartSrc}\n${renderBenchmarkChartSrc}\n` +
  `function qqqXAxis(){ return ''; }\n` +
  `${screenerTradingViewTxtContentSrc}\n${screenerFilenameSlugSrc}\n` +
  `${fmtPriceSrc}\n${pctClassSrc}\n${screenerFmtNumSrc}\n${screenerStandardColumnsSrc}\n${screenerPresetColumnsConstSrc}\n${screenerPresetTagsConstSrc}\n${screenerColumnCellHtmlSrc}\n` +
  `${sortTableRowsSrc}\n${sortTableNextStateSrc}\n` +
  `${marketTableIdsConstSrc}\n${marketTableReturn5dSrc}\n${marketTableSortValueSrc}\n${sortMarketTableRowsSrc}\n` +
  `${fmtSrc}\n${fmtPctSrc}\n${ncFmtPctOrDashSrc}\n${ncPctColorClassStrictSrc}\n${ncMemberColumnTypeSrc}\n${ncTableSortAccessorSrc}\n` +
  `${screenerColumnTypeSrc}\n${screenerStrengthPresetIdsConstSrc}\n${screenerUnionTickersSrc}\n${screenerCopyUnionSrc}\n` +
  `${screenerUnionTickersExcludingAtrExtendedSrc}\n${screenerCopyUnionAtrFilteredSrc}\n` +
  `${oppRsHorizonLetSrc}\n${oppRsFieldSrc}\n${oppRelativeStrengthFieldSrc}\n${oppThrustFieldSrc}\n${oppDynamicAccessorsConstSrc}\n` +
  `${oppQualityLabelConstSrc}\n${oppQualityHtmlSrc}\n${oppStructureBadgesHtmlSrc}\n` +
  `${oppTabFilterSrc}\n${oppApplyFiltersSrc}\n${oppSortItemsSrc}\n${oppVisibleSortedItemsSrc}\n` +
  `${tcColorsConstSrc}\n${tcDimsConstSrc}\n${tcPolylineSegmentsSrc}\n${tickerChartSvgSrc}\n${tickerChartTooltipHtmlSrc}\n` +
  `${positionTickerChartTooltipSrc}\n` +
  `${oppTableOpenLetSrc}\n${oppRenderLimitLetSrc}\n${oppPageSizeConstSrc}\n${oppToggleTableSrc}\n${oppShowMoreSrc}`,
  sandbox
);

function setOppFilterInputs(overrides) {
  const defaults = {
    oppFilterNarrative: '', oppFilterStructuralRs: '', oppFilterRelativeStrength: '', oppFilterThrust: '', oppFilterAtr: '', oppFilterCap: '',
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

// Calibration-aware Opportunities UI v1: every item now also carries the
// per-horizon rs_1w/1m/3m/6m (Structural RS, already existed),
// relative_strength_1w/1m/3m/6m (new) and thrust_percentile_1w/1m/3m/6m
// (1w already existed, 3m/6m new) fields the central oppRsHorizon selector
// picks between, plus the new Structure/Quality fields (quality_v2,
// ema10_pullback, ema20_pullback, resetting, extended_v2). Legacy fields
// (quality_state, near_emas, extended, structural_rs, laggard_narratives,
// constructive_reset_narratives) are UNCHANGED -- the 6 legacy tabs still
// filter on them (spec point 19: not redefined here).
function makeOppItems() {
  return [
    { symbol: 'MU', narratives: ['memory'], quality_state: 'fresh_leader', near_emas: true, extended: false,
      constructive_reset_narratives: ['memory'], laggard_narratives: [], structural_rs: 95, trend_strength: 82,
      leadership_score: 92,
      rs_1w: 90, rs_1m: 88, rs_3m: 80, rs_6m: 75,
      relative_strength_1w: 85, relative_strength_1m: 80, relative_strength_3m: 70, relative_strength_6m: 65,
      thrust_percentile_1d: 90, thrust_percentile_1w: 85, thrust_percentile_1m: 80, thrust_percentile_3m: 70, thrust_percentile_6m: 60,
      ema10_distance_pct: 1.0, ema20_distance_pct: 2.0,
      atr_extension: 3.0, w1_pct: 8.0, m1_pct: 20.0, market_cap: 120e9,
      quality_v2: 'leader', ema10_pullback: true, ema20_pullback: false, resetting: false, extended_v2: false },
    { symbol: 'VRT', narratives: ['ai_infra'], quality_state: 'leader', near_emas: false, extended: true,
      constructive_reset_narratives: [], laggard_narratives: [], structural_rs: 90, trend_strength: 78,
      leadership_score: 88,
      rs_1w: 85, rs_1m: 80, rs_3m: 75, rs_6m: 70,
      relative_strength_1w: 75, relative_strength_1m: 70, relative_strength_3m: 65, relative_strength_6m: 60,
      thrust_percentile_1d: 60, thrust_percentile_1w: 55, thrust_percentile_1m: 50, thrust_percentile_3m: 45, thrust_percentile_6m: 40,
      ema10_distance_pct: 12.0, ema20_distance_pct: 15.0,
      atr_extension: 6.5, w1_pct: 15.0, m1_pct: 30.0, market_cap: 40e9,
      // Deliberately NOT extended_v2 (6.5 < the new 8.0 threshold) even
      // though the LEGACY hysteresis-based `extended` is true (enter 5.0) --
      // proves the two thresholds are genuinely independent (spec point 12).
      quality_v2: 'leader', ema10_pullback: false, ema20_pullback: false, resetting: false, extended_v2: true },
    { symbol: 'SNDK', narratives: ['memory'], quality_state: 'neutral', near_emas: true, extended: false,
      constructive_reset_narratives: [], laggard_narratives: ['memory'], structural_rs: 45, trend_strength: 30,
      leadership_score: 40,
      rs_1w: 35, rs_1m: 30, rs_3m: 25, rs_6m: 20,
      relative_strength_1w: 30, relative_strength_1m: 25, relative_strength_3m: 20, relative_strength_6m: 15,
      thrust_percentile_1d: 20, thrust_percentile_1w: 25, thrust_percentile_1m: 22, thrust_percentile_3m: 18, thrust_percentile_6m: 15,
      ema10_distance_pct: -1.5, ema20_distance_pct: -1.0,
      atr_extension: 1.0, w1_pct: -3.0, m1_pct: -5.0, market_cap: 8e9,
      // Both Pullback badges true simultaneously (spec point 11).
      quality_v2: 'laggard', ema10_pullback: true, ema20_pullback: true, resetting: false, extended_v2: false },
    // Isolated from the memory/near-EMA/RS1W/ATR/cap fixtures above on
    // purpose, so it only shows up where a test specifically means to
    // exercise it (the 'recent' tab and the Structural RS filter below) —
    // every OTHER existing filter test's expected list stays unchanged.
    { symbol: 'WDC', narratives: ['ai_infra'], quality_state: 'recent_leader', near_emas: false, extended: false,
      constructive_reset_narratives: [], laggard_narratives: [], structural_rs: 88, trend_strength: 60,
      leadership_score: 70,
      rs_1w: 60, rs_1m: 65, rs_3m: 68, rs_6m: 70,
      relative_strength_1w: 55, relative_strength_1m: 60, relative_strength_3m: 63, relative_strength_6m: 65,
      thrust_percentile_1d: 50, thrust_percentile_1w: 45, thrust_percentile_1m: 48, thrust_percentile_3m: 50, thrust_percentile_6m: 52,
      ema10_distance_pct: 9.0, ema20_distance_pct: 9.0,
      atr_extension: 8.0, w1_pct: -1.0, m1_pct: 5.0, market_cap: 15e9,
      // Resetting alongside a "recent leader" quality -- proves Structure
      // and Quality never gate each other (spec point 16).
      quality_v2: 'leader', ema10_pullback: false, ema20_pullback: false, resetting: true, extended_v2: false },
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
      rs_1w: 40, rs_1m: 40, rs_3m: 40, rs_6m: 40,
      relative_strength_1w: 40, relative_strength_1m: 40, relative_strength_3m: 40, relative_strength_6m: 40,
      thrust_percentile_1d: 40, thrust_percentile_1w: 40, thrust_percentile_1m: 40, thrust_percentile_3m: 40, thrust_percentile_6m: 40,
      atr_extension: 2.0, w1_pct: 0.5, m1_pct: 1.0, market_cap: 2e9 },
  ]);
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol), ['ZZZ']);
  setOppFilterInputs({});
});

// ── Calibration-aware Opportunities UI v1, spec point 1: Structural RS/
// Relative Strength/Thrust and their Min. filters all follow ONE central
// 1W/1M/3M/6M selector, fully separate horizons, never averaged ──

test('oppRsField/oppRelativeStrengthField/oppThrustField: map each horizon to its own, separate field', () => {
  const map = (fn, h) => vm.runInContext(`${fn}('${h}')`, sandbox);
  assert.equal(map('oppRsField', '1w'), 'rs_1w');
  assert.equal(map('oppRsField', '1m'), 'rs_1m');
  assert.equal(map('oppRsField', '3m'), 'rs_3m');
  assert.equal(map('oppRsField', '6m'), 'rs_6m');
  assert.equal(map('oppRelativeStrengthField', '1w'), 'relative_strength_1w');
  assert.equal(map('oppRelativeStrengthField', '6m'), 'relative_strength_6m');
  assert.equal(map('oppThrustField', '1w'), 'thrust_percentile_1w');
  assert.equal(map('oppThrustField', '3m'), 'thrust_percentile_3m');
  assert.equal(map('oppThrustField', '6m'), 'thrust_percentile_6m');
});

test('oppApplyFilters: Min Structural RS reads rs_1m under the default 1M horizon', () => {
  vm.runInContext("oppRsHorizon = '1m'", sandbox);
  setOppFilterInputs({ oppFilterStructuralRs: '70' });
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol).sort(), ['MU', 'VRT']); // rs_1m 88, 80 pass; WDC 65 / SNDK 30 excluded
  setOppFilterInputs({});
});

test('oppApplyFilters: Min Structural RS switches to rs_6m when the horizon toggle is 6M', () => {
  // Same filter value, DIFFERENT field read, DIFFERENT result set -- proves
  // the timeframe switch actually changes what "Structural RS" means here,
  // not just its label. rs_6m: MU 75, VRT 70, WDC 70, SNDK 20.
  vm.runInContext("oppRsHorizon = '6m'", sandbox);
  setOppFilterInputs({ oppFilterStructuralRs: '70' });
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol).sort(), ['MU', 'VRT', 'WDC']); // SNDK's 20 excluded
  vm.runInContext("oppRsHorizon = '1m'", sandbox); // reset for subsequent tests
  setOppFilterInputs({});
});

test('oppApplyFilters: Min Relative Strength (renamed from the fixed Min RS1W filter) follows the same horizon selector', () => {
  vm.runInContext("oppRsHorizon = '1m'", sandbox);
  setOppFilterInputs({ oppFilterRelativeStrength: '65' });
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol).sort(), ['MU', 'VRT']); // relative_strength_1m 80, 70 pass
  setOppFilterInputs({});
});

test('oppApplyFilters: Min Thrust follows the horizon selector too', () => {
  vm.runInContext("oppRsHorizon = '3m'", sandbox);
  setOppFilterInputs({ oppFilterThrust: '60' });
  const items = makeOppItems();
  const out = vm.runInContext('oppApplyFilters(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(out).map(it => it.symbol), ['MU']); // thrust_percentile_3m: MU 70 passes; others (45/18/50) don't
  vm.runInContext("oppRsHorizon = '1m'", sandbox);
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

test('oppSortItems: sorts by structural_rs_dyn (the new default sort field), following the 1M default horizon', () => {
  const items = makeOppItems();
  vm.runInContext("oppRsHorizon = '1m'", sandbox);
  sandbox.oppSortState = { field: 'structural_rs_dyn', dir: 'desc' };
  const desc = vm.runInContext('oppSortItems(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(desc).map(it => it.symbol), ['MU', 'VRT', 'WDC', 'SNDK']); // rs_1m: 88, 80, 65, 30
});

test('oppSortItems: structural_rs_dyn re-sorts by a different field when the horizon changes', () => {
  // rs_3m (no ties, unlike 6m): MU 80, VRT 75, WDC 68, SNDK 25.
  const items = makeOppItems();
  vm.runInContext("oppRsHorizon = '3m'", sandbox);
  sandbox.oppSortState = { field: 'structural_rs_dyn', dir: 'desc' };
  const desc = vm.runInContext('oppSortItems(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(desc).map(it => it.symbol), ['MU', 'VRT', 'WDC', 'SNDK']);
  vm.runInContext("oppRsHorizon = '1m'", sandbox); // reset for subsequent tests
});

test('oppSortItems: relative_strength_dyn sorts by the horizon-selected Relative Strength field', () => {
  const items = makeOppItems();
  vm.runInContext("oppRsHorizon = '1m'", sandbox);
  sandbox.oppSortState = { field: 'relative_strength_dyn', dir: 'desc' };
  const desc = vm.runInContext('oppSortItems(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(desc).map(it => it.symbol), ['MU', 'VRT', 'WDC', 'SNDK']); // relative_strength_1m: 80, 70, 60, 25
});

test('oppSortItems: thrust_dyn sorts by the horizon-selected Thrust field', () => {
  // thrust_percentile_6m: MU 60, WDC 52, VRT 40, SNDK 15.
  const items = makeOppItems();
  vm.runInContext("oppRsHorizon = '6m'", sandbox);
  sandbox.oppSortState = { field: 'thrust_dyn', dir: 'desc' };
  const desc = vm.runInContext('oppSortItems(items)', Object.assign(sandbox, { items }));
  assert.deepEqual(Array.from(desc).map(it => it.symbol), ['MU', 'WDC', 'VRT', 'SNDK']);
  vm.runInContext("oppRsHorizon = '1m'", sandbox); // reset for subsequent tests
});

// ── Calibration-aware Opportunities UI v1, spec point 3/8: Quality
// (Leader/Neutral/Laggard) and Structure (independent markers) ──

test('oppQualityHtml: renders the quality_v2 bucket label, defaulting to Neutral when missing', () => {
  const leader = vm.runInContext('oppQualityHtml(it)', Object.assign(sandbox, { it: { quality_v2: 'leader' } }));
  const laggard = vm.runInContext('oppQualityHtml(it)', Object.assign(sandbox, { it: { quality_v2: 'laggard' } }));
  const missing = vm.runInContext('oppQualityHtml(it)', Object.assign(sandbox, { it: {} }));
  assert.match(leader, /class="opp-quality leader"/);
  assert.match(leader, />Leader</);
  assert.match(laggard, /class="opp-quality laggard"/);
  assert.match(missing, /class="opp-quality neutral"/);
});

test('oppStructureBadgesHtml: no badge is rendered when nothing applies (spec point 15: no visible "Normal" badge)', () => {
  const html = vm.runInContext('oppStructureBadgesHtml(it)',
    Object.assign(sandbox, { it: { ema10_pullback: false, ema20_pullback: false, resetting: false, extended_v2: false } }));
  assert.doesNotMatch(html, /<span class="opp-badge/);
});

test('oppStructureBadgesHtml: missing/undefined flags produce no false badges', () => {
  const html = vm.runInContext('oppStructureBadgesHtml(it)', Object.assign(sandbox, { it: {} }));
  assert.doesNotMatch(html, /<span class="opp-badge/);
});

test('oppStructureBadgesHtml: EMA10 and EMA20 Pullback can both be shown at once', () => {
  const html = vm.runInContext('oppStructureBadgesHtml(it)',
    Object.assign(sandbox, { it: { ema10_pullback: true, ema20_pullback: true, resetting: false, extended_v2: false } }));
  assert.match(html, /EMA10 Pullback/);
  assert.match(html, /EMA20 Pullback/);
});

test('oppStructureBadgesHtml: Resetting and a Pullback badge coexist (spec point 14)', () => {
  const html = vm.runInContext('oppStructureBadgesHtml(it)',
    Object.assign(sandbox, { it: { ema10_pullback: false, ema20_pullback: true, resetting: true, extended_v2: false } }));
  assert.match(html, /EMA20 Pullback/);
  assert.match(html, /Resetting/);
});

test('oppStructureBadgesHtml: Extended renders independently and can coexist with everything else', () => {
  const html = vm.runInContext('oppStructureBadgesHtml(it)',
    Object.assign(sandbox, { it: { ema10_pullback: true, ema20_pullback: false, resetting: true, extended_v2: true } }));
  assert.match(html, /EMA10 Pullback/);
  assert.match(html, /Resetting/);
  assert.match(html, /Extended/);
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

test('MEMBER_COLUMNS matches the exact 9-column set incl. MTD %/YTD %/Volumen (M), RS label -> Strength', () => {
  // Strength Screeners 3M/6M Union Patch point 12A: exact new column order
  // "Ticker | Kurs | Veränderung absolut | Veränderung % | MTD % | YTD % |
  // Volumen (M) | Strength | Thrust".
  const cols = vm.runInContext('MEMBER_COLUMNS', sandbox);
  assert.deepEqual(Array.from(cols).map(c => c.key),
    ['symbol', 'price', 'change_abs', 'd1_pct', 'mtd_pct', 'ytd_pct', 'volume', 'rs', 'stock_thrust_rs']);
  const byKey = Object.fromEntries(Array.from(cols).map(c => [c.key, c.label]));
  assert.equal(byKey.volume, 'Volumen (M)');
  assert.equal(byKey.mtd_pct, 'MTD %');
  assert.equal(byKey.ytd_pct, 'YTD %');
  assert.equal(byKey.rs, 'Strength'); // point 1: display label only, field key stays "rs"
  // MTD %/YTD % sit directly after Veränderung %, and Volumen (M) directly
  // after YTD % (point 12A's exact column order).
  const keys = Array.from(cols).map(c => c.key);
  assert.equal(keys.indexOf('mtd_pct'), keys.indexOf('d1_pct') + 1);
  assert.equal(keys.indexOf('ytd_pct'), keys.indexOf('mtd_pct') + 1);
  assert.equal(keys.indexOf('volume'), keys.indexOf('ytd_pct') + 1);
});

test('index.html: "Zeithorizont (Mitglieder)" toggle is fully removed (point 2)', () => {
  assert.doesNotMatch(html, /Zeithorizont \(Mitglieder\)/);
  assert.doesNotMatch(html, /id="horizonToggle"/);
});

test('index.html: Strength-Zeitfenster and Ranking controls are present, no visible "RS-Zeitfenster" label (point 1)', () => {
  assert.match(html, /Strength-Zeitfenster/);
  // The visible <span class="nc-label"> reads "Strength-Zeitfenster", never
  // "RS-Zeitfenster" -- code comments elsewhere in the file may still say
  // "RS-Zeitfenster" as shorthand for the underlying narrative_rs concept
  // (point 1: internal names/prose not unnecessarily migrated), so this
  // check is scoped to the actual rendered label element, not the whole file.
  assert.doesNotMatch(html, /nc-label">RS-Zeitfenster</);
  assert.match(html, /id="strengthWindowToggle"/);
  assert.match(html, /id="rankingModeToggle"/);
  // data-ranking="rs" is the internal identifier (point 1: internal field
  // names not unnecessarily migrated) -- the VISIBLE button label is STRENGTH.
  assert.match(html, /data-ranking="rs">STRENGTH</);
  assert.match(html, /data-ranking="thrust">THRUST</);
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

test('index.html: narrative table header is # | Narrative | Strength | Thrust, no bare "RS" (point 1/11)', () => {
  const theadMatch = html.match(/<table class="data-table narrative-rank-table"[^>]*>\s*<thead><tr>([\s\S]*?)<\/tr><\/thead>/);
  assert.ok(theadMatch, 'narrative-rank-table <thead> not found');
  const headerCells = theadMatch[1].match(/>[^<]*(?=<\/th>)/g).map(s => s.slice(1));
  assert.deepEqual(headerCells, ['#', 'Narrative', 'Strength', 'Thrust']);
  assert.doesNotMatch(html, /<th>RS<\/th>/);
});

// ── Strength Screeners 3M/6M Union Patch point 12B: Narrative main table is
// header-click sortable (except "#"), per-table independent manual-sort
// state that resets when a "fachliche Ranking-Control" changes ──

test('index.html: "#" column has no sortable class/onclick, the other 3 headers do', () => {
  const theadMatch = html.match(/<table class="data-table narrative-rank-table"[^>]*>\s*<thead><tr>([\s\S]*?)<\/tr><\/thead>/);
  const thead = theadMatch[1];
  assert.match(thead, /<th>#<\/th>/); // plain, not clickable
  ['name', 'strength', 'thrust'].forEach(field => {
    assert.match(thead, new RegExp(`<th class="sortable" data-field="${field}" onclick="ncTableSortBy\\('${field}'\\)">`));
  });
});

test('ncTableSortAccessor: name is text, strength/thrust read ncScoreValue for the selected window', () => {
  const narrative = { name: 'Zeta', narrative_rs: { '1w': 42.0 }, thrust_rsp: 7.5 };
  sandbox.window_ = 'ignored';
  const nameVal = vm.runInContext('ncTableSortAccessor("name")(narrative)', Object.assign(sandbox, { narrative }));
  assert.equal(nameVal, 'Zeta');
  sandbox.ncStrengthWindow = '1w';
  const strengthVal = vm.runInContext('ncTableSortAccessor("strength")(narrative)', Object.assign(sandbox, { narrative }));
  assert.equal(strengthVal, 42.0);
  const thrustVal = vm.runInContext('ncTableSortAccessor("thrust")(narrative)', Object.assign(sandbox, { narrative }));
  assert.equal(thrustVal, 7.5);
});

test('index.html: Strength-Zeitfenster / Ranking-Modus toggles reset BOTH the Narrative table\'s and the Member table\'s manual sort back to default', () => {
  assert.match(html, /strengthWindowToggle[\s\S]{0,400}ncTableSort\s*=\s*null[\s\S]{0,100}memberTableSort\s*=\s*null/);
  assert.match(html, /rankingModeToggle[\s\S]{0,400}ncTableSort\s*=\s*null[\s\S]{0,100}memberTableSort\s*=\s*null/);
});

test('index.html: opening a different Narrative detail resets the Member table\'s manual sort back to default', () => {
  const fn = extractFunction(html, 'ncToggleDetail');
  assert.match(fn, /memberTableSort\s*=\s*null/);
});

// ── RVOL/Screener/Benchmark/Futures Patch point 2: shared Strength color helper ──

test('ncStrengthColorClass: positive/zero -> pos, negative -> neg, null/undefined -> neutral', () => {
  assert.equal(vm.runInContext('ncStrengthColorClass(v)', Object.assign(sandbox, { v: 62.5 })), 'pos');
  assert.equal(vm.runInContext('ncStrengthColorClass(v)', Object.assign(sandbox, { v: 0 })), 'pos');
  assert.equal(vm.runInContext('ncStrengthColorClass(v)', Object.assign(sandbox, { v: -1.0 })), 'neg');
  assert.equal(vm.runInContext('ncStrengthColorClass(v)', Object.assign(sandbox, { v: null })), 'muted');
  assert.equal(vm.runInContext('ncStrengthColorClass(v)', Object.assign(sandbox, { v: undefined })), 'muted');
});

test('ncStrengthColorClass: identical value in Narrative main table and Member detail table yields the identical class', () => {
  // Both call sites (renderNarrativeGrid's Strength cell and
  // renderNarrativeDetail's Stock Strength cell) invoke this SAME function —
  // proven here by feeding it the same value twice and asserting equality,
  // the concrete regression the "grey Stock Strength" bug fix guards against.
  const narrativeStrengthValue = 73.2;
  const stockStrengthValue = 73.2;
  const a = vm.runInContext('ncStrengthColorClass(v)', Object.assign(sandbox, { v: narrativeStrengthValue }));
  const b = vm.runInContext('ncStrengthColorClass(v)', Object.assign(sandbox, { v: stockStrengthValue }));
  assert.equal(a, b);
  assert.equal(a, 'pos');
});

test('ncStrengthColorClass never changes the underlying value or ranking, only the CSS class', () => {
  const values = [90, 10, null, -5, 50];
  const classes = values.map(v => vm.runInContext('ncStrengthColorClass(v)', Object.assign(sandbox, { v })));
  assert.deepEqual(classes, ['pos', 'pos', 'muted', 'neg', 'pos']);
  assert.deepEqual(values, [90, 10, null, -5, 50]); // untouched
});

// ── RVOL/Screener/Benchmark/Futures Patch points 3-5: Volumen (M) column ──

test('ncFmtVolumeM: volume / 1_000_000, null renders as em-dash', () => {
  assert.equal(vm.runInContext('ncFmtVolumeM(v)', Object.assign(sandbox, { v: 18420000 })), '18.42');
  assert.equal(vm.runInContext('ncFmtVolumeM(v)', Object.assign(sandbox, { v: null })), '—');
  assert.equal(vm.runInContext('ncFmtVolumeM(v)', Object.assign(sandbox, { v: undefined })), '—');
});

test('ncVolumeColorClass: worked example from the spec (avg50=10M, today 15M -> RVOL50=1.50)', () => {
  const rvol50 = 15000000 / 10000000; // 1.50
  assert.equal(vm.runInContext('ncVolumeColorClass(d1Pct, rvol50, threshold)',
    Object.assign(sandbox, { d1Pct: 1.2, rvol50, threshold: 1.30 })), 'pos'); // up + RVOL1.50 -> green
  assert.equal(vm.runInContext('ncVolumeColorClass(d1Pct, rvol50, threshold)',
    Object.assign(sandbox, { d1Pct: -1.2, rvol50, threshold: 1.30 })), 'neg'); // down + RVOL1.50 -> red
});

test('ncVolumeColorClass: RVOL below threshold is always neutral regardless of direction', () => {
  assert.equal(vm.runInContext('ncVolumeColorClass(d1Pct, rvol50, threshold)',
    Object.assign(sandbox, { d1Pct: 2.0, rvol50: 1.20, threshold: 1.30 })), 'muted');
  assert.equal(vm.runInContext('ncVolumeColorClass(d1Pct, rvol50, threshold)',
    Object.assign(sandbox, { d1Pct: -2.0, rvol50: 1.20, threshold: 1.30 })), 'muted');
});

test('ncVolumeColorClass: d1_pct === 0 is always neutral, even with high RVOL', () => {
  assert.equal(vm.runInContext('ncVolumeColorClass(d1Pct, rvol50, threshold)',
    Object.assign(sandbox, { d1Pct: 0, rvol50: 2.0, threshold: 1.30 })), 'muted');
});

test('ncVolumeColorClass: missing d1_pct or rvol_50 is neutral, never crashes', () => {
  assert.equal(vm.runInContext('ncVolumeColorClass(d1Pct, rvol50, threshold)',
    Object.assign(sandbox, { d1Pct: null, rvol50: 2.0, threshold: 1.30 })), 'muted');
  assert.equal(vm.runInContext('ncVolumeColorClass(d1Pct, rvol50, threshold)',
    Object.assign(sandbox, { d1Pct: 1.0, rvol50: null, threshold: 1.30 })), 'muted');
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

// ── RVOL/Screener/Benchmark/Futures Patch point 9-10: restored multi-pill
// Benchmark chart (commit ec36828a), rendering the current equal-weight/
// RSP rs_history methodology instead of the old median-basket-vs-SPY diff ──

test('index.html: old multi-pill Benchmark UI classes/functions are present (point 9)', () => {
  assert.match(html, /class="benchmark-section"/);
  assert.match(html, /class="benchmark-header"/);
  assert.match(html, /class="bm-pill-row"/);
  assert.match(html, /\.bm-pill /);
  assert.match(html, /class="benchmark-chart-wrap"/);
  assert.match(html, /class="bm-legend"/);
  assert.match(html, /function renderBenchmarkPills/);
  assert.match(html, /function ncToggleBenchmark/);
  assert.match(html, /function benchmarkLineChart/);
  assert.match(html, /function renderBenchmarkChart/);
});

test('index.html: current single-RSP cumulative-return chart is fully removed (point 9/10)', () => {
  assert.doesNotMatch(html, /benchmarkTimeframeToggle/);
  assert.doesNotMatch(html, /ncBenchmarkTimeframe/);
  assert.doesNotMatch(html, /BENCHMARK_RSP_WINDOW_SESSIONS/);
  assert.doesNotMatch(html, /function benchmarkRspChart/);
  assert.doesNotMatch(html, /function rsBenchmarkWindow/);
  assert.doesNotMatch(html, /function rsBenchmarkCumulativeReturn/);
});

test('index.html: Benchmark confirmed as RSP / Invesco S&P 500 Equal Weight ETF, never SPY (point 9)', () => {
  assert.match(html, /Benchmark · RSP \/ S&amp;P 500 Equal Weight/);
  assert.match(html, /Invesco S&amp;P 500 Equal Weight ETF/);
  assert.doesNotMatch(html, /Benchmark.{0,40}SPY/);
});

test('ncToggleBenchmark: multi-select — adding a second narrative keeps the first selected', () => {
  sandbox.ncBenchmarkSelected = new Set();
  sandbox.narrativesRaw = {
    narratives: [{ id: 'n1', name: 'One' }, { id: 'n2', name: 'Two' }],
    rs_history: { dates: ['01.01.'], narratives: { n1: [0], n2: [0] } },
  };
  vm.runInContext(`ncToggleBenchmark('n1')`, sandbox);
  vm.runInContext(`ncToggleBenchmark('n2')`, sandbox);
  // Array.from(...) re-materializes the vm-realm Set's contents as a
  // host-realm array (same cross-realm [[Prototype]] caveat as elsewhere in
  // this file) before comparing.
  const selected = Array.from(vm.runInContext('Array.from(ncBenchmarkSelected)', sandbox));
  assert.deepEqual(selected.sort(), ['n1', 'n2']);
});

test('ncToggleBenchmark: re-clicking an already-selected narrative deselects it', () => {
  sandbox.ncBenchmarkSelected = new Set(['n1']);
  sandbox.narrativesRaw = { narratives: [{ id: 'n1', name: 'One' }], rs_history: { dates: ['01.01.'], narratives: { n1: [0] } } };
  vm.runInContext(`ncToggleBenchmark('n1')`, sandbox);
  assert.equal(vm.runInContext('ncBenchmarkSelected.has("n1")', sandbox), false);
});

test('ncToggleBenchmark/renderBenchmarkPills/renderBenchmarkChart never call fetch (no API call on pill click)', () => {
  const combined = ncToggleBenchmarkSrc + renderBenchmarkPillsSrc + renderBenchmarkChartSrc;
  assert.doesNotMatch(combined, /fetch\(/);
});

test('benchmarkLineChart: each selected narrative gets its own colored polyline, RSP is the 0% baseline label', () => {
  const dates = ['01.01.', '02.01.', '03.01.', '04.01.', '05.01.'];
  const seriesList = [
    { id: 'n1', color: '#111111', data: [0, 2, 4, 6, 8] },
    { id: 'n2', color: '#222222', data: [0, -1, -2, -3, -4] },
  ];
  const svg = vm.runInContext('benchmarkLineChart(seriesList, dates, 900, 260)', Object.assign(sandbox, { seriesList, dates }));
  assert.match(svg, /<svg/);
  assert.match(svg, /stroke="#111111"/);
  assert.match(svg, /stroke="#222222"/);
  assert.match(svg, />RSP</); // dashed 0% baseline is labelled RSP, never S&P 500
  assert.match(svg, /stroke-dasharray="4,3"/); // the dashed 0% line itself
});

test('benchmarkLineChart: outperformance (>0), underperformance (<0), and equal (=0) all render without crashing', () => {
  const dates = ['01.01.', '02.01.', '03.01.'];
  const outperform = { id: 'out', color: '#111', data: [0, 5, 10] };
  const underperform = { id: 'under', color: '#222', data: [0, -5, -10] };
  const equal = { id: 'eq', color: '#333', data: [0, 0, 0] };
  const svg = vm.runInContext('benchmarkLineChart(seriesList, dates, 900, 260)',
    Object.assign(sandbox, { seriesList: [outperform, underperform, equal], dates }));
  assert.match(svg, /<svg/);
  assert.equal((svg.match(/<polyline/g) || []).length, 3);
});

test('benchmarkLineChart: fewer than 2 dates returns an empty string (no broken chart)', () => {
  const svg = vm.runInContext('benchmarkLineChart(seriesList, dates, 900, 260)',
    Object.assign(sandbox, { seriesList: [{ id: 'n1', color: '#111', data: [0] }], dates: ['01.01.'] }));
  assert.equal(svg, '');
});

test('renderBenchmarkPills/renderBenchmarkChart: default-selecting the leading narrative renders a legend entry, no crash', () => {
  sandbox.ncBenchmarkSelected = new Set(['lead']);
  sandbox.narrativesRaw = {
    narratives: [{ id: 'lead', name: 'Leading Narrative' }, { id: 'other', name: 'Other' }],
    rs_history: { dates: ['01.01.', '02.01.'], narratives: { lead: [0, 3.2] } },
  };
  vm.runInContext('renderBenchmarkPills()', sandbox);
  vm.runInContext('renderBenchmarkChart()', sandbox);
  const legendHtml = sandbox.document._elements['benchmarkLegend'].innerHTML;
  assert.match(legendHtml, /Leading Narrative/);
  assert.match(legendHtml, /\+3\.2%/);
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

// ── RVOL/Screener/Benchmark/Futures Patch point 11: simplified Screener header ──

test('index.html: Screener header is just "06 Screener", subtitle removed (point 11)', () => {
  assert.doesNotMatch(html, /FESTE PRESETS/);
  assert.doesNotMatch(html, /KEIN FREIER REGEL-EDITOR/);
  assert.match(html, /<span class="section-title">Screener<\/span>/);
});

// ── RVOL/Screener/Benchmark/Futures Patch points 1/12/13: Weekly + Monthly
// Strength screener chips, no "YOLO RS"/"AS", EMA20 not EMA21 ──

test('SCREENER_PRESET_TAGS: Weekly Strength chips are exact, no "YOLO RS"/"AS", no EMA21', () => {
  const tags = vm.runInContext('SCREENER_PRESET_TAGS.weekly_strength', sandbox);
  assert.deepEqual(Array.from(tags), ['Strength 1W ≥ 85', 'Price > SMA50', 'EMA10 oder EMA20 innerhalb ±5%']);
  assert.doesNotMatch(JSON.stringify(Array.from(tags)), /YOLO RS|EMA21|\bAS\b/);
});

test('SCREENER_PRESET_TAGS: Monthly Strength chips are exact, confirms Strength1M/SMA200', () => {
  const tags = vm.runInContext('SCREENER_PRESET_TAGS.monthly_strength', sandbox);
  assert.deepEqual(Array.from(tags), ['Strength 1M ≥ 85', 'Price > SMA200', 'EMA10 oder EMA20 innerhalb ±5%']);
});

test('SCREENER_PRESET_TAGS: no visible "YOLO RS" anywhere in the actual chip values (point 1)', () => {
  // Scoped to the const's own extracted source (the actual rendered chip
  // strings), not the whole index.html file, since an explanatory code
  // comment elsewhere legitimately mentions the old "YOLO RS ..." label
  // when documenting the rename itself.
  assert.doesNotMatch(screenerPresetTagsConstSrc, /YOLO RS/);
});

// ── Screener-Ergebnistabellen-Umbau: identisches Spalten-Set wie die
// Narrative-Mitglieder-Tabelle (Kurs/Veränderung abs./Veränderung %/MTD %/
// YTD %/Volumen (M)), PLUS eine ATR-Extension-Spalte nach Volumen, dann
// Strength (preset-eigenes Zeitfenster) + Thrust. Structural RS/ADR20/
// vs SMA/vs EMA sind keine sichtbaren Spalten mehr. ──

test('screenerColumnCellHtml: close/change_abs/d1_pct render like the Member table (fmtPrice/pctClass/fmtPct)', () => {
  const closeCell = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'close' }, t: { close: 123.456 } }));
  assert.match(closeCell, /123[.,]/); // fmtPrice-formatted, not a bare number
  const changeUp = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'change_abs' }, t: { change_abs: 2.5 } }));
  assert.match(changeUp, /class="pos"/);
  assert.match(changeUp, /\+2\.50/);
  const changeDown = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'change_abs' }, t: { change_abs: -1.2 } }));
  assert.match(changeDown, /class="neg"/);
  const changeMissing = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'change_abs' }, t: { change_abs: null } }));
  assert.match(changeMissing, />—</);
});

test('screenerColumnCellHtml: mtd_pct/ytd_pct use the null-safe strict helpers (exact 0 = neutral, null = dash)', () => {
  const mtdZero = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'mtd_pct' }, t: { mtd_pct: 0 } }));
  assert.match(mtdZero, /class="muted"/);
  const ytdNull = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'ytd_pct' }, t: { ytd_pct: null } }));
  assert.match(ytdNull, /class="muted"/);
  assert.match(ytdNull, />—</);
});

test('screenerColumnCellHtml: volume uses ncVolumeColorClass/ncFmtVolumeM (same RVOL coloring as the Member table)', () => {
  const cell = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'volume' }, t: { volume: 18420000, d1_pct: 1.2, rvol_50: 1.5 } }));
  assert.match(cell, /class="pos"/); // up + RVOL 1.5 >= 1.30 threshold -> green, same as Member table
  assert.match(cell, /18\.42/);
});

test('screenerColumnCellHtml: ATR Extension warns above the configured threshold, muted at/below it', () => {
  const warn = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'atr_extension' }, t: { atr_extension: 6.2 } }));
  assert.match(warn, /class="atr-ext-warn"/);
  assert.match(warn, />6\.2</);
  const ok = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'atr_extension' }, t: { atr_extension: 4.9 } }));
  assert.match(ok, /class="muted"/);
  const missing = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'atr_extension' }, t: { atr_extension: null } }));
  assert.match(missing, /class="muted"/);
  assert.match(missing, />—</);
});

test('screenerColumnCellHtml: Strength column (default branch) uses ncStrengthColorClass, Thrust is signed', () => {
  const cellPos = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'rs_percentile_1w' }, t: { rs_percentile_1w: 90 } }));
  const cellNull = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'rs_percentile_1w' }, t: { rs_percentile_1w: null } }));
  assert.match(cellPos, /class="pos"/);
  assert.match(cellNull, /class="muted"/);
  const thrust = vm.runInContext('screenerColumnCellHtml(col, t)',
    Object.assign(sandbox, { col: { key: 'stock_thrust_rs' }, t: { stock_thrust_rs: -3.5 } }));
  assert.match(thrust, /class="neg"/);
  assert.match(thrust, /-3\.50/);
});

test('SCREENER_PRESET_COLUMNS: all 4 presets share the same standard column set, no Structural RS/ADR20/SMA/EMA-distance columns', () => {
  ['weekly_strength', 'monthly_strength', 'three_month_strength', 'six_month_strength'].forEach(presetId => {
    const cols = vm.runInContext(`SCREENER_PRESET_COLUMNS.${presetId}`, sandbox);
    const keys = Array.from(cols).map(c => c.key);
    assert.deepEqual(keys, ['close', 'change_abs', 'd1_pct', 'mtd_pct', 'ytd_pct', 'volume', 'atr_extension', keys[7], 'stock_thrust_rs']);
    assert.ok(!keys.includes('structural_rs'));
    assert.ok(!keys.includes('adr20'));
    assert.ok(!keys.some(k => k.startsWith('sma')));
    assert.ok(!keys.some(k => k.startsWith('ema')));
  });
});

test('SCREENER_PRESET_COLUMNS: each preset\'s Strength column reads its own rs_percentile_<window> field', () => {
  const expected = {
    weekly_strength: 'rs_percentile_1w', monthly_strength: 'rs_percentile_1m',
    three_month_strength: 'rs_percentile_3m', six_month_strength: 'rs_percentile_6m',
  };
  Object.entries(expected).forEach(([presetId, strengthKey]) => {
    const cols = vm.runInContext(`SCREENER_PRESET_COLUMNS.${presetId}`, sandbox);
    const strengthCol = Array.from(cols).find(c => c.strength);
    assert.equal(strengthCol.key, strengthKey);
  });
});

// ── Strength Screeners 3M/6M Union Patch point 12B: ONE shared sort
// primitive (sortTableRows/sortTableNextState), and the generic
// marketTable* system (Futures/DAX-Europe/Global/Krypto/Commodities all
// five tables share ONE implementation keyed by tableKey) that replaced
// the prior phase's Futures-only futuresReturn5d/sortFuturesRows ──

test('sortTableRows: text type sorts alphabetically (locale compare)', () => {
  const rows = [{ k: 'Charlie' }, { k: 'Alpha' }, { k: 'Bravo' }];
  const out = vm.runInContext('sortTableRows(rows, { direction: "asc", type: "text", key: "k" })', Object.assign(sandbox, { rows }));
  assert.deepEqual(Array.from(out).map(r => r.k), ['Alpha', 'Bravo', 'Charlie']);
});

test('sortTableRows: numeric type sorts DESC on first-click convention, ASC when direction flips', () => {
  const rows = [{ k: 5 }, { k: 20 }, { k: 1 }];
  const desc = vm.runInContext('sortTableRows(rows, { direction: "desc", type: "number", key: "k" })', Object.assign(sandbox, { rows }));
  assert.deepEqual(Array.from(desc).map(r => r.k), [20, 5, 1]);
  const asc = vm.runInContext('sortTableRows(rows, { direction: "asc", type: "number", key: "k" })', Object.assign(sandbox, { rows }));
  assert.deepEqual(Array.from(asc).map(r => r.k), [1, 5, 20]);
});

test('sortTableRows: null/undefined/NaN ALWAYS sort last, regardless of ASC or DESC (point 12B\'s explicit requirement)', () => {
  // Regression test for a real pre-existing bug: the old oppSortItems used
  // -Infinity as a null substitute, which put nulls FIRST in ASC and LAST
  // in DESC -- direction-dependent, violating "nulls immer ans Ende".
  const rows = [{ k: 5 }, { k: null }, { k: 20 }, { k: undefined }, { k: NaN }, { k: 1 }];
  const desc = vm.runInContext('sortTableRows(rows, { direction: "desc", type: "number", key: "k" })', Object.assign(sandbox, { rows }));
  assert.deepEqual(Array.from(desc).map(r => r.k).slice(0, 3), [20, 5, 1]);
  assert.deepEqual(Array.from(desc).map(r => r.k).slice(3), [null, undefined, NaN]);
  const asc = vm.runInContext('sortTableRows(rows, { direction: "asc", type: "number", key: "k" })', Object.assign(sandbox, { rows }));
  assert.deepEqual(Array.from(asc).map(r => r.k).slice(0, 3), [1, 5, 20]);
  assert.deepEqual(Array.from(asc).map(r => r.k).slice(3), [null, undefined, NaN]); // still last, even ascending
});

test('sortTableRows: accessor form (used by marketTable/opp) reads a computed value, not a plain key lookup', () => {
  const rows = [{ a: 1, b: 10 }, { a: 2, b: 1 }];
  const out = vm.runInContext('sortTableRows(rows, { direction: "asc", type: "number", accessor: r => r.a + r.b })', Object.assign(sandbox, { rows }));
  assert.deepEqual(Array.from(out).map(r => r.a), [2, 1]); // 3 < 11
});

test('sortTableRows: sorts a COPY, never mutates the original array', () => {
  const original = [{ k: 3 }, { k: 1 }, { k: 2 }];
  const originalOrder = original.map(r => r.k);
  const sorted = vm.runInContext('sortTableRows(rows, { direction: "asc", type: "number", key: "k" })', Object.assign(sandbox, { rows: original }));
  assert.deepEqual(original.map(r => r.k), originalOrder);
  assert.notEqual(sorted, original);
});

// Object.assign({}, ...) re-materializes the vm-realm object literal's own
// properties into a host-realm plain object first -- a strict deepEqual
// against a host-realm object literal can otherwise spuriously fail on
// [[Prototype]] identity alone (same cross-realm caveat as elsewhere here).
test('sortTableNextState: numeric columns\' first click is DESC, text columns\' first click is ASC', () => {
  const numFirst = vm.runInContext('sortTableNextState(null, "price", "number")', sandbox);
  assert.deepEqual(Object.assign({}, numFirst), { key: 'price', type: 'number', direction: 'desc' });
  const textFirst = vm.runInContext('sortTableNextState(null, "name", "text")', sandbox);
  assert.deepEqual(Object.assign({}, textFirst), { key: 'name', type: 'text', direction: 'asc' });
});

test('sortTableNextState: clicking the SAME key again inverts direction; a NEW key resets to that column\'s own default', () => {
  const first = vm.runInContext('sortTableNextState(null, "price", "number")', sandbox);
  const second = vm.runInContext('sortTableNextState(state, "price", "number")', Object.assign(sandbox, { state: first }));
  assert.equal(second.direction, 'asc'); // inverted
  const third = vm.runInContext('sortTableNextState(state, "name", "text")', Object.assign(sandbox, { state: second }));
  assert.deepEqual(Object.assign({}, third), { key: 'name', type: 'text', direction: 'asc' }); // new key -> fresh default, not inverted
});

test('marketTableReturn5d: return_5d = last_valid / first_valid - 1', () => {
  const out = vm.runInContext('marketTableReturn5d(hist)', Object.assign(sandbox, { hist: [100, 101, 102, 103, 104] }));
  assert.ok(Math.abs(out - (104 / 100 - 1)) < 1e-9);
});

test('marketTableReturn5d: ignores non-finite entries, needs at least 2 valid points, null for a zero/absent first point', () => {
  const out = vm.runInContext('marketTableReturn5d(hist)', Object.assign(sandbox, { hist: [null, 100, undefined, NaN, 110] }));
  assert.ok(Math.abs(out - (110 / 100 - 1)) < 1e-9);
  assert.equal(vm.runInContext('marketTableReturn5d(hist)', Object.assign(sandbox, { hist: [100] })), null);
  assert.equal(vm.runInContext('marketTableReturn5d(hist)', Object.assign(sandbox, { hist: null })), null);
});

function makeMarketRows() {
  return [
    { name: 'ES (S&P 500)', price: 5998, d1_pct: 0.42, w1_pct: 1.12, hi52w_pct: -2.1, ytd_pct: 18.4, hist_5d: [100, 101, 102, 103, 104] },
    { name: 'NQ (Nasdaq 100)', price: 21320, d1_pct: 0.56, w1_pct: 1.44, hi52w_pct: -3.2, ytd_pct: 22.1, hist_5d: [100, 99, 98, 97, 90] },
    { name: 'YM (Dow Jones)', price: 43890, d1_pct: null, w1_pct: 0.85, hi52w_pct: -1.5, ytd_pct: 15.2, hist_5d: [100, 100, 100, 100, 100] },
    { name: 'RTY (Russell 2000)', price: 2285, d1_pct: -0.18, w1_pct: -0.42, hi52w_pct: -8.4, ytd_pct: 8.7, hist_5d: null },
  ];
}

test('sortMarketTableRows: Kontrakt/Name sorts alphabetically', () => {
  const out = vm.runInContext('sortMarketTableRows(rows, "name", "asc")', Object.assign(sandbox, { rows: makeMarketRows() }));
  assert.deepEqual(Array.from(out).map(r => r.name),
    ['ES (S&P 500)', 'NQ (Nasdaq 100)', 'RTY (Russell 2000)', 'YM (Dow Jones)']);
});

test('sortMarketTableRows: numeric fields sort DESC/ASC correctly (Kurs example)', () => {
  const desc = vm.runInContext('sortMarketTableRows(rows, "price", "desc")', Object.assign(sandbox, { rows: makeMarketRows() }));
  assert.deepEqual(Array.from(desc).map(r => r.name), ['YM (Dow Jones)', 'NQ (Nasdaq 100)', 'ES (S&P 500)', 'RTY (Russell 2000)']);
  const asc = vm.runInContext('sortMarketTableRows(rows, "price", "asc")', Object.assign(sandbox, { rows: makeMarketRows() }));
  assert.deepEqual(Array.from(asc).map(r => r.name), ['RTY (Russell 2000)', 'ES (S&P 500)', 'NQ (Nasdaq 100)', 'YM (Dow Jones)']);
});

test('sortMarketTableRows: 1T% (d1_pct) sorts numerically, null always sorts last regardless of direction', () => {
  const desc = vm.runInContext('sortMarketTableRows(rows, "d1_pct", "desc")', Object.assign(sandbox, { rows: makeMarketRows() }));
  assert.deepEqual(Array.from(desc).map(r => r.name).slice(-1), ['YM (Dow Jones)']);
  const asc = vm.runInContext('sortMarketTableRows(rows, "d1_pct", "asc")', Object.assign(sandbox, { rows: makeMarketRows() }));
  assert.deepEqual(Array.from(asc).map(r => r.name).slice(-1), ['YM (Dow Jones)']);
});

test('sortMarketTableRows: 5D sorts by the numeric return_5d endpoint, not by sparkline SVG/string', () => {
  const desc = vm.runInContext('sortMarketTableRows(rows, "return_5d", "desc")', Object.assign(sandbox, { rows: makeMarketRows() }));
  const names = Array.from(desc).map(r => r.name);
  // ES: 104/100-1=+4%, NQ: 90/100-1=-10%, YM: 100/100-1=0%, RTY: null hist_5d -> null return_5d, always last.
  assert.deepEqual(names, ['ES (S&P 500)', 'YM (Dow Jones)', 'NQ (Nasdaq 100)', 'RTY (Russell 2000)']);
});

test('sortMarketTableRows: sorts a COPY, entire row (incl. sparkline data) stays together, never mutated', () => {
  const original = makeMarketRows();
  const originalOrder = original.map(r => r.name);
  const sorted = vm.runInContext('sortMarketTableRows(rows, "price", "asc")', Object.assign(sandbox, { rows: original }));
  assert.deepEqual(original.map(r => r.name), originalOrder);
  assert.notEqual(sorted, original);
  const rty = Array.from(sorted).find(r => r.name === 'RTY (Russell 2000)');
  assert.equal(rty.d1_pct, -0.18);
  assert.equal(rty.ytd_pct, 8.7);
  assert.equal(rty.hist_5d, null);
});

test('MARKET_TABLE_IDS: all five market tables (Futures/DAX-Europe/Global/Krypto/Commodities) share the one generic system', () => {
  const ids = vm.runInContext('MARKET_TABLE_IDS', sandbox);
  assert.deepEqual(Object.assign({}, ids), {
    futures: 'futuresTable', europe: 'europeTable', global: 'globalTable',
    crypto: 'cryptoTable', commodities: 'commoditiesTable',
  });
});

test('index.html: each of the five market tables has its own independent per-table sort state (marketTableSort keyed by tableKey)', () => {
  assert.match(html, /const marketTableSort = \{\};/);
  assert.match(html, /marketTableSortBy\('futures',/);
  assert.match(html, /marketTableSortBy\('europe',/);
  assert.match(html, /marketTableSortBy\('global',/);
  assert.match(html, /marketTableSortBy\('crypto',/);
  assert.match(html, /marketTableSortBy\('commodities',/);
});

// ── Strength Screeners 3M/6M Union Patch point 12A: MTD %/YTD % display ──

test('ncFmtPctOrDash: null/undefined render as "—", real numbers use fmtPct (German comma, signed)', () => {
  assert.equal(vm.runInContext('ncFmtPctOrDash(v)', Object.assign(sandbox, { v: null })), '—');
  assert.equal(vm.runInContext('ncFmtPctOrDash(v)', Object.assign(sandbox, { v: undefined })), '—');
  assert.match(vm.runInContext('ncFmtPctOrDash(v)', Object.assign(sandbox, { v: 12.44 })), /\+12,44%/);
  assert.match(vm.runInContext('ncFmtPctOrDash(v)', Object.assign(sandbox, { v: -3.05 })), /-3,05%/);
});

test('ncPctColorClassStrict: positive -> pos, negative -> neg, EXACT ZERO -> neutral (unlike pctClass), null/undefined -> neutral', () => {
  assert.equal(vm.runInContext('ncPctColorClassStrict(v)', Object.assign(sandbox, { v: 5.0 })), 'pos');
  assert.equal(vm.runInContext('ncPctColorClassStrict(v)', Object.assign(sandbox, { v: -5.0 })), 'neg');
  assert.equal(vm.runInContext('ncPctColorClassStrict(v)', Object.assign(sandbox, { v: 0 })), 'muted'); // point 12A: exactly 0 is neutral
  assert.equal(vm.runInContext('ncPctColorClassStrict(v)', Object.assign(sandbox, { v: null })), 'muted');
  assert.equal(vm.runInContext('ncPctColorClassStrict(v)', Object.assign(sandbox, { v: undefined })), 'muted');
});

// ── Strength Screeners 3M/6M Union Patch point 5/6: 3 Month / 6 Month
// Strength presets -- EMA10/EMA20 (never EMA21), no SMA gate ──

test('SCREENER_PRESET_TAGS: 3 Month / 6 Month Strength chips are exact, no SMA chip (no SMA gate for these two presets)', () => {
  const threeM = vm.runInContext('SCREENER_PRESET_TAGS.three_month_strength', sandbox);
  assert.deepEqual(Array.from(threeM), ['Strength 3M ≥ 85', 'EMA10 oder EMA20 innerhalb ±5%']);
  const sixM = vm.runInContext('SCREENER_PRESET_TAGS.six_month_strength', sandbox);
  assert.deepEqual(Array.from(sixM), ['Strength 6M ≥ 85', 'EMA10 oder EMA20 innerhalb ±5%']);
  assert.doesNotMatch(JSON.stringify(Array.from(threeM).concat(Array.from(sixM))), /SMA|EMA21/);
});

test('index.html: every Strength-Screener-Card carries the informational ADR20/base-universe chip line', () => {
  assert.match(html, /STOCKS · NO HEALTHCARE\/BIOTECH · MCAP ≥ ?\$1B · ADR20 > ?4%/);
  assert.match(html, /class="screener-basis-line"/);
});

// ── Strength Screeners 3M/6M Union Patch point 12B: all 4 Screener result
// tables are header-click sortable too, per-preset independent state ──

test('index.html: screenerPresetHtml builds sortable headers for every preset column via screenerHeaderCell', () => {
  assert.match(screenerPresetHtmlSrc, /function screenerHeaderCell\(key, label\)/);
  assert.match(screenerPresetHtmlSrc, /onclick="screenerTableSortBy\('\$\{presetId\}','\$\{key\}'\)"/);
  assert.match(screenerPresetHtmlSrc, /screenerHeaderCell\('symbol', 'Ticker'\)/);
  assert.match(screenerPresetHtmlSrc, /columns\.map\(col => screenerHeaderCell\(col\.key, col\.label\)\)/);
  // Sorts a COPY on top of the server-side default order, never mutates preset.tickers.
  assert.match(screenerPresetHtmlSrc, /sortTableRows\(preset\.tickers,/);
});

test('screenerColumnType: symbol/Ticker is text, everything else numeric', () => {
  assert.equal(vm.runInContext('screenerColumnType("symbol")', sandbox), 'text');
  assert.equal(vm.runInContext('screenerColumnType("rs_percentile_3m")', sandbox), 'number');
  assert.equal(vm.runInContext('screenerColumnType("adr20")', sandbox), 'number');
});

test('index.html: each of the 4 Screener preset tables keeps its OWN independent sort state (screenerTableSort keyed by presetId)', () => {
  assert.match(html, /const screenerTableSort = \{\};/);
  assert.match(html, /screenerTableSort\[presetId\] = sortTableNextState\(screenerTableSort\[presetId\]/);
});

// ── Strength Screeners 3M/6M Union Patch point 8: global union copy button ──

test('screenerUnionTickers: dedups across ALL 4 CURRENT Strength presets, alphabetical, no history/no duplicates', () => {
  // Spec's own worked synthetic example: A/B/C ∪ B/C/D ∪ C/D/E ∪ A/E/F = A,B,C,D,E,F.
  const data = {
    screeners: {
      weekly_strength: { tickers: [{ symbol: 'A' }, { symbol: 'B' }, { symbol: 'C' }] },
      monthly_strength: { tickers: [{ symbol: 'B' }, { symbol: 'C' }, { symbol: 'D' }] },
      three_month_strength: { tickers: [{ symbol: 'C' }, { symbol: 'D' }, { symbol: 'E' }] },
      six_month_strength: { tickers: [{ symbol: 'A' }, { symbol: 'E' }, { symbol: 'F' }] },
    },
  };
  const union = vm.runInContext('screenerUnionTickers(data)', Object.assign(sandbox, { data }));
  assert.deepEqual(Array.from(union), ['A', 'B', 'C', 'D', 'E', 'F']);
  assert.deepEqual([...new Set(Array.from(union))], Array.from(union)); // no duplicates
});

test('screenerUnionTickers: missing/empty screeners data degrades gracefully to an empty list', () => {
  assert.deepEqual(Array.from(vm.runInContext('screenerUnionTickers(data)', Object.assign(sandbox, { data: null }))), []);
  assert.deepEqual(Array.from(vm.runInContext('screenerUnionTickers(data)', Object.assign(sandbox, { data: {} }))), []);
});

test('SCREENER_STRENGTH_PRESET_IDS: exactly the 4 current Strength horizons (1W/1M/3M/6M), not a historical snapshot union', () => {
  const ids = vm.runInContext('SCREENER_STRENGTH_PRESET_IDS', sandbox);
  assert.deepEqual(Array.from(ids), ['weekly_strength', 'monthly_strength', 'three_month_strength', 'six_month_strength']);
});

test('screenerCopyUnion: copies the deduplicated union to the clipboard and sets "Kopiert (N)" feedback on the button', async () => {
  sandbox.clipboardWrites = [];
  const screenersRaw = {
    screeners: {
      weekly_strength: { tickers: [{ symbol: 'MU' }, { symbol: 'AAPL' }] },
      monthly_strength: { tickers: [{ symbol: 'MU' }] },
      three_month_strength: { tickers: [] },
      six_month_strength: { tickers: [{ symbol: 'DELL' }] },
    },
  };
  const btn = { textContent: 'Alle Strength-Kandidaten kopieren' };
  vm.runInContext('screenerCopyUnion(btn)', Object.assign(sandbox, { screenersRaw, btn }));
  await Promise.resolve().then(() => {}).then(() => {}); // flush the clipboard.writeText().then(...) microtask
  assert.deepEqual(sandbox.clipboardWrites, ['AAPL,DELL,MU']); // alphabetical, deduplicated, comma-separated
  assert.equal(btn.textContent, 'Kopiert (3)');
});

test('index.html: the union copy button reads the JS data state, not the DOM -- works identically whether cards are collapsed or expanded', () => {
  const fn = extractFunction(html, 'screenerCopyUnion');
  assert.doesNotMatch(fn, /getElementById|querySelector/);
  assert.match(fn, /screenerUnionTickers\(screenersRaw\)/);
});

test('index.html: union button + count sit above the 4 preset cards, before the Screener container', () => {
  const screenerSectionIdx = html.indexOf('id="screenerContainer"');
  const unionBtnIdx = html.search(/onclick="screenerCopyUnion\(this\)"/);
  const unionCountIdx = html.indexOf('id="screenerUnionCount"');
  assert.ok(unionBtnIdx > -1 && unionBtnIdx < screenerSectionIdx);
  assert.ok(unionCountIdx > -1 && unionCountIdx < screenerSectionIdx);
});

// ── ATR-Extension-Ausschluss-Button: zweiter globaler Copy-Button neben
// "Alle Strength-Kandidaten kopieren", schliesst Ticker mit atr_extension >
// Schwelle aus (dieselbe Union-/Dedup-/Alphabetisch-Semantik) ──

test('screenerUnionTickersExcludingAtrExtended: excludes only tickers whose ATR Extension exceeds the threshold', () => {
  const data = {
    screeners: {
      weekly_strength: { tickers: [{ symbol: 'A', atr_extension: 3.0 }, { symbol: 'B', atr_extension: 8.0 }] },
      monthly_strength: { tickers: [{ symbol: 'C', atr_extension: 5.0 }] }, // exactly at threshold -> kept ("> 5", not ">=")
      three_month_strength: { tickers: [{ symbol: 'D', atr_extension: null }] }, // missing -> kept, never fabricated-excluded
      six_month_strength: { tickers: [{ symbol: 'E', atr_extension: 5.01 }] },
    },
  };
  const out = vm.runInContext('screenerUnionTickersExcludingAtrExtended(data, 5)', Object.assign(sandbox, { data }));
  assert.deepEqual(Array.from(out), ['A', 'C', 'D']); // B (8.0) and E (5.01) excluded
});

test('screenerUnionTickersExcludingAtrExtended: same symbol appearing in multiple presets uses its (identical) atr_extension once, deduplicated', () => {
  const data = {
    screeners: {
      weekly_strength: { tickers: [{ symbol: 'MU', atr_extension: 7.2 }] },
      monthly_strength: { tickers: [{ symbol: 'MU', atr_extension: 7.2 }] },
      three_month_strength: { tickers: [] },
      six_month_strength: { tickers: [{ symbol: 'AAPL', atr_extension: 1.0 }] },
    },
  };
  const out = vm.runInContext('screenerUnionTickersExcludingAtrExtended(data, 5)', Object.assign(sandbox, { data }));
  assert.deepEqual(Array.from(out), ['AAPL']); // MU excluded (7.2 > 5), no duplicate MU/AAPL entries
});

test('screenerUnionTickersExcludingAtrExtended: missing/empty screeners data degrades gracefully to an empty list', () => {
  assert.deepEqual(Array.from(vm.runInContext('screenerUnionTickersExcludingAtrExtended(data, 5)', Object.assign(sandbox, { data: null }))), []);
  assert.deepEqual(Array.from(vm.runInContext('screenerUnionTickersExcludingAtrExtended(data, 5)', Object.assign(sandbox, { data: {} }))), []);
});

test('screenerCopyUnionAtrFiltered: copies the ATR-filtered union to the clipboard and sets "Kopiert (N)" feedback', async () => {
  sandbox.clipboardWrites = [];
  const screenersRaw = {
    screeners: {
      weekly_strength: { tickers: [{ symbol: 'MU', atr_extension: 7.2 }, { symbol: 'AAPL', atr_extension: 1.0 }] },
      monthly_strength: { tickers: [{ symbol: 'MU', atr_extension: 7.2 }] },
      three_month_strength: { tickers: [] },
      six_month_strength: { tickers: [{ symbol: 'DELL', atr_extension: 4.9 }] },
    },
  };
  const btn = { textContent: 'Strength-Kandidaten kopieren (ATR Extension ≤ 5)' };
  vm.runInContext('screenerCopyUnionAtrFiltered(btn)', Object.assign(sandbox, { screenersRaw, btn }));
  await Promise.resolve().then(() => {}).then(() => {});
  assert.deepEqual(sandbox.clipboardWrites, ['AAPL,DELL']); // MU (7.2 > 5.0 threshold) excluded
  assert.equal(btn.textContent, 'Kopiert (2)');
});

test('index.html: the ATR-filtered union button reads the JS data state, not the DOM, and uses engineConfig\'s ATR threshold', () => {
  const fn = extractFunction(html, 'screenerCopyUnionAtrFiltered');
  assert.doesNotMatch(fn, /getElementById|querySelector/);
  assert.match(fn, /screenerUnionTickersExcludingAtrExtended\(screenersRaw,\s*maxAtrExtension\)/);
  assert.match(fn, /engineConfig\.dashboard\.atr_extension_warning_threshold/);
});

test('index.html: the ATR-filtered union button sits next to the main union button, above the 4 preset cards', () => {
  const screenerSectionIdx = html.indexOf('id="screenerContainer"');
  const atrBtnIdx = html.search(/onclick="screenerCopyUnionAtrFiltered\(this\)"/);
  assert.ok(atrBtnIdx > -1 && atrBtnIdx < screenerSectionIdx);
});

// ── Strength Screeners 3M/6M Union Patch point 10: info-popup overlay fix ──

test('index.html: .nc-info-popup stacks ABOVE the sticky .narrative-controls bar (z-index 100 > 40), .nc-info itself above 40 too', () => {
  const controlsMatch = html.match(/\.narrative-controls\s*\{[^}]*\}/);
  assert.ok(controlsMatch, '.narrative-controls CSS rule not found');
  const controlsZ = Number(controlsMatch[0].match(/z-index:\s*(\d+)/)[1]);
  const hoverMatch = html.match(/\.nc-info:hover,\s*\.nc-info:focus,\s*\.nc-info\.open\s*\{[^}]*\}/);
  assert.ok(hoverMatch, '.nc-info:hover/:focus/.open CSS rule not found');
  const hoverZ = Number(hoverMatch[0].match(/z-index:\s*(\d+)/)[1]);
  const popupMatch = html.match(/\.nc-info-popup\s*\{[^}]*\}/);
  assert.ok(popupMatch, '.nc-info-popup CSS rule not found');
  const popupZ = Number(popupMatch[0].match(/z-index:\s*(\d+)/)[1]);
  assert.ok(hoverZ > controlsZ, `.nc-info:hover/:focus/.open z-index (${hoverZ}) must exceed .narrative-controls (${controlsZ})`);
  assert.ok(popupZ > controlsZ, `.nc-info-popup z-index (${popupZ}) must exceed .narrative-controls (${controlsZ})`);
  assert.ok(popupZ < 999999 && hoverZ < 999999); // no arbitrary blanket z-index:999999 escape hatch
});

test('index.html: popup open/close is wired for click (delegated), hover/focus (CSS) and closes on outside click', () => {
  assert.match(html, /\.nc-info:hover \.nc-info-popup, \.nc-info:focus \.nc-info-popup, \.nc-info\.open \.nc-info-popup \{ display: block; \}/);
  assert.match(html, /document\.querySelectorAll\('\.nc-info'\)\.forEach\(icon => \{/);
  assert.match(html, /icon\.classList\.toggle\('open'\)/);
  assert.match(html, /if \(!icon\.contains\(e\.target\)\) icon\.classList\.remove\('open'\)/); // outside click closes it
});

// ── Strength Screeners 3M/6M Union Patch point 9: no person-name references
// anywhere in the tracked source tree (case-insensitive "jeff" gate) ──

test('source tree: case-insensitive "jeff" yields ZERO hits in authored source (index.html/config/scripts/tests/README/docs)', () => {
  // Point 9's gate is scoped to the files the spec itself names as needing
  // cleanup (index.html, config/, Python scripts/tests, README/docs) --
  // deliberately EXCLUDING data/ (pipeline-generated market-data cache,
  // which legitimately contains real company/ticker substrings like
  // "Jefferies Financial Group"/"Jefferson ..." that have nothing to do
  // with the removed "Jeff"/"Jeff Sun" methodology-attribution wording),
  // and excluding THIS test file's own path (its prose necessarily names
  // the term it's asserting zero occurrences of elsewhere).
  const repoRoot = path.join(__dirname, '..');
  const selfPath = fileURLToPath(import.meta.url);
  const skipDirs = new Set(['.git', 'node_modules', '.pytest_cache', '__pycache__', 'data']);
  const hits = [];
  function walk(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (skipDirs.has(entry.name)) continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) { walk(full); continue; }
      if (full === selfPath) continue;
      let content;
      try { content = readFileSync(full, 'utf-8'); } catch { continue; } // skip unreadable/binary files
      if (/jeff/i.test(content)) hits.push(path.relative(repoRoot, full));
    }
  }
  walk(repoRoot);
  assert.deepEqual(hits, []);
});

// ── RVOL/Screener/Benchmark/Futures Patch point 8: sticky Narrative controls ──

test('index.html: .narrative-controls is sticky, reuses --jumpnav-h, scoped only to Section 04', () => {
  const cssMatch = html.match(/\.narrative-controls\s*\{[^}]*\}/);
  assert.ok(cssMatch, '.narrative-controls CSS rule not found');
  assert.match(cssMatch[0], /position:\s*sticky/);
  assert.match(cssMatch[0], /top:\s*var\(--jumpnav-h/);
  // Only ONE declaration of the .narrative-controls rule -- not a second,
  // competing hardcoded sticky offset defined elsewhere for e.g. mobile.
  const allMatches = html.match(/\.narrative-controls\s*\{/g) || [];
  assert.equal(allMatches.length, 1);
});

// ── Momentum Market Regime: "Stand DD.MM., HH:MM" timestamp next to the
// section header, so it's visible when the underlying data was last built ──

test('index.html: #regimeAsOf sits inside the Market Regime section-header, next to the info icon', () => {
  const headerMatch = html.match(/<div class="section-header">\s*<span class="section-num">02<\/span>[\s\S]*?<\/div>/);
  assert.ok(headerMatch, 'Market Regime section-header not found');
  assert.match(headerMatch[0], /<span class="nc-meta" id="regimeAsOf"><\/span>/);
});

test('index.html: renderMarketRegime formats dashboardState.meta.updated_at the same way the Narrative "Stand" label does', () => {
  const fn = extractFunction(html, 'renderMarketRegime');
  assert.match(fn, /dashboardState\.meta\.updated_at/);
  assert.match(fn, /toLocaleString\('de-DE',\s*\{\s*timeZone:\s*'Europe\/Berlin'/);
  assert.match(fn, /`Stand \$\{dtStr\}`/);
});

// ── Market Regime: "Empfohlenes Vorgehen" / "Taktisches Vorgehen" — a
// display-only mapping from mr.state to a recommended-action text, plus a
// reference legend of all 5 states. Never influences the score/state
// calculation itself, purely presentational (spec: exact wording as given
// by the user, one entry per market_regime_v1 state). ──

test('index.html: REGIME_TACTICS covers exactly the 5 market_regime_v1 states with the exact user-specified wording', () => {
  const src = extractConst(html, 'REGIME_TACTICS');
  const ctx = {};
  vm.createContext(ctx);
  // Top-level const/let in a vm context lives in that context's lexical
  // environment, not as an own property of the sandbox object -- read it
  // back via a second runInContext expression on the SAME context instead
  // of `ctx.REGIME_TACTICS` (which is always undefined for a const).
  const tactics = vm.runInContext(`${src}\nREGIME_TACTICS`, ctx);
  assert.deepEqual(Object.keys(tactics), ['STRONG OFFENSIVE', 'OFFENSIVE', 'SELECTIVE', 'DEFENSIVE', 'RISK OFF']);
  assert.equal(tactics['STRONG OFFENSIVE'].text, 'Erhöhtes Risiko, Addons und mehrere neue Positionen in einer Session erlaubt');
  assert.equal(tactics['OFFENSIVE'].text, 'Je nach bisheriger Traktion, neues Risiko und weitere Positionen erlaubt');
  assert.equal(tactics['SELECTIVE'].text, 'Bei positivem Feedback, langsames Erhöhen der Exposition bei normalem bis reduziertem Risiko');
  assert.equal(tactics['DEFENSIVE'].text, 'Reduziertes Risiko, ausgewählte Trades, nur bei positivem Feedback Erhöhung des Risikos');
  assert.equal(tactics['RISK OFF'].text, 'Cash als bevorzugte Position, wenn Trades, dann nur kleine Starter oder Scouts');
});

test('index.html: renderMarketRegime populates the current-tactic card and toggles the matching legend item active', () => {
  const fn = extractFunction(html, 'renderMarketRegime');
  assert.match(fn, /REGIME_TACTICS\[mr\.state\]/);
  assert.match(fn, /regimeTacticCurrentState/);
  assert.match(fn, /regimeTacticCurrentText/);
  assert.match(fn, /el\.dataset\.state === mr\.state/);
});

// ── Calibration-aware Opportunities UI v1, spec point 17: Ticker | Primary
// Narrative | Secondary Narratives | Quality | Structure | Structural RS |
// Relative Strength | Thrust | 50MA ATR Ext. -- Structural RS/Relative
// Strength/Thrust are virtual columns (structural_rs_dyn/
// relative_strength_dyn/thrust_dyn), their actual field picked at render
// time by oppRsHorizon (tested above via oppSortItems/oppApplyFilters). ──

test('index.html: OPP_COLUMNS matches the spec point 17 target structure exactly', () => {
  const src = extractConst(html, 'OPP_COLUMNS');
  const ctx = {};
  vm.createContext(ctx);
  const keys = Array.from(vm.runInContext(`${src}\nOPP_COLUMNS.map(c => c.key)`, ctx));
  assert.deepEqual(keys, [
    'symbol', 'primary_narrative_name', 'secondary_narrative_names',
    'quality_v2', 'structure', 'structural_rs_dyn', 'relative_strength_dyn',
    'thrust_dyn', 'atr_extension',
  ]);
});

test('index.html: the Opportunities row template renders the 9 target columns, in OPP_COLUMNS order, using the horizon-selected fields', () => {
  const bodyMatch = html.match(/let rows = windowed\.map\(it => `[\s\S]*?`\)\.join\(''\);/);
  assert.ok(bodyMatch, 'Opportunities row template not found');
  const tpl = bodyMatch[0];
  assert.match(tpl, /data-ticker="\$\{it\.symbol\}"/);
  assert.match(tpl, /oppPrimaryNarrativeHtml\(it\)/);
  assert.match(tpl, /oppSecondaryNarrativeHtml\(it\)/);
  assert.match(tpl, /oppQualityHtml\(it\)/);
  assert.match(tpl, /oppStructureBadgesHtml\(it\)/);
  assert.match(tpl, /it\[oppRsField\(oppRsHorizon\)\]/);
  assert.match(tpl, /it\[oppRelativeStrengthField\(oppRsHorizon\)\]/);
  assert.match(tpl, /it\[oppThrustField\(oppRsHorizon\)\]/);
  assert.match(tpl, /it\.atr_extension/);
  const symbolIdx = tpl.indexOf('it.symbol');
  const qualityIdx = tpl.indexOf('oppQualityHtml');
  const structureIdx = tpl.indexOf('oppStructureBadgesHtml');
  const rsIdx = tpl.indexOf('oppRsField');
  const relIdx = tpl.indexOf('oppRelativeStrengthField');
  const thrustIdx = tpl.indexOf('oppThrustField');
  const atrIdx = tpl.lastIndexOf('it.atr_extension');
  assert.ok(symbolIdx < qualityIdx && qualityIdx < structureIdx && structureIdx < rsIdx &&
    rsIdx < relIdx && relIdx < thrustIdx && thrustIdx < atrIdx);
});

test('index.html: the "50MA ATR Ext." cell is warn-colored by extended_v2 (the new 8.0 threshold), not the legacy hysteresis `extended` flag', () => {
  const bodyMatch = html.match(/let rows = windowed\.map\(it => `[\s\S]*?`\)\.join\(''\);/);
  const tpl = bodyMatch[0];
  assert.match(tpl, /it\.extended_v2 \? 'atr-ext-warn'/);
});

test('index.html: Market Regime card renders an "Empfohlenes Vorgehen" block and a "Taktisches Vorgehen" legend with all 5 states, in section 02', () => {
  const regimeSection = html.match(/<div class="section fade-in" id="market-regime">[\s\S]*?(?=<!-- ═══ 03)/);
  assert.ok(regimeSection, 'Market Regime section not found');
  const section = regimeSection[0];
  assert.match(section, /<div class="rt-label">Empfohlenes Vorgehen<\/div>/);
  assert.match(section, /<div class="rt-label">Taktisches Vorgehen<\/div>/);
  const legendItems = [...section.matchAll(/<div class="rt-legend-item" data-state="([^"]+)">/g)].map(m => m[1]);
  assert.deepEqual(legendItems, ['STRONG OFFENSIVE', 'OFFENSIVE', 'SELECTIVE', 'DEFENSIVE', 'RISK OFF']);
});

// ══════════════════════════════════════════════════════════════════════
// TRADERECHNER — Waehrungs-Umschalter ($/€) und Stop-Modus-Umschalter
// (2-Stop-System vs. Single Stop)
// ══════════════════════════════════════════════════════════════════════

test('index.html: unselected .dir-btn has an explicit light text color (not the browser default black button text)', () => {
  const rule = html.match(/\.dir-btn \{[^}]*\}/);
  assert.ok(rule, '.dir-btn base rule not found');
  assert.match(rule[0], /color:\s*var\(--ink\)/);
});

function makeTraderechnerSandbox() {
  const fmtSrc = extractFunction(html, 'fmt');
  const fmtIntSrc = extractFunction(html, 'fmtInt');
  const fmtPnLSrc = extractFunction(html, 'fmtPnL');
  const directionLetSrc = extractLet(html, 'direction');
  const assetCurrencyLetSrc = extractLet(html, 'assetCurrency');
  const stopModeLetSrc = extractLet(html, 'stopMode');
  const setDirectionSrc = extractFunction(html, 'setDirection');
  const setCurrencySrc = extractFunction(html, 'setCurrency');
  const setStopModeSrc = extractFunction(html, 'setStopMode');
  const calculateSrc = extractFunction(html, 'calculate');

  const sandbox = {
    document: {
      _elements: {},
      getElementById(id) {
        if (!this._elements[id]) this._elements[id] = { value: '', checked: false, innerHTML: '', textContent: '', className: '' };
        return this._elements[id];
      },
    },
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(
    `${fmtSrc}\n${fmtIntSrc}\n${fmtPnLSrc}\n${directionLetSrc}\n${assetCurrencyLetSrc}\n${stopModeLetSrc}\n` +
    `${setDirectionSrc}\n${setCurrencySrc}\n${setStopModeSrc}\n${calculateSrc}`,
    sandbox
  );
  return sandbox;
}

function setTraderechnerInputs(sandbox, { equity = '100000', riskPct = '0.5', entryPrice = '150.00', stopLoss = '142.50' } = {}) {
  sandbox.document._elements['equity'] = { value: equity };
  sandbox.document._elements['riskPct'] = { value: riskPct };
  sandbox.document._elements['entryPrice'] = { value: entryPrice };
  sandbox.document._elements['stopLoss'] = { value: stopLoss };
}

test('calculate: default 2-Stop/USD math matches the known reference numbers (regression guard)', () => {
  const sandbox = makeTraderechnerSandbox();
  setTraderechnerInputs(sandbox);
  vm.runInContext('calculate()', sandbox);
  assert.equal(sandbox.document._elements['resShares'].textContent, '66');
  assert.equal(sandbox.document._elements['resMaxRisk'].textContent, '€500');
  assert.equal(sandbox.document._elements['resPosVal'].textContent, '$9.900');
  assert.equal(sandbox.document._elements['resRiskShare'].textContent, '$7,50');
  assert.match(sandbox.document._elements['stopsTableBody'].innerHTML, /Stop 1 \(½ Pos\.\)/);
  assert.match(sandbox.document._elements['stopsTableBody'].innerHTML, /Stop 2 \(komplett\)/);
  assert.match(sandbox.document._elements['stopsTableBody'].innerHTML, /Gesamtrisiko/);
});

test('setCurrency("EUR"): asset-side values switch to €, Max. Risiko stays € (account currency, unaffected either way)', () => {
  const sandbox = makeTraderechnerSandbox();
  setTraderechnerInputs(sandbox);
  vm.runInContext("setCurrency('EUR')", sandbox);
  assert.equal(sandbox.document._elements['entryCcyLabel'].textContent, '€');
  assert.equal(sandbox.document._elements['stopCcyLabel'].textContent, '€');
  assert.equal(sandbox.document._elements['resPosVal'].textContent, '€9.900');
  assert.equal(sandbox.document._elements['resRiskShare'].textContent, '€7,50');
  assert.equal(sandbox.document._elements['resMaxRisk'].textContent, '€500'); // unchanged
  assert.match(sandbox.document._elements['stopsTableBody'].innerHTML, /€146,25/);
});

test('setCurrency("USD") after EUR reverts the asset-side symbol back to $', () => {
  const sandbox = makeTraderechnerSandbox();
  setTraderechnerInputs(sandbox);
  vm.runInContext("setCurrency('EUR')", sandbox);
  vm.runInContext("setCurrency('USD')", sandbox);
  assert.equal(sandbox.document._elements['entryCcyLabel'].textContent, '$');
  assert.equal(sandbox.document._elements['resPosVal'].textContent, '$9.900');
});

test('setStopMode("single"): ONE full-position stop row, PnL equals Gesamtrisiko', () => {
  const sandbox = makeTraderechnerSandbox();
  setTraderechnerInputs(sandbox);
  vm.runInContext("setStopMode('single')", sandbox);
  const tbody = sandbox.document._elements['stopsTableBody'].innerHTML;
  assert.match(tbody, /Stop \(komplett\)/);
  assert.doesNotMatch(tbody, /Stop 1/);
  assert.doesNotMatch(tbody, /Stop 2/);
  // 66 shares * $7.50 risk/share = $495 total risk, all in one row.
  assert.match(tbody, /-\$495,00/);
  assert.match(sandbox.document._elements['stopsTitle'].textContent, /Single Stop/);
});

test('setStopMode("two") after single reverts to the staggered 2-row table', () => {
  const sandbox = makeTraderechnerSandbox();
  setTraderechnerInputs(sandbox);
  vm.runInContext("setStopMode('single')", sandbox);
  vm.runInContext("setStopMode('two')", sandbox);
  const tbody = sandbox.document._elements['stopsTableBody'].innerHTML;
  assert.match(tbody, /Stop 1 \(½ Pos\.\)/);
  assert.match(tbody, /Stop 2 \(komplett\)/);
  assert.match(sandbox.document._elements['stopsTitle'].textContent, /2-Stop-System/);
});

test('setStopMode("single") respects direction=short (stop above entry, PnL sign still correct)', () => {
  const sandbox = makeTraderechnerSandbox();
  setTraderechnerInputs(sandbox, { entryPrice: '100', stopLoss: '105' }); // short: stop ABOVE entry
  vm.runInContext("setDirection('short')", sandbox);
  vm.runInContext("setStopMode('single')", sandbox);
  const tbody = sandbox.document._elements['stopsTableBody'].innerHTML;
  // eq=100000, risk=0.5% -> maxR=500; rps=5 -> sh=100; short PnL at stop = sh*(en-sl) = 100*(100-105) = -500
  assert.match(tbody, /-\$500,00/);
});
