import {test} from 'node:test';
import assert from 'node:assert/strict';
import {escapeHTML, format, selectBars} from '../assets/market-overview.mjs';

test('missing values stay unavailable, including numeric-looking strings',()=>{
  for(const v of [null, undefined, NaN, Infinity, '42']) assert.equal(format(v),'—');
  assert.equal(format(0),'0,0');
});
test('untrusted labels cannot inject markup',()=>{
  assert.equal(escapeHTML('<img onerror="x">&'), '&lt;img onerror=&quot;x&quot;&gt;&amp;');
});
test('periods anchor to session date, exclude future and invalid candles',()=>{
  const bar=date=>({date,o:1,h:2,l:1,c:2,v:10});
  const bars=['2025-12-31','2026-01-02','2026-10-01','2026-11-18','2026-11-19'].map(bar);
  bars.push({...bar('2026-11-17'),c:null});
  assert.deepEqual(selectBars(bars,'1M','2026-11-18').map(b=>b.date),['2026-11-18']);
  assert.deepEqual(selectBars(bars,'YTD','2026-11-18').map(b=>b.date),['2026-01-02','2026-10-01','2026-11-18']);
  assert.equal(selectBars(bars,'1Y','invalid').length,0);
});
