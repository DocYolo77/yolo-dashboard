export const finite = v => typeof v === 'number' && Number.isFinite(v);
export const escapeHTML = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function format(v, digits = 1) { return finite(v) ? v.toLocaleString('de-DE', {minimumFractionDigits:digits, maximumFractionDigits:digits}) : '—'; }
export function selectBars(bars, range, asOf) {
  const end = new Date(`${asOf}T12:00:00Z`);
  if (!Number.isFinite(end.getTime())) return [];
  const start = new Date(end);
  if (range === 'YTD') start.setUTCMonth(0, 1);
  else start.setUTCMonth(start.getUTCMonth() - ({'1M':1,'3M':3,'6M':6,'1Y':12}[range] ?? 12));
  return bars.filter(b => /^\d{4}-\d{2}-\d{2}$/.test(b.date) && b.date >= start.toISOString().slice(0,10) && b.date <= asOf && ['o','h','l','c','v'].every(k => finite(b[k])));
}
const tone = v => finite(v) ? (v >= 50 ? 'positive' : v >= 30 ? 'neutral' : 'negative') : 'muted';
const signed = v => finite(v) ? `${v > 0 ? '+' : ''}${format(v)}` : '—';
const pct = v => finite(v) ? `${format(v)} %` : '—';
const $ = id => document.getElementById(id);
const text = (id, value) => { $(id).textContent = value; };
const colored = (id, value, cls) => { text(id,value); $(id).className = cls; };
const details = (id, rows) => { $(id).innerHTML = rows.map(([k,v,c='']) => `<dt>${escapeHTML(k)}</dt><dd class="${c}">${escapeHTML(v)}</dd>`).join(''); };
const empty = (id, heading, message) => { $(id).innerHTML = `<div class="empty"><strong>${escapeHTML(heading)}</strong><p>${escapeHTML(message)}</p></div>`; };
const colors = {ema10:'#4b8eff',ema20:'#62c9e9',sma50:'#efaa39',sma200:'#f65b63'};

function chartFrame(width, height, low, high, labels, draw, reference) {
  const left=40, right=width-40, top=12, bottom=height-26;
  const x=i => left + i * (right-left) / Math.max(labels.length-1,1);
  const y=v => bottom - (v-low)/(high-low || 1)*(bottom-top);
  let svg='';
  for(let i=0;i<4;i++){ const value=low+(high-low)*i/3, yy=y(value); svg+=`<line class="grid" x1="${left}" y1="${yy}" x2="${right}" y2="${yy}"/><text x="${left-7}" y="${yy+4}" text-anchor="end">${format(value,0)}</text>`; }
  if(finite(reference) && reference>=low && reference<=high) svg+=`<line class="reference" x1="${left}" y1="${y(reference)}" x2="${right}" y2="${y(reference)}"/>`;
  const steps=Math.min(7,labels.length);
  for(let i=0;i<steps;i++){const idx=Math.round(i*(labels.length-1)/Math.max(steps-1,1));svg+=`<text x="${x(idx)}" y="${height-5}" text-anchor="middle">${escapeHTML(labels[idx])}</text>`;}
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img">${svg}${draw({x,y,left,right,top,bottom})}</svg>`;
}
function miniChart(id, values, dates, bars=false, percent=false) {
  const series=values ?? [];
  if(!series.length || series.some(v=>!finite(v)) || dates?.length!==series.length){empty(id,'Keine Historie','Für dieses Universum liegt keine vollständige Zeitreihe vor.');return;}
  const low=percent?0:Math.min(0,...series)*1.15, high=percent?100:Math.max(1,...series)*1.15;
  const width=Math.max(280,Math.round($(id).clientWidth));
  const height=Math.max(117,Math.round($(id).clientHeight));
  $(id).innerHTML=chartFrame(width,height,low,high,dates,({x,y,bottom})=>{
    if(bars) return series.map((v,i)=>`<rect x="${x(i)-1}" y="${Math.min(y(0),y(v))}" width="${Math.max(1,(width-80)/series.length-1)}" height="${Math.max(1,Math.abs(y(v)-y(0)))}" fill="${v>=0?'#3dce81':'#e65b65'}"><title>${escapeHTML(dates[i])}: ${format(v)}</title></rect>`).join('');
    const path=series.map((v,i)=>`${i?'L':'M'}${x(i)},${y(v)}`).join(' ');
    return `<path d="${path} L${x(series.length-1)},${bottom} L${x(0)},${bottom}Z" fill="#3bc98412"/><path d="${path}" fill="none" stroke="#46d88c" stroke-width="2"/>`;
  },percent?50:0);
  $(id).querySelector('svg').setAttribute('aria-label',`${bars?'McClellan Oscillator':'Nasdaq-100-Breadth über SMA50'}, ${series.length} Sitzungen, letzter Wert ${format(series.at(-1))}${percent?' Prozent':''}`);
}
function priceChart(data, range) {
  const bars=selectBars(data.qqq_chart?.bars ?? [],range,data.meta.as_of_date);
  document.querySelectorAll('[data-range]').forEach(b=>{b.classList.toggle('selected',b.dataset.range===range);b.setAttribute('aria-pressed',String(b.dataset.range===range));});
  if(!bars.length){empty('priceChart','QQQ-Kurshistorie noch nicht verfügbar','Für diesen Datenstand wurde noch keine QQQ-Kurshistorie bereitgestellt.');text('priceCaption','Kursangabe oben aus dem bestehenden Markt-Snapshot');return;}
  const latest=bars.at(-1);
  $('legend').innerHTML=Object.entries(colors).map(([key,color])=>`<span style="--line:${color}">${key.toUpperCase()} ${format(latest[key],2)}</span>`).join('');
  const prices=bars.flatMap(b=>[b.h,b.l,...Object.keys(colors).map(k=>b[k])]).filter(finite);
  const low=Math.min(...prices)*.98,high=Math.max(...prices)*1.02;
  const width=Math.max(280,Math.round($('priceChart').clientWidth));
  const height=Math.max(245,Math.round($('priceChart').clientHeight));
  $('priceChart').innerHTML=chartFrame(width,height,low,high,bars.map(b=>b.date.slice(5).split('-').reverse().join('.')),({x,y,bottom})=>{
    let marks=''; const w=Math.max(.7,Math.min(6,(width-80)/bars.length*.65));const maxV=Math.max(1,...bars.map(b=>b.v));
    bars.forEach((b,i)=>{const c=b.c>=b.o?'#42ce85':'#ee6068';marks+=`<g><title>${escapeHTML(b.date)} · O ${format(b.o,2)} · H ${format(b.h,2)} · L ${format(b.l,2)} · C ${format(b.c,2)} · Vol ${format(b.v,0)}</title><line x1="${x(i)}" y1="${y(b.h)}" x2="${x(i)}" y2="${y(b.l)}" stroke="${c}"/><rect x="${x(i)-w/2}" y="${Math.min(y(b.o),y(b.c))}" width="${w}" height="${Math.max(1,Math.abs(y(b.o)-y(b.c)))}" fill="${c}"/><rect x="${x(i)-w/2}" y="${bottom-b.v/maxV*30}" width="${w}" height="${b.v/maxV*30}" fill="${c}" opacity=".35"/></g>`;});
    for(const [key,color] of Object.entries(colors)){let pen=false;const path=bars.map((b,i)=>{if(!finite(b[key])){pen=false;return '';}const command=pen?'L':'M';pen=true;return `${command}${x(i)},${y(b[key])}`;}).join(' ');marks+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="1.6"/>`;}
    return marks;
  });
  $('priceChart').querySelector('svg').setAttribute('aria-label',`QQQ Tageskerzen mit Volumen und vier gleitenden Durchschnitten, ${bars[0].date} bis ${latest.date}`);
  text('priceCaption',`${bars[0].date} – ${latest.date} · ${bars.length} Sitzungen · OHLCV, splitbereinigt · Werte beim Überfahren`);
}
function rows(id, items, render, columns=3) {
  $(id).innerHTML=items.length?items.map(render).join(''):`<tr><td colspan="${columns}" class="muted">Keine passenden Daten vorhanden.</td></tr>`;
}
function render(data) {
  const {regime:r,health:h,nasdaq:n,breadth:b}=data;
  const date=new Date(`${data.meta.as_of_date}T12:00:00Z`).toLocaleDateString('de-DE',{day:'numeric',month:'short',year:'numeric'});
  text('dataStatus',`Daten: ${date} · ${data.meta.eod_pinned?'EOD-Version':'bestehender Datenstand'}`);
  text('notice', data.meta.eod_pinned ? 'QQQ-Kurshistorie verfügbar. Die übrigen Marktindikatoren sind noch nicht vollständig auf denselben Handelstag abgestimmt.' : 'QQQ-Kurshistorie noch nicht verfügbar. Die Marktindikatoren sind noch nicht vollständig auf denselben Handelstag abgestimmt.');
  $('notice').hidden=data.meta.session_synchronized === true;
  colored('regimeState',r.state ?? '—',`regime-state ${tone(r.score)}`);
  details('regimeDetails',[['Score',`${format(r.score)} / 100`,tone(r.score)],['Marktbreite',format(r.components?.breadth?.score),tone(r.components?.breadth?.score)],['Momentum',format(r.components?.momentum?.score),tone(r.components?.momentum?.score)]]);
  colored('healthScore',format(h.score),tone(h.score));colored('healthState',h.base_state ?? '—',`badge ${tone(h.score)}`);
  details('healthDetails',[['Trend',format(h.components?.trend),tone(h.components?.trend)],['Breadth',format(h.components?.breadth),tone(h.components?.breadth)],['Veränderung',signed(h.score_delta),h.score_delta>=0?'positive':'negative']]);
  colored('breadthScore',pct(b.above_sma50),tone(b.above_sma50));
  details('breadthDetails',[['NDX Hochs',format(n.new_highs,0),'positive'],['NDX Tiefs',format(n.new_lows,0),'negative'],['H/L-Verhältnis',n.new_lows>0?format(n.new_highs/n.new_lows):'—']]);
  colored('mcoScore',signed(n.mco),finite(n.mco)?(n.mco>=0?'positive':'negative'):'muted');
  details('mcoDetails',[['Summation Index',signed(n.summation),n.summation>=0?'positive':'negative'],['Universum',`${format(n.n_components,0)} NDX-Titel`]]);
  text('qqqQuote',`QQQ ${format(data.benchmarks?.QQQ?.c,2)}`);
  const structure=[...['SPY','QQQ','IWM','RSP'].map(s=>[s,format(data.benchmarks?.[s]?.c,2),'']),['Über SMA 50 · YOLO',pct(b.above_sma50),tone(b.above_sma50)],['Über SMA 200 · YOLO',pct(b.above_sma200),tone(b.above_sma200)],['Eligible Universe',`${format(b.universe_size,0)} Titel`,'']];
  $('structure').innerHTML=structure.map(([k,v,c])=>`<div class="structure-row"><span>${escapeHTML(k)}</span><b class="${c}">${escapeHTML(v)}</b></div>`).join('');
  rows('narratives',data.narratives,v=>`<tr><td>${escapeHTML(v.name)}</td><td><div class="bar-cell"><div class="bar-track"><i style="width:${Math.max(0,Math.min(100,v.score))}%"></i></div><span>${format(v.score,0)}</span></div></td></tr>`,2);
  rows('leadership',data.leadership,v=>`<tr><td><a href="index.html#screener">${escapeHTML(v.symbol)}</a></td><td>${escapeHTML(v.name)}</td><td><span class="rank">${format(v.score,0)}</span></td></tr>`);
  rows('opportunities',data.opportunities,v=>`<tr><td><a href="index.html#opportunities">${escapeHTML(v.symbol)}</a></td><td>${escapeHTML(v.setup)}</td><td><span class="rank">${format(v.score,0)}</span></td></tr>`);
  let activeRange='1Y', resizeTimer;
  const drawCharts=()=>{
    miniChart('breadthChart',n.kma_history?.sma50,n.kma_dates,false,true);
    miniChart('mcoChart',n.mco_history,n.mco_zscore_dates,true);
    priceChart(data,activeRange);
  };
  drawCharts();
  document.querySelectorAll('[data-range]').forEach(button=>{button.disabled=!data.qqq_chart?.bars?.length;button.addEventListener('click',()=>{activeRange=button.dataset.range;priceChart(data,activeRange);});});
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawCharts,100);});
  text('source',`${data.meta.source} · Markt-Snapshot aktualisiert: ${data.meta.snapshot_updated_at ?? 'unbekannt'}`);
}
if(typeof document!=='undefined') {
  $('help').addEventListener('click',()=>$('methodDialog').showModal());
  fetch('data/market_overview.json',{cache:'no-cache'}).then(r=>{if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json();}).then(render).catch(()=>{
    text('dataStatus','Daten nicht verfügbar');text('notice','Market Overview konnte nicht geladen werden. Bitte den lokalen Datenstand neu bauen und die Seite erneut laden.');$('notice').hidden=false;
    for(const id of ['priceChart','breadthChart','mcoChart'])empty(id,'Daten nicht verfügbar','Der Datenabruf ist fehlgeschlagen.');
    text('healthState','Nicht verfügbar');
  });
}
