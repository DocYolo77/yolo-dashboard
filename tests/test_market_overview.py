import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from scripts.build_market_overview import build


class OverviewTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.day='2026-11-18'
        self.payloads={
            'dashboard_state':{'meta':{'as_of_date':self.day},'market_regime':{'score':25.6},'qqq_health':{'score':10.1},'opportunities':{'items':[]}},
            'market_features':{'meta':{'date_range':['2026-01-02',self.day]},'tickers':{
                'A':{'eligible':True,'sma50_distance_pct':2,'sma200_distance_pct':None},
                'B':{'eligible':True,'sma50_distance_pct':-2,'sma200_distance_pct':0},
                'C':{'eligible':False,'sma50_distance_pct':3}}},
            'narratives':{'meta':{'date_range':['2026-01-02',self.day]},'narratives':[]},
            'snapshot':{'meta':{'updated_at':'2026-11-19T01:00:00Z'},'qqq_breadth':{'new_highs':0,'new_lows':0},'regime':{'QQQ':{'price':100}}}}
        self.write()

    def write(self):
        for name,value in self.payloads.items():
            (self.root/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')

    def test_preserves_scores_and_excludes_ineligible_and_missing_values(self):
        result=build(self.root)
        self.assertEqual(result['regime'],self.payloads['dashboard_state']['market_regime'])
        self.assertEqual(result['health']['score'],10.1)
        self.assertEqual(result['breadth']['above_sma50'],50)
        self.assertEqual(result['breadth']['above_sma200'],0)
        self.assertFalse(result['meta']['eod_pinned'])
        self.assertEqual(result['qqq_chart']['bars'],[])

    def test_refuses_different_session_dates(self):
        self.payloads['narratives']['meta']['date_range'][-1]='2026-11-17'
        self.write()
        with self.assertRaisesRegex(ValueError,'matching session'):
            build(self.root)

    def make_store(self):
        store=self.root/'store'
        store.mkdir()
        manifest={'as_of_date':self.day,'version':'test-revision','market':'market.json','charts':'charts/'}
        (store/'latest.json').write_text(json.dumps(manifest))
        last={'date':self.day,'o':120,'h':124,'l':119,'c':123,'v':100,'ema10':121,'ema20':120,'sma50':119,'sma200':110}
        (store/'market.json').write_text(json.dumps({'as_of_date':self.day,'benchmarks':{'QQQ':last}}))
        (store/'charts').mkdir()
        chart={'as_of_date':self.day,'symbol':'QQQ','bars':[
            {**last,'date':(date.fromisoformat(self.day)-timedelta(days=14-i)).isoformat()} for i in range(15)]}
        (store/'charts/QQQ.json').write_text(json.dumps(chart))
        self.set_breadth()
        return store,chart

    def test_pins_eod_revision_and_refuses_mixed_payload_dates(self):
        store,chart=self.make_store()
        result=build(self.root,store)
        self.assertEqual(result['qqq_chart'],chart)
        self.assertEqual(result['health']['price_structure']['close'],123)
        self.assertEqual(result['health']['components']['trend'],100)
        self.assertEqual(result['health']['price_structure']['atr14'],5)
        self.assertEqual(result['health']['as_of_date'],self.day)
        self.assertNotEqual(result['health']['score'],10.1)
        chart['as_of_date']='2026-11-17'
        (store/'charts/QQQ.json').write_text(json.dumps(chart))
        with self.assertRaisesRegex(ValueError,'payload date'):
            build(self.root,store)

    def test_refuses_stale_actual_candle_even_if_metadata_claims_today(self):
        store,chart=self.make_store()
        chart['bars'][-1]['date']='2026-11-17'
        (store/'charts/QQQ.json').write_text(json.dumps(chart))
        with self.assertRaisesRegex(ValueError,'last candle/session'):
            build(self.root,store)

    def test_refuses_stale_breadth_and_inconsistent_scalar(self):
        store,_=self.make_store()
        b=self.payloads['snapshot']['qqq_breadth']
        b['hl_dates'][-1]='17.11.'
        self.write()
        with self.assertRaisesRegex(ValueError,'score session'):
            build(self.root,store)
        b['hl_dates'][-1]='18.11.'
        b['hl_osc']=5
        self.write()
        with self.assertRaisesRegex(ValueError,'history endpoint'):
            build(self.root,store)

    def test_refuses_conflicting_benchmark_and_chart_prices(self):
        store,_=self.make_store()
        market=json.loads((store/'market.json').read_text())
        market['benchmarks']['QQQ']['c']=100
        (store/'market.json').write_text(json.dumps(market))
        with self.assertRaisesRegex(ValueError,'benchmark/chart disagree'):
            build(self.root,store)

    def test_previous_state_is_used_for_score_delta(self):
        store,_=self.make_store()
        history=self.root/'history/daily'
        history.mkdir(parents=True)
        (history/'2026-11-17.json').write_text(json.dumps({'qqq_health':{'score':40,'extended':False}}))
        (history/'2026-11-19.json').write_text(json.dumps({'qqq_health':{'score':99,'extended':True}}))
        health=build(self.root,store)['health']
        self.assertEqual(health['prev_score'],40)
        self.assertEqual(health['score_delta'],round(health['score']-40,1))

    def set_breadth(self):
        b=self.payloads['snapshot']['qqq_breadth']
        labels=['18.11.']*20
        for key in ('mco_zscore_dates','summation_zscore_dates','hl_dates','kma_dates'):
            b[key]=labels.copy()
        for history,scalar in [('mco_history','mco'),('mco_zscore_history','mco_zscore'),('summation_zscore_history','summation_zscore'),('summation_zscore_sma10_history','summation_zscore_sma10'),('hl_history','hl_osc')]:
            b[history]=[0]*20;b[scalar]=0
        b['kma_history']={k:[50]*20 for k in ('ema10','ema21','sma50','sma200')}
        b['kma_now']={k:50 for k in b['kma_history']}
        b['n_components']=100
        self.write()

    def test_missing_opportunity_score_does_not_become_zero(self):
        self.payloads['dashboard_state']['opportunities']['items']=[{'symbol':'A','ema20_pullback':True,'structural_rs':None}]
        self.write()
        self.assertIsNone(build(self.root)['opportunities'][0]['score'])


if __name__=='__main__':
    unittest.main()
