"""Audit the review payload itself: dates and price inputs must match the chart."""
import json
from pathlib import Path
import unittest


class ExportAuditTests(unittest.TestCase):
    def test_review_payload_has_one_qqq_session_and_price_basis(self):
        data=json.loads((Path(__file__).resolve().parents[1]/'data/market_overview.json').read_text(encoding='utf-8'))
        day=data['meta']['as_of_date']
        self.assertTrue(data['meta']['eod_pinned'])
        self.assertTrue(data['meta']['session_synchronized'])
        self.assertEqual(data['health']['as_of_date'],day)
        self.assertEqual(data['qqq_chart']['as_of_date'],day)
        self.assertEqual(data['qqq_chart']['symbol'],'QQQ')
        self.assertEqual(len(data['qqq_chart']['bars']),252)
        last=data['qqq_chart']['bars'][-1]
        self.assertEqual(last['date'],day)
        self.assertEqual(data['benchmarks']['QQQ']['date'],day)
        for field in ('c','ema10','ema20','sma50','sma200'):
            self.assertEqual(data['benchmarks']['QQQ'][field],last[field])
            self.assertEqual(data['health']['price_structure']['close' if field=='c' else field],last[field])


if __name__=='__main__':
    unittest.main()
