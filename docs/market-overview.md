# Phase 2 · Market Overview

Branch: `phase-2/market-overview`. Lokaler Einstieg: `market-overview.html`;
die bestehende Oberfläche verlinkt die neue Ansicht in ihrer Navigation.
Layout nach dem bereitgestellten Mockup: vier Kennzahlenkarten, QQQ-Chart,
zwei Nasdaq-Charts und vier Tabellen. Kein Deployment und kein Merge.

## Lokale Vorschau

```sh
python scripts/build_market_overview.py
python -m http.server 8766 --bind 127.0.0.1
```

Dann `http://127.0.0.1:8766/market-overview.html` öffnen. Die Ansicht lädt
einen kompakten Export statt der großen Rohdateien. Nach neuen Eingangsdaten
den Export erneut bauen; kein bestehender Refresh-Workflow wird geändert.

## Phase-1-Anbindung

```sh
python scripts/build_market_overview.py --eod-store /private/yolo-eod
```

Der Adapter liest `latest.json` einmal und lädt `market.json` sowie
`charts/QQQ.json` aus derselben Version. Manifest, beide Payloads sowie
Dashboard-Scores, Market Features und Narratives müssen denselben Sitzungstag
haben. Unterschiedliche Tage brechen den Export ab. Preise und Durchschnitte
der Benchmarks stammen dann aus Phase 1. Kerzen, Volumen, EMA10/20 und
SMA50/200 sind als SVG mit 1M/3M/6M/YTD/1Y-Auswahl und OHLCV-Tooltips
implementiert. Es werden keine Zugangsdaten benötigt oder übernommen.

Im aktuellen Checkout fehlen QQQ-Kerzen. Der Standardexport markiert daher
den Chart als nicht verfügbar und deaktiviert die Zeitraumwahl. Er füllt
keine Kurse auf. IWM/RSP sind ohne Phase-1-Export ebenfalls nicht verfügbar.

## Bedeutung und Grenzen

- Market Regime und QQQ Health übernehmen die vorhandenen Backend-Scores.
- Market Breadth zeigt den Anteil über SMA50 im eligible YOLO-Universum;
  fehlende Durchschnitte werden aus dem jeweiligen Nenner ausgeschlossen.
- Nasdaq-Breadth-Chart, neue Hochs/Tiefs, McClellan und Summation Index
  stammen aus dem bestehenden Nasdaq-100-Snapshot. Sie sind ausdrücklich
  nicht als gesamter US-Aktienmarkt beschriftet.
- Market Structure zeigt ETF-Preise (SPY/QQQ/IWM/RSP), keine Indexstände.
- Narrative Strength zeigt den bestehenden Structural Score. Leadership
  sortiert das vorhandene RS-3M-Perzentil. Fehlende Unternehmensnamen bleiben
  Tickersymbole.
- Opportunities übernimmt vorhandene Pullback-/Reset-/Fresh-Leader-Flags,
  lässt Extended-/Laggard-Titel aus und sortiert nach Structural RS. Die
  Spaltenüberschrift benennt diesen Wert; es wird kein Setup-Score erfunden.
- Der Legacy-Snapshot hat nur einen Erstellungszeitpunkt. Deshalb behauptet
  die Ansicht auch bei vorhandenem Phase-1-Export keine Synchronität aller
  Legacy-Indikatoren. Ein sichtbarer Hinweis benennt diese Einschränkung.

## Checks

```sh
python -m unittest discover -s tests -p test_market_overview.py -v
node --test tests/market_overview.test.mjs
node --test tests/dashboard_filters.test.mjs
```
