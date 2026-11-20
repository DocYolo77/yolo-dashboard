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
python scripts/build_market_overview.py --data /private/review-inputs --eod-store /private/yolo-eod --history-dir data/history/daily
```

Der Adapter liest `latest.json` einmal und lädt `market.json` sowie
`charts/QQQ.json` aus derselben Version. Manifest, beide Payloads sowie
Dashboard-Scores, Market Features und Narratives müssen denselben Sitzungstag
haben. Unterschiedliche Tage brechen den Export ab. Preise und Durchschnitte
der Benchmarks stammen dann aus Phase 1. Kerzen, Volumen, EMA10/20 und
SMA50/200 sind als SVG mit 1M/3M/6M/YTD/1Y-Auswahl und OHLCV-Tooltips
implementiert. Es werden keine Zugangsdaten benötigt oder übernommen.

QQQ Health verwendet im EOD-Modus die gemeinsame kanonische Funktion
`build_qqq_health` aus `build_dashboard_states.py`. Der bisherige State Builder
verwendet dieselbe Funktion. Gewichte, Formeln und Schwellenwerte sind identisch.
Kurs und vier Durchschnitte kommen direkt aus der letzten gepinnten QQQ-Kerze;
ATR14 ist der einfache Mittelwert der letzten 14 True Ranges. ATR%, Rundung und
ATR-Extension entsprechen der bisherigen `calc_atr_extension`-Definition.
`--history-dir` bindet die vorherigen Tagesstates für Delta und Hysterese an.
Es erfolgt kein Rückgriff auf den alten Yahoo-QQQ-Schlusskurs.

Der Review-Export enthält jetzt echte QQQ-Kerzen für den 19.11.2026. Ein
Neuaufbau ohne `--eod-store` bleibt als Legacy-Vorschau möglich, markiert aber
fehlende QQQ-Kerzen und deaktiviert die Zeitraumwahl. Der Review-CI-Check
verlangt dagegen einen vollständigen, konsistenten EOD-Export.

## Bedeutung und Grenzen

- Market Regime übernimmt den vorhandenen Backend-Score. QQQ Health verwendet
  die vorhandenen Formeln mit der Kursbasis des gepinnten Massive-Charts.
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
- Der Legacy-Snapshot hat nur einen Erstellungszeitpunkt. Im EOD-Modus prüft
  der Adapter deshalb zusätzlich die tatsächlichen letzten Datumslabels von
  Nasdaq-Breadth, McClellan, Summation und Hoch-/Tief-Historie sowie die
  Übereinstimmung ihrer skalaren Endwerte mit den Historien. Die Legacy-
  Tages-/Monatslabels werden gegen den vollständigen Score-Handelstag geprüft.
  Veraltete Kerzen, abweichende Benchmark-Kurse oder verschiedene Datumsstände
  brechen den Export ab, statt nur das Anzeigedatum umzuschreiben.

## Verifizierter Review-Stand

- Eingangsdaten: öffentlicher Repository-Commit
  `d99d216500bab9f952e96a1990d4e8ebbe59dc19`, privat heruntergeladen; kein Merge.
- Phase-1-Version: `2026-11-19-35e8374a51244eadb44d78f7d8f4920a`;
  460 Sitzungen im privaten Store, 252 QQQ-Kerzen im kompakten Review-Export.
- Manifest, Chart-Ende, Benchmark, Feature-/Narrative-Daten und QQQ Health:
  `2026-11-19`. Chart- und Health-Schlusskurs: `645.04 USD`.
- Der alte auf 19.11. datierte Health-State verwendete `659.51 USD`, den
  tatsächlichen Schlusskurs vom 18.11. Die Kursbasis ist korrigiert.
- Der numerische Health-Score bleibt `10.0 / 100` (`RISK OFF`), weil die
  korrigierten Kursbedingungen dieselben Trend-Punkte ergeben. Unveränderte
  Score-Zahl bedeutet hier nicht unveränderte Kursbasis.
- Der volle Store und SQLite-Dateien bleiben außerhalb von Git. Versioniert
  wird nur der kompakte Frontend-Export, ohne Schlüssel oder Laufzeitdatenbank.
- Dies ist ein einzelner echter Snapshot-Lauf. Die Abnahme über fünf echte
  Handelssitzungen steht weiterhin aus; kein Deployment.

## Checks

```sh
python -m unittest discover -s tests -p test_market_overview.py -v
node --test tests/market_overview.test.mjs
node --test tests/dashboard_filters.test.mjs
```

Die Branch-Action `Validate Dashboard 2.0 Market Overview` führt außerdem
die bestehenden V1-/V1.1-Score-Tests und die Konsistenzprüfung des tatsächlich
versionierten Frontend-Exports aus. Sie ruft Massive nicht erneut ab.
Die SVG-Charts richten ihre Zeichenfläche an der tatsächlichen Kartenbreite
aus; Achsenbeschriftungen bleiben auch auf Mobilgeräten lesbar. Ein
Zeitraumwechsel und Größenänderungen zeichnen die gleichen gepinnten Daten neu.
