# Dashboard 2.0 · EOD-Datenstand

Der Builder bereitet einen Massive-Datenstand für einen abgeschlossenen US-Handelstag auf. Der bisherige Refresh-Workflow und die bisherige Oberfläche behalten ihre Einstiegspunkte.

## Lokaler Lauf

Voraussetzungen sind Python 3.11, ein Massive-Stocks-Key in der Umgebungsvariable `MASSIVE_API_KEY` und ein privates, dauerhaftes Verzeichnis außerhalb des Checkouts.

```sh
python -m pip install -r requirements-eod.txt
python -m unittest discover -s tests -p test_eod_snapshot.py -v
python scripts/build_eod_snapshot.py --probe
python scripts/build_eod_snapshot.py --store /private/yolo-eod
```

Der Standard-Slot ist 20:30 Uhr New York. `--slot next-morning --request-delay 12.1` nutzt 00:30 Uhr New York am Kalendertag nach der Sitzung und die Basic-Rate. Der erste Lauf braucht für 460 Handelssitzungen mindestens 460 Grouped-Daily-Anfragen zuzüglich Referenz und Splits. Ein warmer Lauf lädt fünf Sitzungen einschließlich der neuen Sitzung sowie die Referenzdaten und Split-Ereignisse. Ein Test mit `--session YYYY-MM-DD` gehört in ein eigenes Replay-Verzeichnis.

## Datenvertrag

- `market_history.sqlite3` ist die dauerhafte, unbereinigte OHLCV-/VWAP-Historie. Der SQLite-Lock im gleichen Verzeichnis serialisiert Writer. Für einen neuen Lauf wird keine Datei aus Git benötigt.
- `latest.json` verweist auf eine einzige Version. Leser laden dieses Manifest einmal und pinnen für die ganze Ansicht dieselbe Version.
- `market.json` enthält die aktuellen Balken und Durchschnitte für SPY, QQQ, IWM und RSP. Der Adapter für die bestehenden Regime- und QQQ-Health-Scores kommt mit Market Overview.
- `screener.json` enthält aktive Stammaktien und die festgelegten ADR-Typen mit gültigem Tagesbar. Features sind Kurs, Volumen, ADR20, ADTV20, RVOL50, Returns und gleitende Durchschnitte. Klassifikation, Market Cap und YOLO-Ranking werden in späteren Stufen ergänzt.
- `charts/<symbol>.json` enthält bis zu 252 tatsächliche Kerzen mit Volumen und EMA10/20, SMA50/200. Für junge Titel bleiben nicht berechenbare Durchschnitte leer.
- Exporte korrigieren Splits bis einschließlich Zieltag auf einer einheitlichen Basis. Dividenden bleiben unberücksichtigt. Dollarvolumen berechnet sich aus VWAP × Volumen. Fehlende Sitzungen werden nicht aufgefüllt.
- Das Manifest wird erst nach einem erfolgreichen Write des Datenstores ersetzt. Fehlgeschlagene oder unvollständige Abrufe lassen die vorherige Version sichtbar. Ein identischer Wiederholungslauf erhält die bestehende Version.

## Abnahme

Der Branch-Workflow prüft Tests und Datenzugang mit dem vorhandenen Repository-Secret. Der Probe-Modus schreibt keine Marktdaten und loggt keine Kurse, Payloads oder Schlüssel. Eine Antwort für den laufenden US-Tag belegt noch keine EOD-Vollständigkeit.

Für die Live-Abnahme sind der erfolgreiche Zugriff mit dem Actions-Key, die Messung zum geplanten Slot und fünf aufeinanderfolgende echte Handelssitzungen notwendig. Für diese Serie muss ein freigegebener Runner auf dasselbe private, dauerhafte Verzeichnis zugreifen. Die lokalen Replays prüfen Mechanik und Fehlerfälle mit synthetischen Daten. Die öffentliche Pages-Instanz ist kein Ziel dieses Datenlaufs.
