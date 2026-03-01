# YOLO Dashboard — Setup-Anleitung

## Kosten: €0

Alles kostenlos. GitHub Pages, GitHub Actions, Yahoo Finance — alles Free Tier.

Die KI-Zusammenfassungen kannst du manuell eintragen (30 Sekunden pro Briefing).
Wenn du irgendwann automatisieren willst, fügst du einfach einen Anthropic API Key hinzu (~€1-2/Monat).

---

## Setup in 6 Schritten

### Schritt 1: GitHub Account erstellen

Falls du noch keinen hast:
1. Gehe zu **https://github.com/signup**
2. Kostenloser Account reicht
3. E-Mail verifizieren

### Schritt 2: Repository erstellen

1. Gehe zu **https://github.com/new**
2. Name: `yolo-dashboard`
3. Wähle **Public** (nötig für kostenloses GitHub Pages)
4. Haken bei **"Add a README file"** NICHT setzen
5. Klicke **"Create repository"**

### Schritt 3: Dateien hochladen

1. Entpacke die ZIP-Datei auf deinem Computer
2. Auf der GitHub-Seite deines neuen Repos klicke auf **"uploading an existing file"**
3. Ziehe den GESAMTEN INHALT des entpackten Ordners per Drag & Drop rein:
   - `index.html`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - `scripts/` Ordner (mit `build_data.py`)
   - `data/` Ordner (mit `briefings.json` und `.gitkeep`)
   - `.github/` Ordner (mit `workflows/refresh_data.yml`)
4. **Wichtig:** Der `.github` Ordner ist versteckt! Auf Mac: `Cmd+Shift+.` im Finder um versteckte Dateien zu zeigen. Auf Windows: Im Explorer unter "Ansicht" → "Ausgeblendete Elemente" aktivieren.
5. Klicke **"Commit changes"**

### Schritt 4: GitHub Pages aktivieren

1. Im Repo: **Settings** (Tab oben)
2. Links im Menü: **Pages**
3. Under "Source": Wähle **"Deploy from a branch"**
4. Branch: **main** / Folder: **/ (root)**
5. Klicke **"Save"**

### Schritt 5: Ersten Datenlauf starten

1. Im Repo: **Actions** (Tab oben)
2. Falls gefragt: "I understand my workflows, go ahead and enable them" klicken
3. Links: Klicke auf **"Refresh Dashboard Data"**
4. Rechts: **"Run workflow"** → Branch: main → Briefing: morning
5. Klicke **"Run workflow"**
6. Warte 2-3 Minuten bis ein grüner Haken ✅ erscheint
7. Fertig! Dein Dashboard ist jetzt live unter:

```
https://DEIN-USERNAME.github.io/yolo-dashboard/
```

### Schritt 6: Briefings manuell eintragen

So trägst du deine täglichen Zusammenfassungen ein:

1. Im Repo: Klicke auf den Ordner **`data`**
2. Klicke auf **`briefings.json`**
3. Klicke auf das **Stift-Symbol** (✏️) oben rechts → "Edit this file"
4. Ändere den Text:

```json
{
  "morning": {
    "text": "<strong>Guten Morgen.</strong> S&P Futures +0.3%, DAX stark vorbörslich. Heute NVDA Earnings nach Börsenschluss — Fokus auf Data-Center-Revenue. VIX bei 14, Regime bleibt BULL.",
    "timestamp": "01. Mär 2026 · 07:00 CET"
  },
  "evening": {
    "text": "<strong>Starker Tag.</strong> S&P +0.8%, QQQ +1.2%. NVDA Earnings besser als erwartet, +6% after hours. Sektorrotation klar in Tech. McClellan Oscillator steigt auf +58. Gold stabil bei $2.650.",
    "timestamp": "01. Mär 2026 · 22:00 CET"
  }
}
```

5. Klicke **"Commit changes"** → **"Commit changes"**
6. Dashboard aktualisiert sich automatisch

**Tipp für Hervorhebungen im Text:**
- Fett: `<strong>Dieser Text ist fett.</strong>`
- Das war's — keine weiteren HTML-Tags nötig

---

## Automatischer Zeitplan

Die **Marktdaten** (Kurse, Regime, Breadth, etc.) werden automatisch aktualisiert:
- **Mo-Fr 07:00 CET** — Morgen-Update
- **Mo-Fr 22:00 CET** — Abend-Update

Die **Briefing-Texte** musst du manuell in `data/briefings.json` eintragen.

---

## Später: KI-Briefings automatisieren (optional, ~€1-2/Monat)

Wenn du die Zusammenfassungen irgendwann automatisch von Claude AI generieren lassen willst:

1. Gehe zu **https://console.anthropic.com/**
2. Account erstellen → API Key generieren
3. $5 Guthaben aufladen (reicht 3-6 Monate)
4. Im GitHub Repo: **Settings → Secrets and variables → Actions**
5. **"New repository secret"** → Name: `ANTHROPIC_API_KEY` → Value: dein Key
6. Fertig — ab jetzt schreibt Claude die Briefings automatisch

Du kannst jederzeit zwischen manuell und automatisch wechseln.
Ohne API Key = manuell. Mit API Key = automatisch.

---

## Dateien im Überblick

```
yolo-dashboard/
├── index.html                         ← Das Dashboard (Frontend)
├── requirements.txt                   ← Python-Pakete
├── .gitignore                         ← Git-Konfiguration
├── README.md                          ← Diese Anleitung
├── .github/workflows/
│   └── refresh_data.yml               ← Automatischer Zeitplan
├── scripts/
│   └── build_data.py                  ← Holt Marktdaten + berechnet alles
└── data/
    ├── briefings.json                 ← ✏️ HIER trägst du Briefings ein
    ├── snapshot.json                  ← Wird automatisch generiert
    └── .gitkeep                       ← Platzhalter
```

---

## Troubleshooting

**Dashboard zeigt nur Platzhalter-Daten?**
→ Hast du unter Actions den ersten Workflow-Lauf gestartet? Muss grün ✅ sein.

**Actions-Lauf schlägt fehl?**
→ Klicke auf den fehlgeschlagenen Run und lies die Fehlermeldung.
→ Häufigster Grund: `.github/workflows` Ordner wurde nicht hochgeladen (versteckter Ordner!).

**Briefing-Text wird nicht angezeigt?**
→ Prüfe ob `data/briefings.json` gültiges JSON ist (keine fehlenden Kommas/Anführungszeichen).
→ Tipp: Kopiere das Beispiel oben und ändere nur den Text.

**Am Wochenende keine Updates?**
→ Korrekt — Börse geschlossen. Du kannst jederzeit manuell unter Actions → Run workflow starten.
