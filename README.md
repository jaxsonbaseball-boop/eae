# Project Apex (v3.5) — High School TrackMan B1 to NCAA Division 1 `Stuff+` & 3D Pitch Lab

**Project Apex** is an end-to-end physical pitch modeling, Bayesian small-sample stabilization, repertoire synergy, and 3D WebGL visualization suite designed to evaluate high school portable **TrackMan B1** bullpen telemetry against an **NCAA Division 1** competitive baseline (`100 = NCAA D1 Average`, `SD = 10`).

## Repository Structure

```text
project-apex-stuff-plus/
├── index.html                              # Self-contained v3.5 3D WebGL & What-If Pitch Lab (GitHub Pages ready)
├── web/
│   └── apex_stuff_pitch_lab.html           # Copy of v3.5 Web App for embedding in Google Sites / iPad
├── src/
│   ├── physics/
│   │   ├── constants.py                    # Immutable thermodynamic, baseball (NFHS vs NCAA), and field geometry constants
│   │   └── trajectory.py                   # Phase 1: 3D RK45 solver, Arden Buck air density, Nathan/Adair drag/lift & SSW
│   ├── ingestion/
│   │   └── trackman_parser.py              # Phase 1: Tripod laser-skew rotation, hardware QA gates & 3-sigma outlier filters
│   ├── models/
│   │   ├── public_stuff_architectures.py   # Phase 0: Reverse-engineered registry of the top 5 public Stuff+ models
│   │   └── college_stuff_engine.py         # Phase 2: 4-head multi-task GBDT cascade & 10x10 location-grid neutralization
│   ├── stats/
│   │   ├── bayesian_scaling.py             # Phase 3: Non-linear sub-88 mph velo cliff, Empirical Bayes shrinkage & 3-yr projection
│   │   └── arsenal_synergy.py              # Phase 4: 23.8-ft commitment tunneling, 50% FB weight & gamma_anchor gating
│   └── reporting/
│       └── scout_narrative.py              # Phase 6: Automated ScoutSpeak synthesis (Scout vs R&D modes) & Pitch Lab simulator
├── docs/
│   └── reverse_engineered_stuff_models.md  # Technical analysis of FanGraphs, PitchingBot, StuffPro, tjStuff+ & Driveline
├── data/
│   └── sample_bullpens/
│       └── marcus_vance_trackman_b1.csv    # Sample portable TrackMan B1 showcase bullpen CSV
├── tests/
│   └── test_trajectory.py                  # Automated unit test suite across Phases 1-6
├── pyproject.toml
├── requirements.txt
└── push_to_github.sh                       # One-command push script for GitHub
```

## Quick Start

### 1. Run the Python Modeling & Verification Suite
```bash
python3 -m unittest discover -s tests -v
```

### 2. Launch the Interactive 3D WebGL Pitch Lab (`index.html`)
Open `index.html` (or `web/apex_stuff_pitch_lab.html`) directly in any modern browser, or enable **GitHub Pages** on the `main` branch (`Settings -> Pages -> Deploy from branch: main / root`) to host the live interactive 3D Pitch Lab online.

### 3. Push to GitHub (via GitHub Desktop or CLI)
- **GitHub Desktop**: Open **GitHub Desktop**, select **File -> Add Local Repository...**, choose this `project-apex-stuff-plus` folder, and click **Publish repository**.
- **Terminal CLI**:
  ```bash
  ./push_to_github.sh https://github.com/jaxsonbaseball-boop/project-apex-stuff-plus.git
  ```
