# BBB Permeability Streamlit GUI

Prototype Streamlit interface for the sparse-label multi-task workflow described in Tabs 4 and 5 of the BBB manuscript. This release establishes the home and documentation pages that will anchor the forthcoming ligand submission workspace.

Link: https://bbb-prediction-jxzbtlzkcchsocobqlnafy.streamlit.app/Documentation

## Features
- Summarises the Tab 4 manuscript narrative: sparse-label multi-task training, blended calibration, and external validation results.
- Highlights Tab 5 evaluation practices, including bootstrap uncertainty, calibration checks, and planned visual assets.
- Provides a documentation page with repository structure, setup instructions, and roadmap.
- Ready for GitHub deployment and Streamlit sharing (`streamlit run streamlit_app.py`).

## Getting started
```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
streamlit run streamlit_app.py
```
The app defaults to `http://localhost:8501`. Use the sidebar to navigate between the home dashboard and documentation. A ligand intake tab will be added in a future iteration.

## Project layout
```
.
├── BBB Manuscript.docx      # Source manuscript (references for Tabs 4–5)
├── README.md
├── requirements.txt
├── streamlit_app.py         # Home dashboard
└── pages/
    └── 1_Documentation.py   # Documentation & runbook
```

## Roadmap
- **In progress:** Design ligand submission tab (SMILES/PDBQT intake, descriptor generation, scoring).
- **Planned:** Calibration overlays for user batches, applicability-domain visualisations, PDF/CSV export pipeline.
- **Contribution guide:** Fork the repo, branch from `main`, follow conventional commits, and open a pull request with screenshots for UI updates.

## Contact
Questions, bug reports, or collaboration requests: **Dr. Sivanesan Dakshanamurthy** — sd233@georgetown.edu
---



