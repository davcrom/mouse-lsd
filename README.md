# psychedelics

## Installation

This project uses functions from the `psyfun` package, which should be installed to allow relative imports in notebooks. From the project root directory, run:
```bash
pip install -e .
```

## Structure

- `psyfun/` — importable package with shared analysis functions
- `scripts/` — top-level analysis scripts (`fetch_data.py`, `dataset_overview.py`, `population_dimensionality.py`, `single_unit.py`)
- `notebooks/` — exploratory and analysis notebooks
- `data/`, `metadata/` — datasets and session metadata
- `figures/` — generated figures
- `histology/`, `openfield/`, `video/`, `stats/` — task- and modality-specific analyses
- `docs/`, `archive/` — documentation and older work
