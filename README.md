# Reliability-Aware Lightweight Network-Flow Intrusion Detection for IoT and IIoT Edge Communication Networks

This repository is the companion research artifact for a study on reliability-aware lightweight intrusion detection in IoT and IIoT edge communication networks. It provides the executable notebook, reusable Python modules, configuration files, experiment wrappers, and representative figures required to reproduce the main evaluation workflow in a form suitable for scholarly dissemination and archival use.

## Abstract

The repository implements a deployment-oriented evaluation framework for lightweight network-flow intrusion detection on IoT and IIoT communication networks. CICIoT2023 is used as the primary benchmark and Edge-IIoTset as an independent validation dataset. The experimental pipeline compares Logistic Regression, Random Forest, LightGBM, and a compact TinyMLP model under binary intrusion detection, multiclass attack-family classification, leave-one-attack-category-out zero-day testing, calibration analysis, feature-group ablation, and model compression settings. Beyond conventional predictive performance, the workflow emphasizes operational reliability through calibration quality, false-alarm burden, model footprint, and inference throughput.

## Repository Scope

- Reproducible notebook-first workflow for the full experimental protocol.
- Modular Python implementation for preprocessing, model training, evaluation, and result export.
- Local command-line entry points for preprocessing, single-model training, and evaluation.
- Support for CICIoT2023 and Edge-IIoTset without redistributing raw datasets.
- Export surfaces for manuscript-oriented figures, tables, metrics, and model artifacts.

## Repository Layout

```text
configs/
  default.yaml                                Default paths, seeds, and model/training settings
docs/
  cas-sc-template.tex                         Journal article template material
figures/
  methodology_pipeline.pdf                    Methodology overview figure
  validation_binary_classification.png        Independent validation figure
  validation_multiclass_classification.png    Independent validation figure
  ciciot_calibration_ece_grouped_bar.png      Calibration comparison figure
  edgeiiot_calibration_ece_grouped_bar.png    Calibration comparison figure
notebooks/
  lightweight_iot_ids_ciciot_edgeiiot_colab.ipynb
results/                                      Generated-output directory
scripts/
  run_preprocessing.py                        Dataset preparation wrapper
  run_training.py                             Single-model training entry point
  run_evaluation.py                           Single-model evaluation entry point
src/
  data/                                       Dataset loading and preprocessing utilities
  evaluation/                                 Metrics and evaluation helpers
  models/                                     Model definitions
  training/                                   Training and inference logic
  utils/                                      Configuration and I/O utilities
LICENSE
README.md
requirements.txt
```

Generated artifacts are written under `results/` and are not intended to be treated as curated source files.

## Environment Setup

Python 3.10 or newer is recommended for local execution. Create an isolated environment if desired, then install the required packages:

```bash
python -m venv .venv
pip install -r requirements.txt
```

The notebook includes a setup cell for packages that may not already be available in Google Colab.

## Data Availability

CICIoT2023 and Edge-IIoTset are not redistributed in this repository. They must be obtained from their original providers and placed under the configured raw-data paths, or the paths should be updated in `configs/default.yaml` or in the notebook configuration section.

Default expected layout:

```text
data/raw/
  CIC_IOT_Dataset2023/
    CICIoT2023.csv
    CSV/MERGED_CSV/Merged*.csv
  Edge-IIoTset/
    DNN-EdgeIIoT-dataset.csv
```

Local `data/` directories are created during execution and are not part of the tracked source repository.

## Reproducibility Workflow

1. Review `configs/default.yaml` and update dataset paths if needed.
2. Place the required raw CSV files under the configured `data/raw/` layout.
3. Run the notebook from the repository root for the full experimental workflow, or use the command-line wrappers for targeted local runs.
4. Inspect generated metrics, models, tables, and figures under `results/`.

Example local commands:

```bash
python scripts/run_preprocessing.py --dataset ciciot
python scripts/run_preprocessing.py --dataset edgeiiot
python scripts/run_training.py --dataset ciciot --task binary --model lgbm
python scripts/run_evaluation.py --dataset ciciot --task binary --model lgbm
```

## Notebook Usage

Open `notebooks/lightweight_iot_ids_ciciot_edgeiiot_colab.ipynb` in Google Colab or Jupyter and execute the sections in order:

```text
Setup -> Imports -> Configuration -> Data loading -> Preprocessing -> Training -> Evaluation -> Result export
```

Google Drive mounting is optional and disabled by default. For a smoke test of the workflow, set `RUN_SYNTHETIC_DEMO = True`. Synthetic-demo outputs are intended only for pipeline verification and should not be reported as study results.

## Experimental Coverage

- Binary intrusion detection on CICIoT2023 and Edge-IIoTset
- Multiclass attack-family classification
- Leave-one-attack-category-out zero-day evaluation
- Calibration and confidence-threshold analysis
- Feature-group ablation
- TinyMLP pruning and dynamic INT8 quantization
- Model-size and throughput measurement
- Export of manuscript-oriented tables and figure source files

## Output Artifacts

Depending on the selected workflow, generated files are written to the following locations:

- `data/processed/` for prepared Parquet datasets
- `results/` for metric and evaluation CSV files
- `results/models/local/` for trained model artifacts and preprocessing bundles
- `results/figures/` for generated figures
- `results/paper_tables_final/` for manuscript-ready table exports created by the notebook

## Reproducibility Notes

- The default experimental seeds are `13`, `21`, `34`, `42`, and `55`.
- The notebook is designed to run from the repository root so that `src/` and `configs/default.yaml` resolve correctly.
- Latency, throughput, and other resource-sensitive measurements depend on the execution hardware and runtime environment.
- Generated outputs and dataset-derived artifacts should generally remain outside version-controlled source revisions.

## Citation

Formal citation metadata can be added once publication details are finalized. Until then, the repository may be cited using the following temporary BibTeX entry:

```bibtex
@misc{keskin_reliability_aware_iot_iiot_ids,
  title        = {Reliability-Aware Lightweight Network-Flow Intrusion Detection for IoT and IIoT Edge Communication Networks},
  author       = {Keskin, Fesih},
  note         = {Research artifact and associated manuscript},
  year         = {2026}
}
```

## License

This repository is released under the MIT License. See `LICENSE` for the full text.

## Contact

Fesih Keskin  
Department of Computer Engineering, Siirt University  
Email: `fesih.keskin@siirt.edu.tr`
