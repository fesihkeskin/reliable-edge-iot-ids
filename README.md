# Reliability-Aware Lightweight Intrusion Detection for IoT and IIoT Edge Communication Networks

Notebook and reusable code for experiments on lightweight network-flow intrusion detection in IoT and IIoT edge communication networks.

## Overview

The study evaluates reliability-aware intrusion detection using CICIoT2023 as the main benchmark and Edge-IIoTset as an independent validation dataset. The workflow compares Logistic Regression, Random Forest, LightGBM, and a compact TinyMLP model under binary detection, multiclass attack classification, leave-one-attack-category-out testing, calibration, feature ablation, and TinyMLP compression settings.

The repository keeps the Colab notebook as the main entry point and extracts reusable preprocessing, model, training, and evaluation code into `src/`.

## Repository Structure

```text
configs/default.yaml                         Default paths and run parameters
docs/data_card.md                            Dataset and preprocessing notes
docs/reproducibility.md                      Environment and run steps
docs/cleanup_notes.md                        Remaining release notes
figures/README.md                            Shared-figure location
notebooks/lightweight_iot_ids_ciciot_edgeiiot_colab.ipynb
paper/manuscript.tex                         Manuscript source
references/README.md                         Reference-file notes
results/README.md                            Generated-output notes
scripts/run_preprocessing.py                 Local preprocessing wrapper
scripts/run_training.py                      Local single-model training wrapper
scripts/run_evaluation.py                    Local evaluation wrapper
src/                                          Reusable data, model, training, and metric code
CITATION.cff
README.md
requirements.txt
```

## Requirements

Use Python 3.10+ for local execution. Install the package set with:

```bash
pip install -r requirements.txt
```

The notebook includes a Colab setup cell for packages that are not always preinstalled.

## Running the Notebook

Open `notebooks/lightweight_iot_ids_ciciot_edgeiiot_colab.ipynb` in Google Colab or Jupyter. Run the sections in order:

```text
Setup -> Imports -> Configuration -> Data loading -> Preprocessing -> Training -> Evaluation -> Result export
```

Google Drive mounting is optional and disabled by default. In the Configuration section, set the dataset paths or edit `configs/default.yaml`. For a notebook smoke test, set `RUN_SYNTHETIC_DEMO = True`; do not report synthetic-demo results as paper results.

## Data

CICIoT2023 and Edge-IIoTset are not redistributed in this repository. Obtain the datasets from their original providers and place them under the configured `data/raw/` paths, or set the notebook path variables directly.

Default expected layout:

```text
data/raw/
  CIC_IOT_Dataset2023/
    CICIoT2023.csv
    CSV/MERGED_CSV/Merged*.csv
  Edge-IIoTset/
    DNN-EdgeIIoT-dataset.csv
```

## Experiments

The notebook and `src/` code support:

- CICIoT2023 binary and multiclass baselines
- CICIoT2023 leave-one-attack-category-out evaluation
- feature-group ablation
- calibration and confidence-threshold analysis
- TinyMLP pruning and INT8 dynamic quantization
- Edge-IIoTset independent validation
- manuscript table and figure export from generated CSV files

For local script use:

```bash
python scripts/run_preprocessing.py --dataset ciciot
python scripts/run_training.py --dataset ciciot --task binary --model lgbm
python scripts/run_evaluation.py --dataset ciciot --task binary --model lgbm
```

## Results

Generated CSV files, model artifacts, tables, and figures are written under the configured results directory, which defaults to `results/`. Generated outputs and dataset-derived files are not committed.

## Citation

Publication details are not available. Use this temporary BibTeX entry until final citation metadata is available:

```bibtex
@misc{keskin_reliability_iot_iiot_ids,
  title = {Reliability-Aware Lightweight Intrusion Detection for IoT and IIoT Edge Communication Networks},
  author = {Keskin, Fesih},
  note = {Associated manuscript}
}
```

## License

No license file is included. A license should be selected before public release.

## Contact

Fesih Keskin, Department of Computer Engineering, Siirt University. Email: `fesih.keskin@siirt.edu.tr`.
