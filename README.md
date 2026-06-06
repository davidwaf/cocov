# COCOV: Continuous Collaborative Verification

A drift-aware, prototype-based framework for continual face verification
with selective human-in-the-loop adaptation, evaluated across five
backbone encoders and three datasets.

COCOV maintains bounded identity memory through prototype assignment,
insertion, and merging while using reviewer-mediated escalation for
uncertain observations. Built on a fixed deep face encoder, the framework
targets continual verification under non-stationary appearance conditions
where identity representations must evolve incrementally without
retraining the underlying feature extractor.

---

## Research Context

This implementation accompanies a PhD thesis on continuous face
verification:

> **Continuous Identity Representation Adaptation for Long-Term Face
> Verification**
> David Wafula, University of the Witwatersrand, 2026

The framework is evaluated against four baselines — static enrollment,
naive OLS memory expansion, replay-based dual memory, and fixed-size
buffer averaging — across five backbone encoders and three datasets with
varying degrees of temporal appearance change.

---

## Results Summary

COCOV consistently outperforms all baselines across all five encoders
evaluated on VGGFace2 (30 independent runs, mean ± std):

| Encoder        | Static AUC | COCOV AUC  | COCOV EER  |
|----------------|-----------|-----------|-----------|
| FaceNet        | 0.9815    | 0.9843±0.0020 | 3.94±0.47% |
| ArcFace-R50    | 0.9782    | 0.9844±0.0018 | 3.86±0.36% |
| ArcFace-R100   | 0.9787    | 0.9844±0.0016 | 3.88±0.34% |
| MobileFaceNet  | 0.9787    | 0.9841±0.0017 | 3.94±0.40% |
| AdaFace        | 0.9778    | 0.9830±0.0016 | 3.94±0.36% |

COCOV achieves its largest gains on FG-NET (age-ordered temporal
evaluation), with +2.77 pp AUC over Static Enrollment for FaceNet and
+1.64 pp for MobileFaceNet, demonstrating that adaptive memory provides
the greatest benefit under genuine temporal appearance drift.

---

## Datasets

| Dataset      | Role                          | Identities    | Ordering    |
|--------------|-------------------------------|---------------|-------------|
| VGGFace2-HQ  | Primary evaluation            | 3,000 subset  | Filename    |
| CACD         | Cross-dataset (cross-age)     | 500 (capped)  | Age (year)  |
| FG-NET       | Cross-dataset (temporal, age) | 82            | Age (label) |

Dataset access:

- VGGFace2: https://www.robots.ox.ac.uk/~vgg/data/vggface2/
- CACD: https://bcsiriuschen.github.io/CARC/
- FG-NET: https://yanweifu.github.io/FG_NET_data/FGNET.zip

---

## Encoders

Five backbone encoders are evaluated. All are fixed throughout
experiments — no fine-tuning is performed.

| Code              | Encoder        | Backbone         | Loss      | Training Data | Input    |
|-------------------|----------------|------------------|-----------|---------------|----------|
| ENC01\_FACENET    | FaceNet        | InceptionResNetV1| Triplet   | VGGFace2      | 160×160  |
| ENC02\_ARCFACE\_R50 | ArcFace-R50  | IResNet-50       | ArcFace   | WebFace600K   | 112×112  |
| ENC03\_ARCFACE\_R100 | ArcFace-R100 | IResNet-100     | ArcFace   | Glint360K     | 112×112  |
| ENC04\_MOBILEFACENET | MobileFaceNet | MobileFaceNet  | ArcFace   | WebFace600K   | 112×112  |
| ENC04\_ADAFACE    | AdaFace        | IResNet-101      | AdaFace   | MS1MV3        | 112×112  |

See [ENCODERS.md](ENCODERS.md) for installation and weight acquisition
instructions per encoder.

---

## Project Structure

```text
cocov/
├── config/
│   └── config.yaml              # All paths, hyperparameters, calibration settings
├── data/
│   ├── dataset.py               # VGGFace2Dataset, CACDDataset, FGNETDataset
│   ├── embeddings.py            # EmbeddingCache with per-encoder subdirectories
│   └── stream.py                # Sequential verification stream construction
├── models/
│   ├── encoder.py               # Multi-backbone encoder registry (5 encoders)
│   └── identity_memory.py       # Prototype memory with assignment/insertion/merging
├── methods/
│   ├── base.py                  # Abstract BaseVerificationMethod
│   ├── static.py                # Static Enrollment baseline
│   ├── ols.py                   # Naive OLS Expansion baseline
│   ├── replay.py                # Replay Dual Memory baseline
│   ├── buffer.py                # Fixed Buffer Averaging baseline
│   └── cocov.py                 # COCOV framework
├── verification/
│   ├── verifier.py              # Similarity and drift computation
│   └── metrics.py               # AUC, EER, TAR@FAR, update counts
├── calibration/
│   └── calibrate.py             # Threshold calibration via grid search
├── experiments/
│   ├── run_experiment.py        # Main experiment runner (30 runs)
│   ├── ablation.py              # Ablation study (6 configurations)
│   └── cross_dataset.py        # Cross-dataset evaluation (CACD, FG-NET)
├── scripts/
│   ├── run_encoder.sh           # Single-encoder pipeline (4 steps)
│   ├── patch_config.py          # Config patching utility
│   └── extract_embeddings.py   # Per-encoder embedding extraction
├── analysis/
│   ├── generate_results.py     # 10 figures + 4 LaTeX tables from JSON results
│   ├── plots.py                 # Legacy figure generation
│   └── results.py               # Legacy LaTeX table generation
├── webapp/
│   └── main.py                  # Reviewer escalation web interface
└── tests/
    ├── test_dataset.py
    ├── test_memory.py
    ├── test_metrics.py
    └── test_verifier.py
```

---

## Installation

```bash
git clone https://github.com/davidwaf/cocov.git
cd cocov
pip install -r requirements.txt
```

Tested on:

- Python 3.13
- PyTorch 2.7.1 with CUDA 12.8
- Ubuntu 24.04
- NVIDIA RTX 1000 Ada Generation (6 GB VRAM)

For encoder-specific dependencies (ArcFace, AdaFace, MobileFaceNet),
see [ENCODERS.md](ENCODERS.md).

---

## Reproducing Experiments

### Quickstart — single encoder

The recommended entry point is `scripts/run_encoder.sh`, which runs the
complete four-step pipeline for one encoder:

```bash
cd /path/to/cocov
bash scripts/run_encoder.sh facenet
```

Valid encoder names: `facenet`, `arcface_r50`, `arcface_r100`,
`mobilefacenet`, `adaface`

This script:
1. Patches `config/config.yaml` with the encoder's settings
2. Extracts and caches embeddings for VGGFace2, CACD, and FG-NET
3. Runs calibration + main experiment (30 runs)
4. Runs cross-dataset evaluation + ablation study

Results are written to `/opt/data/cocov/results/ENC0X_*/`.

---

### Step-by-step

#### 1 — Configure paths

Edit `config/config.yaml` to set your dataset and output paths:

```yaml
paths:
  vggface2_root: "/path/to/VGGface2_None_norm_512_true_bygfpgan"
  cacd_root:     "/path/to/cacd/cacd_split"
  fgnet_root:    "/path/to/FGNET/images"
  embeddings_dir: "/path/to/embeddings"
  results_dir:   "/path/to/results"
```

#### 2 — Extract embeddings

```bash
python scripts/extract_embeddings.py \
    --encoder facenet \
    --config config/config.yaml
```

Embeddings are cached under `{embeddings_dir}/{encoder}/`. Subsequent
runs load from cache automatically.

#### 3 — Run calibration and main experiment

```bash
python experiments/run_experiment.py --config config/config.yaml
```

#### 4 — Run ablation study

```bash
python experiments/ablation.py --config config/config.yaml
```

#### 5 — Run cross-dataset evaluation

```bash
python experiments/cross_dataset.py --config config/config.yaml
```

#### 6 — Generate figures and tables

```bash
python analysis/generate_results.py \
    --results_dir /opt/data/cocov/results \
    --output_dir  /opt/data/cocov/analysis
```

Outputs:

```text
analysis/
├── figures/
│   ├── fig1_main_auc_grouped.pdf
│   ├── fig2_main_eer_grouped.pdf
│   ├── fig3_cocov_vs_static_scatter.pdf
│   ├── fig4_ablation_heatmap.pdf
│   ├── fig5_ablation_bars.pdf
│   ├── fig6_cross_dataset_auc.pdf
│   ├── fig6_cross_dataset_eer.pdf
│   ├── fig8_updates_bar.pdf
│   ├── fig9_radar.pdf
│   └── fig10_cocov_gain.pdf
└── tables/
    ├── tab1_main_results.tex
    ├── tab2_cross_dataset.tex
    ├── tab3_ablation.tex
    └── tab4_encoder_summary.tex
```

All figures are saved as both `.pdf` (for LaTeX) and `.png` (for
preview). LaTeX tables can be included directly with `\input{}`.

---

## Reviewer Web Application

The reviewer interface handles escalated observations that fall outside
automatic acceptance bounds.

```bash
python webapp/main.py
```

Open `http://localhost:8000`

The interface presents probe image, enrolled reference, similarity score,
and drift value. The reviewer selects: confirm, assign, create, or
reject. All decisions are logged to
`/opt/data/logs/reviewer_log.json`.

In experiments, reviewer responses are **simulated** using ground-truth
labels, providing an upper bound on collaborative performance under ideal
reviewer conditions.

---

## Evaluated Methods

| Method                  | Memory     | Updates      | Supervision |
|-------------------------|------------|--------------|-------------|
| Static Enrollment       | Fixed      | None         | None        |
| Naive OLS Expansion     | Unbounded  | Per sample   | None        |
| Replay Dual Memory      | Bounded    | Scheduled    | None        |
| Fixed Buffer Averaging  | Fixed-size | Per sample   | None        |
| **COCOV**               | **Bounded**| **Drift-gated** | **Conditional** |

---

## Ablation Configurations

| Configuration          | Component Disabled       |
|------------------------|--------------------------|
| COCOV-Full             | Reference configuration  |
| COCOV-NoDrift          | Drift gate               |
| COCOV-NoMerge          | Prototype merging        |
| COCOV-NoReviewer       | Reviewer escalation      |
| COCOV-Unbounded        | Memory bound (K_max)     |
| COCOV-SinglePrototype  | Multiple prototypes      |

Key ablation findings (across all 5 encoders):
- **Drift gate** is the most critical component (−0.006 to −0.010 AUC)
- **Single prototype** causes the second-largest drop (−0.003 to −0.008)
- **Removing merge** slightly improves AUC (+0.0008) — suggesting merge
  is conservative; removing it allows more prototype diversity
- **Removing the memory bound** has negligible impact — the drift gate
  implicitly limits update frequency

---

## COCOV Hyperparameters

All hyperparameters are calibrated per-encoder on a held-out
calibration set of 200 identities not used in evaluation.

| Parameter        | Symbol         | Description                              |
|------------------|----------------|------------------------------------------|
| `assign_threshold` | ρ_assign     | Max cosine distance for assignment       |
| `new_threshold`    | ρ_new        | Min cosine distance for new prototype    |
| `merge_threshold`  | ρ_merge      | Min cosine similarity for merging        |
| `momentum`         | γ            | Momentum for prototype assignment update |
| `max_prototypes`   | K_max        | Max prototypes per identity              |
| `verification_threshold` | τ_ver | Min similarity for acceptance            |
| `drift_threshold`  | τ_Δ          | Max drift for update eligibility         |

---

## Metrics

| Metric      | Description                                        |
|-------------|----------------------------------------------------|
| AUC         | Area under the ROC curve                           |
| EER         | Equal error rate                                   |
| TAR@FAR=1%  | True acceptance rate at 1% false acceptance rate   |
| Updates     | Total prototype update operations per run          |

Results are reported as mean ± std across independent runs (30 for
VGGFace2, 5 for CACD and FG-NET).

---

## Hardware

Experiments conducted on:

- GPU: NVIDIA RTX 1000 Ada Generation (6 GB VRAM)
- CPU: Intel i9, 32 cores
- RAM: 64 GB
- OS: Ubuntu 24.04

Embedding extraction uses GPU. Verification, calibration, and evaluation
operate on CPU from pre-extracted embeddings.

---

## Tests

```bash
pytest tests/ -v
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{wafula2026cocov,
  author = {Wafula, David},
  title  = {Continuous Identity Representation Adaptation
             for Long-Term Face Verification},
  school = {University of the Witwatersrand},
  year   = {2026}
}
```

---

## License

MIT License. See `LICENSE` for details.
