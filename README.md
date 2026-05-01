# COCOV: Continuous Collaborative Verification

A drift-aware, prototype-based framework for continuous
face verification with selective human-in-the-loop
adaptation. Identity representations evolve incrementally
through bounded prototype updates governed by appearance
drift, reducing supervisory overhead while maintaining
stable verification performance under non-stationary
conditions.

Built on a fixed deep face encoder, COCOV manages identity
memory through principled prototype assignment, insertion,
and merging, triggered selectively when incoming
observations deviate meaningfully from established identity
structure. A lightweight web-based reviewer interface
handles escalated observations that fall outside automatic
acceptance bounds.

---

## Research Context

This implementation accompanies a PhD thesis on continuous
face verification. The framework is evaluated against four
baselines --- static enrollment, naive OLS memory
expansion, replay-based dual memory, and fixed-size buffer
averaging --- across three datasets with varying degrees
of temporal appearance change.

---

## Datasets

| Dataset | Role | Identities | Ordering |
|---|---|---|---|
| VGGFace2-HQ | Primary evaluation | 3,000 subset | Filename |
| CACD | Cross-dataset validation | 2,000 | Age (year) |
| FG-NET | Diagnostic | 82 | Age (label) |

Dataset access:
- VGGFace2: https://www.robots.ox.ac.uk/~vgg/data/vggface2/
- CACD: https://bcsiriuschen.github.io/CARC/
- FG-NET: https://yanweifu.github.io/FG_NET_data/FGNET.zip

---

## Project Structure

```
cocov/
├── config/
│   └── config.yaml          # All hyperparameters
├── data/
│   ├── dataset.py           # Dataset loaders
│   ├── embeddings.py        # Embedding cache
│   └── stream.py            # Stream construction
├── models/
│   ├── encoder.py           # FaceNet encoder wrapper
│   └── identity_memory.py   # Prototype memory
├── methods/
│   ├── base.py              # Abstract base class
│   ├── static.py            # Static enrollment
│   ├── ols.py               # Naive OLS expansion
│   ├── replay.py            # Replay dual memory
│   ├── buffer.py            # Buffer averaging
│   └── cocov.py             # COCOV framework
├── verification/
│   ├── verifier.py          # Similarity and drift
│   └── metrics.py           # AUC, EER, TAR, updates
├── calibration/
│   └── calibrate.py         # Threshold calibration
├── experiments/
│   ├── run_experiment.py    # Main experiment runner
│   ├── ablation.py          # Ablation study
│   └── cross_dataset.py     # Cross-dataset evaluation
├── analysis/
│   ├── plots.py             # Figure generation
│   └── results.py           # LaTeX tables
├── webapp/
│   └── main.py              # Reviewer interface
└── tests/
```

---

## Installation

```bash
git clone https://github.com/yourusername/cocov.git
cd cocov
pip install -r requirements.txt
```

Tested on:
- Python 3.13
- PyTorch 2.7.1 with CUDA 11.8
- Ubuntu 24.04

---

## Configuration

All hyperparameters and dataset paths are defined in
`config/config.yaml`. Edit this file before running
any experiment.

Key parameters:

```yaml
dataset:
  vggface2:
    n_identities: 3000
    enrollment_size: 5

prototype:
  assign_threshold: 0.3
  new_threshold: 0.6
  merge_threshold: 0.95
  momentum: 0.9
  max_prototypes: 10

verification:
  threshold: 0.5        # calibrated automatically

drift:
  threshold: 0.35       # calibrated automatically
```

---

## Reproducing Experiments

### Step 1: Extract embeddings

Embeddings are extracted once and cached. All subsequent
runs load from cache.

```bash
python3 -c "
import sys, yaml
sys.path.insert(0, '.')
from data.dataset import VGGFace2Dataset
from data.embeddings import EmbeddingCache
from models.encoder import FaceEncoder

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

encoder = FaceEncoder(device='cuda')
ds = VGGFace2Dataset(
    root=config['paths']['vggface2_root'],
    min_images=20
)
selected = ds.select_identities(3000, seed=42)
partition = ds.build_partition(selected, 5, 1.0, 42)
cache = EmbeddingCache(
    config['paths']['embeddings_dir'], encoder, 'vggface2'
)
cache.extract_and_cache(partition)
print('Extraction complete.')
"
```

### Step 2: Run calibration and main experiment

```bash
python3 experiments/run_experiment.py \
    --config config/config.yaml
```

### Step 3: Run ablation study

```bash
python3 experiments/ablation.py \
    --config config/config.yaml
```

### Step 4: Run cross-dataset evaluation

```bash
python3 experiments/cross_dataset.py \
    --config config/config.yaml
```

### Step 5: Generate figures and tables

```bash
python3 -c "
import sys, json
sys.path.insert(0, '.')
from analysis.plots import PlotGenerator
from analysis.results import ResultsAnalyser

pg = PlotGenerator(
    '/opt/data/results',
    '/opt/data/results/figures'
)
ra = ResultsAnalyser('/opt/data/results')

agg = ra.load_aggregated_results()
abl = ra.load_ablation_results()
cal = ra.load_calibration_results()

pg.generate_all(
    aggregated_results=agg,
    ablation_results=abl,
    calibration_results=cal
)
ra.save_all_tables()
print('Done.')
"
```

---

## Reviewer Web Application

The reviewer interface handles escalated observations
that fall outside automatic acceptance bounds.

```bash
python3 webapp/main.py
```

Open `http://localhost:8000` in your browser.

The interface presents the reviewer with the probe image,
enrolled reference image, similarity score, and drift
value. The reviewer selects one of four actions:
confirm, assign, create, or reject.

All decisions are logged to `/opt/data/logs/reviewer_log.json`.

---

## Evaluated Methods

| Method | Memory | Updates | Supervision |
|---|---|---|---|
| Static Enrollment | Fixed | None | None |
| Naive OLS Expansion | Unbounded | Per sample | None |
| Replay Dual Memory | Bounded | Scheduled | None |
| Fixed Buffer Averaging | Fixed-size | Per sample | None |
| COCOV | Bounded | Drift-based | Conditional |

---

## Ablation Configurations

| Configuration | Component Disabled |
|---|---|
| COCOV-Full | Reference |
| COCOV-NoDrift | Drift gate |
| COCOV-NoMerge | Prototype merging |
| COCOV-NoReviewer | Reviewer escalation |
| COCOV-Unbounded | Memory bound (K_max) |
| COCOV-SinglePrototype | Multiple prototypes |

---

## Metrics

- **AUC**: Area under the ROC curve
- **EER**: Equal error rate
- **TAR@FAR=1%**: True accept rate at 1% false accept rate
- **Updates**: Total prototype update operations

Results are reported as means ± standard deviations
across five independent runs with different random seeds.

---

## Hardware

Experiments were conducted on:
- GPU: NVIDIA RTX 1000 Ada Generation (6GB VRAM)
- CPU: Intel i9, 32 cores
- RAM: 64GB
- OS: Ubuntu 24.04

Embedding extraction uses the GPU. All verification,
updating, and evaluation runs on CPU using pre-extracted
embeddings.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{wafula2026cocov,
  author = {Wafula, David},
  title  = {Continuous Collaborative Verification:
             Drift-Aware Prototype-Based Face Verification},
  school = {University of the Witwatersrand},
  year   = {2026}
}
```

---

## License

MIT License. See LICENSE file for details.
