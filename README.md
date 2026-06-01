# COCOV: Continuous Collaborative Verification

A drift-aware, prototype-based framework for continual
face verification with selective human-in-the-loop
adaptation.

COCOV maintains bounded identity memory through
prototype assignment, insertion, and merging while
using reviewer-mediated escalation for uncertain
observations. Built on a fixed deep face encoder, the
framework targets continual verification under
non-stationary appearance conditions where identity
representations must evolve incrementally without
retraining the underlying feature extractor.

---

## Research Context

This implementation accompanies a PhD thesis on
continuous face verification.

The framework is evaluated against four comparative
baselines — static enrollment, naive OLS memory
expansion, replay-based dual memory, and fixed-size
buffer averaging.

The reported thesis experiments use two datasets with
distinct roles:

* **VGGFace2-HQ** — primary large-scale evaluation
  environment used for calibration, baseline
  comparison, and ablation analysis.

* **FG-NET** — cross-dataset temporal evaluation used
  to assess behaviour under verified age-ordered
  appearance progression and threshold transferability.

---

## Datasets

| Dataset     | Role                              | Identities   | Ordering    |
| ----------- | --------------------------------- | ------------ | ----------- |
| VGGFace2-HQ | Primary evaluation                | 3,000 subset | Filename    |
| FG-NET      | Cross-dataset temporal evaluation | 82           | Age (label) |

Dataset access:

* VGGFace2:
  https://www.robots.ox.ac.uk/~vgg/data/vggface2/

* FG-NET:
  https://yanweifu.github.io/FG_NET_data/FGNET.zip

---

## Project Structure

```text
cocov/
├── config/
│   └── config.yaml
├── data/
│   ├── dataset.py
│   ├── embeddings.py
│   └── stream.py
├── models/
│   ├── encoder.py
│   └── identity_memory.py
├── methods/
│   ├── base.py
│   ├── static.py
│   ├── ols.py
│   ├── replay.py
│   ├── buffer.py
│   └── cocov.py
├── verification/
│   ├── verifier.py
│   └── metrics.py
├── calibration/
│   └── calibrate.py
├── experiments/
│   ├── run_experiment.py
│   ├── ablation.py
│   └── cross_dataset.py
├── analysis/
│   ├── plots.py
│   └── results.py
├── webapp/
│   └── main.py
└── tests/
```

---

## Installation

```bash
git clone https://github.com/davidwaf/cocov.git
cd cocov
pip install -r requirements.txt
```

Tested on:

* Python 3.13
* PyTorch 2.7.1 with CUDA 11.8
* Ubuntu 24.04

---

## Configuration

All configuration, dataset paths, hyperparameters,
calibration settings, and runtime options are defined in
`config/config.yaml`.

The examples below reflect the current repository
configuration used for reproducible execution.

Key parameters:

```yaml
dataset:
  vggface2:
    n_identities: 3000
    enrollment_size: 5
    min_images_per_identity: 20

experiment:
  n_runs: 30
  impostor_ratio: 1.0

verification:
  threshold: 0.5

drift:
  threshold: 0.35

prototype:
  assign_threshold: 0.3
  new_threshold: 0.6
  merge_threshold: 0.95
  momentum: 0.5
  max_prototypes: 10
```

The reported thesis evaluation uses:

* **30 independent runs** for the primary VGGFace2
  evaluation.
* **5 independent runs** for FG-NET cross-dataset
  evaluation.

---

## Reproducing Experiments

### Step 1 — Extract embeddings

Embeddings are extracted once and cached. Subsequent
experiments load directly from cache.

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

partition = ds.build_partition(
    selected,
    5,
    1.0,
    42
)

cache = EmbeddingCache(
    config['paths']['embeddings_dir'],
    encoder,
    'vggface2'
)

cache.extract_and_cache(partition)

print('Extraction complete.')
"
```

---

### Step 2 — Run calibration and primary experiment

```bash
python3 experiments/run_experiment.py \
    --config config/config.yaml
```

---

### Step 3 — Run ablation study

```bash
python3 experiments/ablation.py \
    --config config/config.yaml
```

---

### Step 4 — Run cross-dataset evaluation

```bash
python3 experiments/cross_dataset.py \
    --config config/config.yaml
```

---

### Step 5 — Generate figures and tables

```bash
python3 -c "
import sys
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

Open:

```text
http://localhost:8000
```

The interface presents:

* probe image
* enrolled reference image
* similarity score
* drift value

The reviewer selects one of four actions:

* confirm
* assign
* create
* reject

All decisions are logged to:

```text
/opt/data/logs/reviewer_log.json
```

---

## Evaluated Methods

| Method                 | Memory     | Updates     | Supervision |
| ---------------------- | ---------- | ----------- | ----------- |
| Static Enrollment      | Fixed      | None        | None        |
| Naive OLS Expansion    | Unbounded  | Per sample  | None        |
| Replay Dual Memory     | Bounded    | Scheduled   | None        |
| Fixed Buffer Averaging | Fixed-size | Per sample  | None        |
| COCOV                  | Bounded    | Drift-based | Conditional |

---

## Ablation Configurations

| Configuration         | Component Disabled      |
| --------------------- | ----------------------- |
| COCOV-Full            | Reference configuration |
| COCOV-NoDrift         | Drift gate              |
| COCOV-NoMerge         | Prototype merging       |
| COCOV-NoReviewer      | Reviewer escalation     |
| COCOV-Unbounded       | Memory bound (`K_max`)  |
| COCOV-SinglePrototype | Multiple prototypes     |

---

## Metrics

The experimental evaluation reports:

* **AUC** — Area under the ROC curve
* **EER** — Equal error rate
* **TAR@FAR=1%** — True acceptance rate at 1% false
  acceptance rate
* **Updates** — Total prototype update operations

Reported thesis results are aggregated across repeated
runs under controlled random seeds.

---

## Hardware

Experiments were conducted on:

* GPU: NVIDIA RTX 1000 Ada Generation (6GB VRAM)
* CPU: Intel i9, 32 cores
* RAM: 64GB
* OS: Ubuntu 24.04

Embedding extraction uses the GPU.

Verification, prototype updating, calibration, and
evaluation operate on CPU using pre-extracted
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

MIT License.

See `LICENSE` for details.
