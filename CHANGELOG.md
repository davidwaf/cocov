# Changelog

All notable changes to COCOV are documented here.

---

## [Unreleased] — Multi-backbone evaluation

### Added

- **`models/encoder.py`** — Multi-backbone encoder registry with five
  encoders: FaceNet (ENC01), ArcFace-R50 (ENC02), ArcFace-R100 (ENC03),
  MobileFaceNet (ENC04), AdaFace (ENC04). Includes `BaseEncoder` abstract
  class, `get_encoder()` factory, and `list_encoders()` utility.
  `FaceEncoder` alias preserved for backward compatibility.

- **`models/encoder.py` — AdaFace subprocess isolation** — AdaFace is
  loaded in a subprocess to prevent `sys.path`/`sys.modules` manipulation
  from corrupting the CUDA UVM context in the main process. State dict is
  passed via pickle to the parent process, which reconstructs the
  IResNet-101 architecture directly.

- **`scripts/run_encoder.sh`** — Four-step single-encoder pipeline
  runner: (1) patch config, (2) extract embeddings, (3) calibrate +
  run experiment, (4) cross-dataset + ablation.

- **`scripts/patch_config.py`** — Encoder registry and config patching
  utility. Maps encoder names to codes, input sizes, and cache
  subdirectories.

- **`scripts/extract_embeddings.py`** — Per-encoder embedding extraction
  with dataset-aware caching for VGGFace2, CACD, and FG-NET.

- **`analysis/generate_results.py`** — Comprehensive results analysis
  script producing 10 publication figures and 4 LaTeX tables from JSON
  result files across all encoders. Reads flat JSON structure produced by
  experiment runners.

- **`ENCODERS.md`** — Encoder installation guide with weight acquisition
  instructions, verification steps, and troubleshooting for all five
  backbones.

### Changed

- **`config/config.yaml`** — Added `encoder` section with per-backbone
  configuration: `name`, `code`, `input_size`, `cache_subdir`,
  `embedding_dim`. Added `cacd` dataset section with `max_identities: 500`
  to prevent OOM during cross-dataset evaluation.

- **`experiments/run_experiment.py`** — Updated to use `get_encoder()`
  factory instead of direct `FaceEncoder` instantiation.

- **`experiments/cross_dataset.py`** — Updated encoder loading; CACD
  capped at 500 identities; `batch_size=4` for memory stability on 6 GB
  VRAM.

- **`experiments/ablation.py`** — Updated encoder loading.

### Fixed

- CUDA context corruption when loading AdaFace via CVLFace — resolved by
  subprocess isolation.
- `onnxruntime` vs `onnxruntime-gpu` conflict for InsightFace ONNX
  models — documented in ENCODERS.md.
- `net.` prefix in AdaFace state dict keys — stripped before
  `load_state_dict` to match raw IResNet-101 parameter names.

---

## [Initial release] — FaceNet baseline

### Added

- Core COCOV framework: prototype assignment, insertion, merging,
  reviewer escalation.
- Four comparison baselines: Static Enrollment, Naive OLS Expansion,
  Replay Dual Memory, Fixed Buffer Averaging.
- Threshold calibration via grid search on held-out identities.
- VGGFace2 and FG-NET dataset loaders.
- Reviewer web application (FastAPI).
- Unit tests for dataset, memory, metrics, and verifier modules.
