# Encoder Installation Guide

This document describes how to install weights and dependencies for each
of the five backbone encoders used in COCOV experiments.

All encoders produce 512-dimensional L2-normalised embeddings. All are
used in fixed mode — no fine-tuning is performed at any stage.

---

## ENC01 — FaceNet (InceptionResNetV1)

**Backbone:** InceptionResNetV1  
**Loss:** Triplet  
**Training data:** VGGFace2  
**Input size:** 160×160  

```bash
pip install facenet-pytorch==2.6.0
```

Weights are downloaded automatically on first use via `facenet-pytorch`.

```python
from models.encoder import get_encoder
encoder = get_encoder("facenet", device="cuda")
```

---

## ENC02 — ArcFace-R50

**Backbone:** IResNet-50  
**Loss:** ArcFace  
**Training data:** WebFace600K  
**Input size:** 112×112  

```bash
pip install onnxruntime-gpu
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Weights are downloaded automatically via InsightFace model zoo on first
use.

```python
from models.encoder import get_encoder
encoder = get_encoder("arcface_r50", device="cuda")
```

If CUDA fails after a previous crash, reset the UVM module:

```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```

---

## ENC03 — ArcFace-R100

**Backbone:** IResNet-100  
**Loss:** ArcFace  
**Training data:** Glint360K  
**Input size:** 112×112  

Same dependencies as ArcFace-R50.

```python
from models.encoder import get_encoder
encoder = get_encoder("arcface_r100", device="cuda")
```

---

## ENC04 — MobileFaceNet

**Backbone:** MobileFaceNet  
**Loss:** ArcFace  
**Training data:** WebFace600K  
**Input size:** 112×112  

Same dependencies as ArcFace-R50 (uses InsightFace ONNX model zoo).

```python
from models.encoder import get_encoder
encoder = get_encoder("mobilefacenet", device="cuda")
```

---

## ENC04 — AdaFace (IResNet-101)

**Backbone:** IResNet-101  
**Loss:** AdaFace (quality-adaptive margin)  
**Training data:** MS1MV3  
**Input size:** 112×112  

AdaFace is loaded via CVLFace pretrained weights. The loader uses
subprocess isolation to prevent `sys.path` manipulation from corrupting
the CUDA context in the main process.

### 1. Install dependencies

```bash
pip install fvcore
```

### 2. Download weights

```bash
git clone https://huggingface.co/minchul/cvlface_adaface_ir101_ms1mv3 \
    /opt/code/CVLface/pretrained/adaface
```

Or using `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="minchul/cvlface_adaface_ir101_ms1mv3",
    local_dir="/opt/code/CVLface/pretrained/adaface"
)
```

### 3. Verify installation

```python
import torch, sys, os
from types import SimpleNamespace
import yaml

weights_dir = "/opt/code/CVLface/pretrained/adaface"
os.chdir(weights_dir)
sys.path.insert(0, weights_dir)
from models import get_model

conf = SimpleNamespace(**yaml.safe_load(open("pretrained_model/model.yaml")))
model = get_model(conf)
ckpt = torch.load("pretrained_model/model.pt", map_location="cpu", weights_only=False)
state = {k[4:] if k.startswith("net.") else k: v for k, v in ckpt.items()}
model.load_state_dict(state)
model.eval()
out = model(torch.randn(1, 3, 112, 112))
print("Output shape:", out.shape)  # torch.Size([1, 512])
```

### 4. Set weights directory

By default the encoder looks for weights at
`/opt/code/CVLface/pretrained/adaface`. Override with:

```bash
export ADAFACE_WEIGHTS_DIR=/your/path/to/adaface
```

```python
from models.encoder import get_encoder
encoder = get_encoder("adaface", device="cuda")
```

---

## Troubleshooting

### CUDA unknown error after encoder load

The `sys.path`/`sys.modules` manipulation required for AdaFace can
corrupt the CUDA UVM context in some environments. Reset with:

```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
python3 -c "import torch; print(torch.cuda.is_available())"
```

### ONNX provider not found

Ensure `onnxruntime-gpu` is installed (not `onnxruntime`):

```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### transformers version conflict (AdaFace)

AdaFace via `AutoModel.from_pretrained` requires `transformers<5.0`.
The subprocess isolation in `AdaFaceEncoder` avoids this by loading
directly from `pretrained_model/model.pt` without using `transformers`.
No version downgrade is needed.
