"""
encoder.py
----------
Encoder abstraction layer for multi-backbone face verification experiments.

Provides a common interface (BaseEncoder) for all face encoders evaluated in
this study. Concrete subclasses wrap five different pre-trained backbones:

    FaceNetEncoder       -- InceptionResNetV1, VGGFace2  (existing baseline)
    ArcFaceR50Encoder    -- IResNet50, WebFace600K        (InsightFace buffalo_l)
    ArcFaceR100Encoder   -- IResNet100, Glint360K         (InsightFace antelopev2)
    AdaFaceEncoder       -- IResNet100, MS1MV3            (mk-minchul/AdaFace)
    ViTArcFaceEncoder    -- ViT-B/16, MS1MV3              (InsightFace / HuggingFace)

All subclasses honour the same contract:
    - encode(image)  -> torch.Tensor shape (512,), L2-normalised, on CPU
    - encode_path()  -> single path convenience wrapper
    - encode_batch() -> batched extraction for cache building
    - info property  -> dict for logging / reproducibility records

Factory
-------
    enc = get_encoder(name, device='cuda')

    Supported names (also used as cache subdirectory keys):
        'facenet'       -> FaceNetEncoder
        'arcface_r50'   -> ArcFaceR50Encoder
        'arcface_r100'  -> ArcFaceR100Encoder
        'adaface'       -> AdaFaceEncoder
        'vitb_arcface'  -> ViTArcFaceEncoder

Each encoder produces 512-dimensional L2-normalised embeddings so that the
downstream COCOV pipeline requires no modification across backbones.

Backward compatibility
----------------------
FaceEncoder is aliased to FaceNetEncoder so that existing import statements
    from models.encoder import FaceEncoder
continue to work without modification.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseEncoder(ABC):
    """
    Abstract base class for all face encoders.

    Every subclass must implement _build_model() and _forward(), and must
    declare name, embed_dim, and img_size as class-level attributes.

    The encode() contract:
        - Accepts a PIL Image (any mode; internally converted to RGB).
        - Returns a 1-D torch.Tensor of shape (embed_dim,) on CPU.
        - The returned tensor is L2-normalised (norm == 1.0).

    The public API (encode, encode_path, encode_batch) is identical across
    all subclasses. The COCOV pipeline calls only these methods; it has no
    knowledge of which concrete encoder is in use.

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'. Falls back to CPU automatically when CUDA is
        unavailable.
    """

    #: Canonical key used in config.yaml and as the cache subdirectory name.
    name: str = NotImplemented
    #: Output embedding dimensionality (512 for all current backbones).
    embed_dim: int = 512
    #: Side length (pixels) of the square input expected by the backbone.
    img_size: int = NotImplemented

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available. Falling back to CPU."
            )
        self._build_model()
        self._build_transform()
        logger.info(
            f"{self.__class__.__name__} loaded on {self.device} "
            f"(img_size={self.img_size}, embed_dim={self.embed_dim})"
        )

    @abstractmethod
    def _build_model(self) -> None:
        """Instantiate the backbone and freeze all parameters."""

    def _build_transform(self) -> None:
        """
        Standard preprocessing: resize to img_size, normalise to [-1, 1].
        Subclasses may override for backbone-specific conventions.
        """
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
        ])

    @abstractmethod
    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone forward pass.

        Parameters
        ----------
        tensor : torch.Tensor
            Batch of shape (N, 3, img_size, img_size) on self.device.

        Returns
        -------
        torch.Tensor
            Raw (pre-normalisation) embeddings, shape (N, embed_dim).
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, image: Image.Image) -> torch.Tensor:
        """
        Extract a unit-normalised embedding from a single PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            Pre-aligned face crop. Internally converted to RGB.

        Returns
        -------
        torch.Tensor
            L2-normalised embedding of shape (embed_dim,) on CPU.
        """
        tensor = self.transform(image.convert("RGB"))
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = self._forward(tensor)
            emb = F.normalize(raw, p=2, dim=1)
        return emb.squeeze(0).cpu()

    def encode_path(self, image_path: str | Path) -> torch.Tensor:
        """
        Extract a unit-normalised embedding from an image file path.

        Parameters
        ----------
        image_path : str or Path

        Returns
        -------
        torch.Tensor
            L2-normalised embedding of shape (embed_dim,) on CPU.

        Raises
        ------
        FileNotFoundError
        RuntimeError
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open image {image_path}: {exc}"
            ) from exc
        return self.encode(image)

    def encode_batch(
        self,
        image_paths: list[str | Path],
        batch_size: int = 64
    ) -> torch.Tensor:
        """
        Extract embeddings for a list of image paths in GPU batches.

        Parameters
        ----------
        image_paths : list of str or Path
        batch_size : int
            Images per GPU batch. Reduce if GPU memory is exceeded.

        Returns
        -------
        torch.Tensor
            L2-normalised embeddings of shape (N, embed_dim) on CPU.

        Raises
        ------
        RuntimeError
            If no embeddings could be extracted.
        """
        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors: list[torch.Tensor] = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self.transform(img))
                except Exception as exc:
                    logger.warning(f"Skipping {path}: {exc}")

            if not tensors:
                continue

            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                raw = self._forward(batch)
                emb = F.normalize(raw, p=2, dim=1)
            all_embeddings.append(emb.cpu())

        if not all_embeddings:
            raise RuntimeError(
                "No embeddings extracted. Check that image paths are valid."
            )
        return torch.cat(all_embeddings, dim=0)

    @property
    def info(self) -> dict:
        """Encoder configuration dict for logging and reproducibility."""
        return {
            "encoder_name": self.name,
            "class": self.__class__.__name__,
            "embed_dim": self.embed_dim,
            "img_size": self.img_size,
            "device": str(self.device),
            "trainable_params": 0,
        }


# ---------------------------------------------------------------------------
# Encoder 1: FaceNet / InceptionResNetV1  (existing baseline)
# ---------------------------------------------------------------------------

class FaceNetEncoder(BaseEncoder):
    """
    InceptionResNetV1 pre-trained on VGGFace2 via facenet_pytorch.

    This is the original single-backbone encoder. It is included here as
    Encoder 1 of the multi-backbone comparison so that all five backbones
    pass through the same experimental pipeline and results are directly
    comparable.

    Weights are downloaded automatically on first use by facenet_pytorch.

    Attributes
    ----------
    name : 'facenet'
    img_size : 160
    embed_dim : 512
    Training data : VGGFace2  (~3.3M images, 9131 identities)
    Loss : Triplet loss (Schroff et al., 2015)
    """

    name = "facenet"
    embed_dim = 512
    img_size = 160

    def __init__(self, device: str = "cuda",
                 pretrained: str = "vggface2") -> None:
        self._pretrained = pretrained
        super().__init__(device)

    def _build_model(self) -> None:
        from facenet_pytorch import InceptionResnetV1
        self.model = (
            InceptionResnetV1(pretrained=self._pretrained)
            .eval()
            .to(self.device)
        )
        for p in self.model.parameters():
            p.requires_grad = False

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)

    @property
    def info(self) -> dict:
        d = super().info
        d.update({"pretrained": self._pretrained,
                   "training_data": "VGGFace2",
                   "loss": "triplet"})
        return d


# ---------------------------------------------------------------------------
# Encoder 2: ArcFace / IResNet50  (InsightFace buffalo_l)
# ---------------------------------------------------------------------------

class ArcFaceR50Encoder(BaseEncoder):
    """
    IResNet50 trained with ArcFace loss on WebFace600K.

    Uses InsightFace's buffalo_l model pack (w600k_r50.onnx). The model is
    accessed via InsightFace's ONNX runtime backend; no PyTorch training code
    is needed. Images are pre-aligned face crops, so the InsightFace detection
    pipeline is bypassed; only the recognition sub-model is used.

    Installation
    ------------
        pip install insightface onnxruntime-gpu   # GPU
        pip install insightface onnxruntime        # CPU

    The model pack downloads automatically to ~/.insightface/models/buffalo_l/
    on first use.

    Attributes
    ----------
    name : 'arcface_r50'
    img_size : 112
    embed_dim : 512
    Training data : WebFace600K  (~600K images, 42K identities)
    Loss : ArcFace (Deng et al., CVPR 2019)
    Pack : buffalo_l
    """

    name = "arcface_r50"
    embed_dim = 512
    img_size = 112

    def _build_model(self) -> None:
        try:
            from insightface.model_zoo import get_model
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required for ArcFaceR50Encoder. "
                "Install: pip install insightface onnxruntime-gpu"
            ) from exc

        model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
        rec_path = os.path.join(model_dir, "w600k_r50.onnx")

        if not os.path.exists(rec_path):
            logger.info("Downloading buffalo_l model pack via InsightFace...")
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

        self.rec_model = get_model(
            rec_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.rec_model.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1
        )

    def _build_transform(self) -> None:
        # InsightFace handles its own preprocessing internally via get_feat.
        # We define transform only to satisfy the ABC; it is never called
        # because encode() and encode_batch() are overridden below.
        self.transform = transforms.Resize((self.img_size, self.img_size))

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Not used — encode() and encode_batch() bypass the tensor pipeline.
        raise NotImplementedError("Use encode() or encode_batch() directly.")

    def _pil_to_bgr(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to (H, W, 3) BGR uint8 numpy array.
        Do NOT resize here — cv2.dnn.blobFromImages inside get_feat
        handles resizing to self.input_size internally.
        """
        arr = np.array(image.convert("RGB"), dtype=np.uint8)  # (H, W, 3) RGB
        return arr[:, :, ::-1].copy()                          # -> BGR

    def encode(self, image: Image.Image) -> torch.Tensor:
        """Extract L2-normalised embedding directly from a PIL image."""
        bgr = self._pil_to_bgr(image)
        feat = self.rec_model.get_feat([bgr])                  # list of (H,W,3)
        emb = torch.from_numpy(feat.squeeze().astype(np.float32))
        return torch.nn.functional.normalize(emb.unsqueeze(0), p=2, dim=1).squeeze(0)

    def encode_path(self, image_path) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.encode(Image.open(path))

    def encode_batch(self, image_paths, batch_size: int = 64) -> torch.Tensor:
        """Extract embeddings for a list of image paths."""
        all_embeddings = []
        for path in image_paths:
            try:
                emb = self.encode_path(path)
                all_embeddings.append(emb)
            except Exception as exc:
                logger.warning(f"Skipping {path}: {exc}")
        if not all_embeddings:
            raise RuntimeError("No embeddings extracted.")
        return torch.stack(all_embeddings, dim=0)

    @property
    def info(self) -> dict:
        d = super().info
        d.update({"pretrained": "buffalo_l",
                   "training_data": "WebFace600K",
                   "loss": "ArcFace"})
        return d


# ---------------------------------------------------------------------------
# Encoder 3: ArcFace / IResNet100  (InsightFace antelopev2)
# ---------------------------------------------------------------------------

class ArcFaceR100Encoder(BaseEncoder):
    """
    IResNet100 trained with ArcFace loss on Glint360K.

    Uses InsightFace's antelopev2 model pack (glintr100.onnx). This is the
    highest-accuracy open-source CNN-family encoder in the InsightFace model
    zoo and serves as the performance ceiling for convolutional backbones in
    this comparison.

    Installation
    ------------
        pip install insightface onnxruntime-gpu

    Model pack downloads to ~/.insightface/models/antelopev2/ on first use.

    Attributes
    ----------
    name : 'arcface_r100'
    img_size : 112
    embed_dim : 512
    Training data : Glint360K  (~17M images, 360K identities)
    Loss : ArcFace (Deng et al., CVPR 2019)
    Pack : antelopev2
    """

    name = "arcface_r100"
    embed_dim = 512
    img_size = 112

    def _build_model(self) -> None:
        try:
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise ImportError(
                "insightface is required for ArcFaceR100Encoder. "
                "Install: pip install insightface onnxruntime-gpu"
            ) from exc

        model_dir = os.path.expanduser("~/.insightface/models/antelopev2")
        rec_path = os.path.join(model_dir, "glintr100.onnx")

        if not os.path.exists(rec_path):
            # Download manually via requests
            import zipfile, urllib.request
            logger.info("Downloading antelopev2 model pack...")
            os.makedirs(model_dir, exist_ok=True)
            zip_path = model_dir + ".zip"
            url = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(os.path.dirname(model_dir))
            os.remove(zip_path)
            logger.info("antelopev2 downloaded.")

        if not os.path.exists(rec_path):
            raise FileNotFoundError(
                f"glintr100.onnx not found at {rec_path} after download."
            )

        self.rec_model = get_model(
            rec_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.rec_model.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1
        )

    def _build_transform(self) -> None:
        # InsightFace handles its own preprocessing internally via get_feat.
        self.transform = transforms.Resize((self.img_size, self.img_size))

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use encode() or encode_batch() directly.")

    def _pil_to_bgr(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        return arr[:, :, ::-1].copy()

    def encode(self, image: Image.Image) -> torch.Tensor:
        bgr = self._pil_to_bgr(image)
        feat = self.rec_model.get_feat([bgr])
        emb = torch.from_numpy(feat.squeeze().astype(np.float32))
        return torch.nn.functional.normalize(emb.unsqueeze(0), p=2, dim=1).squeeze(0)

    def encode_path(self, image_path) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.encode(Image.open(path))

    def encode_batch(self, image_paths, batch_size: int = 64) -> torch.Tensor:
        all_embeddings = []
        for path in image_paths:
            try:
                all_embeddings.append(self.encode_path(path))
            except Exception as exc:
                logger.warning(f"Skipping {path}: {exc}")
        if not all_embeddings:
            raise RuntimeError("No embeddings extracted.")
        return torch.stack(all_embeddings, dim=0)

    @property
    def info(self) -> dict:
        d = super().info
        d.update({"pretrained": "antelopev2",
                   "training_data": "Glint360K",
                   "loss": "ArcFace"})
        return d


# ---------------------------------------------------------------------------
# Encoder 4: AdaFace / IResNet100  (mk-minchul/AdaFace)
# ---------------------------------------------------------------------------

class AdaFaceEncoder(BaseEncoder):
    """
    IResNet101 trained with AdaFace (quality-adaptive margin) on MS1MV3.

    Loaded via CVLFace pretrained weights at:
        /opt/code/CVLface/pretrained/adaface/

    AdaFace adapts the angular margin during training based on image quality
    (estimated from feature norm), making it robust to degraded/low-quality
    inputs — directly relevant to COCOV's operational deployment framing.

    Attributes
    ----------
    name : 'adaface'
    img_size : 112
    embed_dim : 512
    Training data : MS1MV3
    Loss : AdaFace (Kim et al., CVPR 2022)
    Backbone : IResNet101
    """

    WEIGHTS_DIR = "/opt/code/CVLface/pretrained/adaface"

    name = "adaface"
    embed_dim = 512
    img_size = 112

    def _build_model(self) -> None:
        import sys, subprocess, tempfile, pickle

        weights_dir = os.environ.get("ADAFACE_WEIGHTS_DIR", self.WEIGHTS_DIR)

        if not os.path.isdir(weights_dir):
            raise FileNotFoundError(
                f"AdaFace weights directory not found: {weights_dir}\n"
                "Download from HuggingFace: minchul/cvlface_adaface_ir101_ms1mv3"
            )

        # Load AdaFace in a subprocess to isolate sys.path/sys.modules manipulation
        # from the main process (prevents CUDA state corruption).
        loader_script = f"""
import sys, os, torch
from types import SimpleNamespace
import yaml, pickle

weights_dir = {repr(weights_dir)}
os.chdir(weights_dir)
sys.path.insert(0, weights_dir)

import yaml as _yaml
from models import get_model as _get_model

conf_path = os.path.join(weights_dir, "pretrained_model", "model.yaml")
raw = _yaml.safe_load(open(conf_path))
conf = SimpleNamespace(**raw)
model = _get_model(conf)

ckpt_path = os.path.join(weights_dir, "pretrained_model", "model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt)

# Save state dict so parent can reload
state = model.state_dict()
with open(sys.argv[1], "wb") as f:
    pickle.dump(state, f)
print("ADAFACE_LOAD_OK")
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as script_f:
            script_f.write(loader_script)
            script_path = script_f.name

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as pkl_f:
            pkl_path = pkl_f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path, pkl_path],
                capture_output=True, text=True, timeout=300
            )
            if "ADAFACE_LOAD_OK" not in result.stdout:
                raise RuntimeError(
                    f"AdaFace subprocess loader failed:\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            # Now load model architecture in main process and restore state
            import sys as _sys
            weights_dir_path = weights_dir

            # Temporarily add weights dir to path just for architecture import
            _sys.path.insert(0, weights_dir_path)
            try:
                import importlib.util as _ilu
                # Load models package from CVLFace path specifically
                _spec = _ilu.spec_from_file_location(
                    "_cvlface_models",
                    os.path.join(weights_dir_path, "models", "__init__.py"),
                    submodule_search_locations=[
                        os.path.join(weights_dir_path, "models")
                    ]
                )
            finally:
                _sys.path.pop(0)

            # Simpler: use the subprocess-saved state dict with a fresh model load
            # Build model architecture inline using iresnet directly
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(
                "_cvlface_iresnet",
                os.path.join(weights_dir, "models", "iresnet", "model.py")
            )
            _iresnet_mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_iresnet_mod)
            model = _iresnet_mod.IR_101(input_size=(112, 112))

            import pickle as _pickle
            with open(pkl_path, 'rb') as f:
                state_dict = _pickle.load(f)
            # CVLFace wraps the backbone in a 'net' attribute —
            # strip the 'net.' prefix to match IR_101's parameter names
            state_dict = {
                k[len('net.'):] if k.startswith('net.') else k: v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)

        finally:
            import os as _os
            for p in [script_path, pkl_path]:
                try:
                    _os.unlink(p)
                except Exception:
                    pass

        self.model = model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        logger.info(
            f"AdaFaceEncoder loaded on {self.device} "
            f"(img_size={self.img_size}, embed_dim={self.embed_dim})"
        )

    def _build_transform(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
        ])

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor.to(self.device))

    @property
    def info(self) -> dict:
        d = super().info
        d.update({"pretrained": "CVLFace/cvlface_adaface_ir101_ms1mv3",
                   "training_data": "MS1MV3",
                   "loss": "AdaFace (Kim et al., CVPR 2022)",
                   "backbone": "IResNet101"})
        return d


# ---------------------------------------------------------------------------
# Encoder 5: ArcFace / ViT-B/16  (Vision Transformer)
# ---------------------------------------------------------------------------

class ViTArcFaceEncoder(BaseEncoder):
    """
    Vision Transformer ViT-B/16 trained with ArcFace loss on MS1MV3.

    Represents the transformer-based backbone family. ViT models apply global
    self-attention across image patches rather than local convolutional filters,
    producing qualitatively different embedding geometry. Including this encoder
    tests whether COCOV's unit-sphere cosine similarity assumptions hold for
    non-CNN features, and whether the ranking of methods over baselines is
    preserved across fundamentally different representational families.

    Weight acquisition — three options (tried in order):
    ---------------------------------------------------
    Option A (preferred): InsightFace model zoo ONNX file.
        The loader checks ~/.insightface/models/ for any ViT ONNX model.
        Set VITARCFACE_WEIGHTS=/path/to/vit.onnx to point to a specific file.

    Option B: PyTorch checkpoint (set VITARCFACE_WEIGHTS=/path/to/model.pt).
        Requires timm: pip install timm
        Expected: standard ViT-B/16 with a 512-d head.

    Option C: HuggingFace / timm ImageNet weights (fallback, testing only).
        This is NOT the ArcFace-trained model. A loud warning is logged.
        Replace with Option A or B before running any real experiments.

    Attributes
    ----------
    name : 'vitb_arcface'
    img_size : 112
    embed_dim : 512
    Training data : MS1MV3
    Loss : ArcFace
    Architecture : ViT-B/16 (Dosovitskiy et al., ICLR 2021)
    """

    name = "vitb_arcface"
    embed_dim = 512
    img_size = 112

    def _build_model(self) -> None:
        weights_path = os.environ.get("VITARCFACE_WEIGHTS", None)

        # Option A: explicit ONNX path via env var
        if weights_path and weights_path.endswith(".onnx"):
            self._load_onnx(weights_path)
            return

        # Option B: explicit PyTorch checkpoint via env var
        if weights_path and weights_path.endswith((".pt", ".pth")):
            self._load_pytorch(weights_path)
            return

        # Option C: scan InsightFace model zoo for any ViT ONNX
        try:
            self._load_insightface_vit()
            return
        except Exception as exc:
            logger.warning(f"InsightFace ViT scan: {exc}")

        # Option D: timm ImageNet weights — pipeline testing only
        logger.warning(
            "ViTArcFaceEncoder: no ArcFace-trained ViT weights found. "
            "Loading ImageNet-pretrained ViT-B/16 from timm for pipeline "
            "testing only. Set VITARCFACE_WEIGHTS before running experiments."
        )
        self._load_timm_vit()

    def _load_insightface_vit(self) -> None:
        from insightface.model_zoo import get_model
        base = os.path.expanduser("~/.insightface/models")
        candidates = [
            os.path.join(base, d, f)
            for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
            for f in os.listdir(os.path.join(base, d))
            if "vit" in f.lower() and f.endswith(".onnx")
        ] if os.path.exists(base) else []

        if not candidates:
            raise FileNotFoundError(
                "No ViT ONNX model found in ~/.insightface/models/"
            )

        rec_path = candidates[0]
        self.rec_model = get_model(
            rec_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.rec_model.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1
        )
        self._backend = "onnx_insightface"
        logger.info(f"ViT-B/16 loaded from InsightFace: {rec_path}")

    def _load_onnx(self, path: str) -> None:
        from insightface.model_zoo import get_model
        self.rec_model = get_model(
            path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.rec_model.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1
        )
        self._backend = "onnx"
        logger.info(f"ViT-B/16 loaded from ONNX: {path}")

    def _load_pytorch(self, path: str) -> None:
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for PyTorch ViT. "
                "Install: pip install timm"
            ) from exc
        self.model = timm.create_model(
            "vit_base_patch16_112", pretrained=False, num_classes=512
        )
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self._backend = "pytorch_timm"
        logger.info(f"ViT-B/16 loaded from PyTorch checkpoint: {path}")

    def _load_timm_vit(self) -> None:
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required. Install: pip install timm"
            ) from exc
        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=512
        )
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self._backend = "pytorch_timm_imagenet_fallback"

    def _build_transform(self) -> None:
        # PyTorch path uses standard transform; ONNX path bypasses it.
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
        ])

    def _pil_to_bgr(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        return arr[:, :, ::-1].copy()

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Only reached for PyTorch (timm) backend.
        return self.model(tensor)

    def encode(self, image: Image.Image) -> torch.Tensor:
        if hasattr(self, "rec_model"):
            # ONNX path — pass BGR list; get_feat handles resize internally
            bgr = self._pil_to_bgr(image)
            feat = self.rec_model.get_feat([bgr])
            emb = torch.from_numpy(feat.squeeze().astype(np.float32))
            return torch.nn.functional.normalize(
                emb.unsqueeze(0), p=2, dim=1).squeeze(0)
        else:
            # PyTorch (timm) path — use standard tensor pipeline
            return super().encode(image)

    def encode_path(self, image_path) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.encode(Image.open(path))

    def encode_batch(self, image_paths, batch_size: int = 64) -> torch.Tensor:
        if hasattr(self, "rec_model"):
            # ONNX path — image-by-image via get_feat
            all_embeddings = []
            for path in image_paths:
                try:
                    all_embeddings.append(self.encode_path(path))
                except Exception as exc:
                    logger.warning(f"Skipping {path}: {exc}")
            if not all_embeddings:
                raise RuntimeError("No embeddings extracted.")
            return torch.stack(all_embeddings, dim=0)
        else:
            # PyTorch path — use batched tensor pipeline from base class
            return super().encode_batch(image_paths, batch_size)

    @property
    def info(self) -> dict:
        d = super().info
        d.update({"pretrained": "MS1MV3_ArcFace",
                   "training_data": "MS1MV3",
                   "loss": "ArcFace",
                   "backend": getattr(self, "_backend", "unknown")})
        return d


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: FaceEncoder was the original class name used throughout the codebase.
#: All existing imports ``from models.encoder import FaceEncoder`` continue
#: to work without any modification.
FaceEncoder = FaceNetEncoder


# ---------------------------------------------------------------------------
# Encoder 6: MobileFaceNet  (InsightFace buffalo_s — w600k_mbf.onnx)
# ---------------------------------------------------------------------------

class MobileFaceNetEncoder(BaseEncoder):
    """
    MobileFaceNet trained with ArcFace loss on WebFace600K.

    Uses InsightFace's buffalo_s model pack (w600k_mbf.onnx). MobileFaceNet
    is a lightweight depthwise-separable CNN designed for edge deployment.
    Its smaller capacity and lower embedding ceiling make it a natural test
    of whether COCOV's selective memory management provides proportionally
    greater benefit when the underlying encoder is weaker — directly
    supporting the practical-viability framing in the thesis.

    The model is loaded via the same InsightFace ONNX path as the IResNet
    encoders; the PIL-direct encode() interface is inherited unchanged.

    Installation
    ------------
        pip install insightface onnxruntime-gpu
        # buffalo_s downloads automatically (~122 MB) on first use

    Attributes
    ----------
    name : 'mobilefacenet'
    img_size : 112
    embed_dim : 512
    Training data : WebFace600K
    Loss : ArcFace
    Pack : buffalo_s  (w600k_mbf.onnx)
    """

    name = "mobilefacenet"
    embed_dim = 512
    img_size = 112

    def _build_model(self) -> None:
        try:
            from insightface.model_zoo import get_model
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required for MobileFaceNetEncoder. "
                "Install: pip install insightface onnxruntime-gpu"
            ) from exc

        model_dir = os.path.expanduser("~/.insightface/models/buffalo_s")
        rec_path = os.path.join(model_dir, "w600k_mbf.onnx")

        if not os.path.exists(rec_path):
            logger.info("Downloading buffalo_s model pack via InsightFace...")
            app = FaceAnalysis(
                name="buffalo_s",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

        self.rec_model = get_model(
            rec_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.rec_model.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1
        )

    def _build_transform(self) -> None:
        self.transform = transforms.Resize((self.img_size, self.img_size))

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use encode() or encode_batch() directly.")

    def _pil_to_bgr(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        return arr[:, :, ::-1].copy()

    def encode(self, image: Image.Image) -> torch.Tensor:
        bgr = self._pil_to_bgr(image)
        feat = self.rec_model.get_feat([bgr])
        emb = torch.from_numpy(feat.squeeze().astype(np.float32))
        return torch.nn.functional.normalize(emb.unsqueeze(0), p=2, dim=1).squeeze(0)

    def encode_path(self, image_path) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.encode(Image.open(path))

    def encode_batch(self, image_paths, batch_size: int = 64) -> torch.Tensor:
        all_embeddings = []
        for path in image_paths:
            try:
                all_embeddings.append(self.encode_path(path))
            except Exception as exc:
                logger.warning(f"Skipping {path}: {exc}")
        if not all_embeddings:
            raise RuntimeError("No embeddings extracted.")
        return torch.stack(all_embeddings, dim=0)

    @property
    def info(self) -> dict:
        d = super().info
        d.update({"pretrained": "buffalo_s",
                   "training_data": "WebFace600K",
                   "loss": "ArcFace",
                   "backbone": "MobileFaceNet"})
        return d


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseEncoder]] = {
    "facenet":         FaceNetEncoder,
    "arcface_r50":     ArcFaceR50Encoder,
    "arcface_r100":    ArcFaceR100Encoder,
    "adaface":         AdaFaceEncoder,
    "vitb_arcface":    ViTArcFaceEncoder,
    "mobilefacenet":   MobileFaceNetEncoder,
}


def get_encoder(name: str, device: str = "cuda") -> BaseEncoder:
    """
    Instantiate and return an encoder by its registered name.

    This is the single point of contact between the configuration layer
    (config.yaml encoder.name) and the concrete encoder classes. The
    experiment runner, calibration module, and extraction scripts all
    call this function; they never instantiate encoder classes directly.

    Parameters
    ----------
    name : str
        One of: 'facenet', 'arcface_r50', 'arcface_r100',
                'adaface', 'vitb_arcface'
    device : str
        'cuda' or 'cpu'. Falls back to CPU if CUDA is unavailable.

    Returns
    -------
    BaseEncoder
        Encoder instance with frozen weights, ready for encode() calls.

    Raises
    ------
    ValueError
        If name is not in the registry.

    Examples
    --------
    >>> enc = get_encoder('arcface_r100', device='cuda')
    >>> z = enc.encode(pil_image)        # torch.Tensor (512,) L2-normalised
    >>> Z = enc.encode_batch(paths)      # torch.Tensor (N, 512)
    >>> print(enc.info)
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Available encoders: {list_encoders()}"
        )
    return _REGISTRY[name](device=device)


def list_encoders() -> list[str]:
    """Return all registered encoder names in alphabetical order."""
    return sorted(_REGISTRY.keys())
