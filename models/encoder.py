"""
encoder.py
----------
FaceNet encoder wrapper using InceptionResnetV1 pretrained on VGGFace2.

Responsibilities:
    - Load and manage the pretrained encoder
    - Preprocess images to the format expected by the encoder
    - Extract l2-normalised 512-dimensional embeddings
    - Run on GPU if available, CPU otherwise

The encoder is fixed throughout all experiments. No fine-tuning
or weight updates are performed at any stage. All methods in the
evaluation operate on embeddings produced by this encoder.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FaceEncoder:
    """
    Wrapper around InceptionResnetV1 for fixed face embedding extraction.

    Images are expected to be pre-aligned and cropped face images.
    Since VGGFace2-HQ images are already aligned at 512x512,
    no detection or alignment is performed here. Images are simply
    resized to 160x160 and normalised before encoding.

    Attributes
    ----------
    device : torch.device
        Device on which the encoder runs (cuda or cpu).
    model : InceptionResnetV1
        Pretrained encoder in evaluation mode.
    transform : transforms.Compose
        Preprocessing pipeline applied to each image.
    embedding_dim : int
        Dimensionality of the output embedding (512).
    """

    def __init__(self, pretrained: str = "vggface2",
                 device: str = "cuda"):
        """
        Initialise the encoder.

        Parameters
        ----------
        pretrained : str
            Pretrained weights to load. Default is 'vggface2'.
        device : str
            Device to run inference on. 'cuda' or 'cpu'.
            Falls back to cpu if cuda is unavailable.
        """
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available. "
                "Falling back to CPU."
            )

        self.embedding_dim = 512

        # Load pretrained model in evaluation mode
        # Weights are downloaded automatically on first run
        self.model = InceptionResnetV1(
            pretrained=pretrained
        ).eval().to(self.device)

        # Freeze all parameters --- encoder is never updated
        for param in self.model.parameters():
            param.requires_grad = False

        # Preprocessing pipeline
        # Input: PIL Image or path
        # Output: normalised tensor in [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        logger.info(
            f"FaceEncoder loaded on {self.device} "
            f"with pretrained='{pretrained}'"
        )

    def encode(self, image: Image.Image) -> torch.Tensor:
        """
        Extract a unit-normalised embedding from a single image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input face image. Expected to be a pre-aligned
            face crop in RGB format.

        Returns
        -------
        torch.Tensor
            Unit-normalised embedding of shape (512,) on CPU.
            Norm is guaranteed to be 1.0.
        """
        tensor = self.transform(image.convert("RGB"))
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)

        # Return as 1D tensor on CPU for storage and comparison
        return embedding.squeeze(0).cpu()

    def encode_path(self, image_path: str | Path) -> torch.Tensor:
        """
        Extract a unit-normalised embedding from an image file path.

        Parameters
        ----------
        image_path : str or Path
            Path to a JPEG or PNG face image.

        Returns
        -------
        torch.Tensor
            Unit-normalised embedding of shape (512,) on CPU.

        Raises
        ------
        FileNotFoundError
            If the image path does not exist.
        RuntimeError
            If the image cannot be opened or processed.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to open image {image_path}: {e}"
            )
        return self.encode(image)

    def encode_batch(self,
                     image_paths: list[str | Path],
                     batch_size: int = 64) -> torch.Tensor:
        """
        Extract embeddings for a list of image paths in batches.

        Batching improves GPU utilisation during the embedding
        extraction phase. All embeddings are returned on CPU.

        Parameters
        ----------
        image_paths : list of str or Path
            List of paths to face images.
        batch_size : int
            Number of images processed per GPU batch.
            Default is 64. Reduce if GPU memory is exceeded.

        Returns
        -------
        torch.Tensor
            Unit-normalised embeddings of shape (N, 512) on CPU,
            where N is the number of input images.
        """
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    tensors.append(self.transform(image))
                except Exception as e:
                    logger.warning(
                        f"Skipping {path}: {e}"
                    )
                    continue

            if not tensors:
                continue

            batch = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                embeddings = self.model(batch)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

        if not all_embeddings:
            raise RuntimeError(
                "No embeddings extracted. "
                "Check that image paths are valid."
            )

        return torch.cat(all_embeddings, dim=0)

    @property
    def info(self) -> dict:
        """
        Return encoder configuration as a dictionary.
        Used for logging and reproducibility records.
        """
        return {
            "model": "InceptionResnetV1",
            "pretrained": "vggface2",
            "embedding_dim": self.embedding_dim,
            "device": str(self.device),
            "trainable_params": 0
        }
