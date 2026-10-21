"""Face detection + embedding ensemble.

Detection: SCRFD-10G from the buffalo_l pack (InsightFace 2.x FaceAnalysis),
passed explicit input sizes (e.g. (1280, 1280) or ((640, 640), (1280, 1280))
to merge scales in one call).

Embedding models (aligned 112x112 ArcFace crops):
  w600k  - InsightFace buffalo_l w600k_r50 (ResNet50/WebFace600K), BGR, mean/std 127.5
  adaface- AdaFace IR-101 ONNX, BGR, mean/std 127.5 (channel order follows the
           ONNX export used here; validate against your export if you swap it)
  lvface - LVFace-B Glint360K ONNX, RGB, mean/std 127.5  ((x-127.5)/127.5)

With flip TTA the embedding is the L2-normalized average of the crop and its
horizontal mirror. Stored vectors are L2-normalized, so cosine == dot product.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, message=".*estimate.*deprecated.*")

MODEL_NAMES = ("w600k", "adaface", "lvface")


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class _OnnxEmbedder:
    """Generic 112x112 ONNX re-encoder; returns raw (unnormalized) embedding."""

    def __init__(self, name: str, model_path: str, rgb: bool, threads: int = 4):
        import onnxruntime
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{name}: model not found at {model_path}")
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = threads
        self.name = name
        self.sess = onnxruntime.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.rgb = rgb

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        img = cv2.resize(crop_bgr, (112, 112))
        x = img.astype(np.float32)
        if self.rgb:
            x = x[..., ::-1]
        x = (x - 127.5) / 127.5
        blob = np.expand_dims(np.transpose(x, (2, 0, 1)), 0)
        out = self.sess.run(None, {self.input_name: blob})[0]
        return out.flatten()


class FaceModels:
    def __init__(
        self,
        insightface_root: str,
        adaface_path: str,
        lvface_path: str,
        enabled: tuple[str, ...] = ("w600k",),
        det_sizes: tuple[int, ...] = (1280,),
        flip: bool = False,
        threads: int = 4,
    ):
        bad = [m for m in enabled if m not in MODEL_NAMES]
        if bad:
            raise ValueError(f"unknown models: {bad}; choose from {MODEL_NAMES}")
        self.enabled = tuple(enabled)
        self.det_sizes = tuple(int(s) for s in det_sizes)
        self.det_config = "+".join(str(s) for s in self.det_sizes)
        self.flip = flip

        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=insightface_root,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1)

        self._embedders: dict[str, object] = {}
        if "w600k" in self.enabled:
            self._embedders["w600k"] = ("insightface", self.app.models["recognition"])
        if "adaface" in self.enabled:
            self._embedders["adaface"] = ("onnx", _OnnxEmbedder("adaface", adaface_path, rgb=False, threads=threads))
        if "lvface" in self.enabled:
            self._embedders["lvface"] = ("onnx", _OnnxEmbedder("lvface", lvface_path, rgb=True, threads=threads))

    # ------------------------------------------------------------------
    def detect(self, bgr: np.ndarray):
        """Run SCRFD on the given BGR image; returns (boxes Nx5, kpss Nx5x2)."""
        sizes = [(s, s) for s in self.det_sizes]
        return self.app.det_model.detect(bgr, input_size=sizes)

    def embed(self, model: str, crop_bgr: np.ndarray) -> np.ndarray:
        """L2-normalized embedding, averaged with its mirror if flip TTA is on."""
        kind, impl = self._embedders[model]
        if kind == "insightface":
            e = impl.get_feat(crop_bgr).flatten()
            if self.flip:
                mirror = np.ascontiguousarray(crop_bgr[:, ::-1])
                e = _l2(e) + _l2(impl.get_feat(mirror).flatten())
            return _l2(e).astype(np.float32)
        e = impl.embed(crop_bgr)
        if self.flip:
            mirror = np.ascontiguousarray(crop_bgr[:, ::-1])
            e = _l2(e) + _l2(impl.embed(mirror))
        return _l2(e).astype(np.float32)


def aligned_crop(bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """ArcFace-template warp from 5 landmarks (same as InsightFace norm_crop)."""
    from insightface.utils import face_align
    return face_align.norm_crop(bgr, kps, image_size=size)
