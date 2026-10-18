"""Inference accepts two image arrays only; annotations are outside this module."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from .locked import CONFIG_FILE, CONFIG_SHA256
from ._vendor.lightglue import LightGlue
from .matchers import Prediction
from .estimate import estimate, perspective
from .reliability import geometry_diagnostics, gate, spatial

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified_file(record):
    path = ROOT / record['file']
    if not path.is_file():
        raise FileNotFoundError(f'Required weights missing: {path}. No fallback is permitted.')
    actual = sha256(path)
    if actual != record['sha256']:
        raise RuntimeError(f'Weight SHA256 mismatch: {path.name}: {actual}')
    return path


def load_config():
    path = ROOT / CONFIG_FILE
    if sha256(path) != CONFIG_SHA256:
        raise RuntimeError('Formal configuration changed: refusing to run a different protocol.')
    return json.loads(path.read_text())


def read_image(path):
    path = Path(path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f'Cannot read image: {path}')
    if raw.dtype != np.uint8:
        raise ValueError(f'Unsupported {raw.dtype} input: formal protocol requires 8-bit PNG display DN; no automatic radiometric conversion.')
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot convert to grayscale: {path}')
    return image


def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(clean(data), f, ensure_ascii=False, indent=2, allow_nan=False)


class Matcher:
    def __init__(self, model='final', device='cpu', budget=1024):
        if model not in ('final', 'structure', 'pretrained'):
            raise ValueError(model)
        if budget not in (128, 1024):
            raise ValueError('Only the existing formal 1024 and equal-budget 128 protocols are supported.')
        self.config = load_config(); self.device = torch.device(device); self.budget = budget; self.name = model
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Use cpu or cuda:N.')
        if self.device.type == 'cuda':
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Check driver/PyTorch or explicitly use --device cpu.')
            torch.cuda.set_device(self.device)
        seed = self.config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(2); cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        p = verified_file(self.config['weights'][model])
        lg_path = verified_file(self.config['weights']['lightglue'])
        from .model import SARModel
        self.net = SARModel(structure=model != 'pretrained')
        self.net.load_state_dict(torch.load(p, map_location='cpu', weights_only=True), strict=True)
        self.net.eval().requires_grad_(False).to(self.device)
        # features=None disables upstream download; identical superpoint dimensions/config.
        self.lg = LightGlue(**self.config['lightglue'])
        sd = torch.load(lg_path, map_location='cpu', weights_only=True)
        for i in range(9):
            for old, new in [(f'self_attn.{i}', f'transformers.{i}.self_attn'), (f'cross_attn.{i}', f'transformers.{i}.cross_attn')]:
                sd = {k.replace(old, new): v for k, v in sd.items()}
        # The upstream strict=False loader ignores these nine obsolete scalar parameters.
        legacy = {f'log_assignment.{i}.r' for i in range(9)}
        if not legacy.issubset(sd): raise RuntimeError('Unexpected LightGlue weight version: legacy signature missing.')
        for key in legacy: sd.pop(key)
        # This deterministic non-learned buffer is absent from upstream v0.1 weights.
        sd.setdefault('confidence_thresholds', self.lg.confidence_thresholds.clone())
        self.lg.load_state_dict(sd, strict=True)
        self.lg.eval().requires_grad_(False).to(self.device)

    def sync(self):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def predict(self, a, b):
        for image in (a, b):
            if image.ndim != 2 or image.dtype != np.uint8: raise ValueError('Expected 2D uint8 grayscale image.')
            if min(image.shape) < 64 or max(image.shape) > 640: raise ValueError('Formal native input bounds: each side 64..640; no resize.')
        if a.shape != b.shape: raise ValueError('Formal geometry protocol requires equal image dimensions; no automatic resampling.')
        if self.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(self.device)
        self.sync(); total = time.perf_counter()
        tensors = [torch.tensor(im, device=self.device, dtype=torch.float32)[None, None] / 255 for im in (a, b)]
        fs = [self.net.extract(t, self.budget) for t in tensors]
        self.sync(); t1 = time.perf_counter()
        if min(f['keypoints'].shape[1] for f in fs) < 2:
            ij = np.empty((0, 2), np.int64); conf = np.empty(0, np.float32)
        else:
            out = self.lg(dict(image0=fs[0], image1=fs[1]))
            ij = out['matches'][0].cpu().numpy(); conf = out['scores'][0].cpu().numpy()
        self.sync(); t2 = time.perf_counter()
        ks = [f['keypoints'][0].cpu().numpy() for f in fs]
        p = Prediction('SAR-SP+LG', ks[0][ij[:, 0]], ks[1][ij[:, 1]], n_keypoints0=len(ks[0]), n_keypoints1=len(ks[1]), keypoints0=ks[0], keypoints1=ks[1])
        e = estimate(p, a.shape, model='homography', seed=42)
        diag = geometry_diagnostics(e, p, a.shape, b.shape, 42) if e.candidate_H is not None else {}
        e = gate(e, diag, 'full'); t3 = time.perf_counter()
        mask = e.inlier_mask if e.inlier_mask is not None else np.zeros(e.n_matches, bool)
        H = e.candidate_H
        failure = 'none' if e.accepted else ('no_keypoints' if min(map(len, ks)) == 0 else 'matches_insufficient' if e.reason == 'too_few_matches' else 'inliers_insufficient' if e.reason == 'few_inliers' else 'geometry_estimation_failed' if e.reason == 'estimation_failed' else 'geometry_rejected')
        rec = dict(accepted=bool(e.accepted), reason=e.reason, failure_type=failure, n_keypoints_A=len(ks[0]), n_keypoints_B=len(ks[1]), n_matches=e.n_matches, n_inliers=e.n_inliers, inlier_ratio=e.inlier_ratio, inlier_fit_rmse_px=e.inlier_rmse,
                   H_AB=H, H_direction='image_A_to_image_B_native_xy', H_is_accepted=bool(e.accepted and H is not None),
                   inlier_support_A=spatial(p.points0[mask], a.shape), inlier_support_B=spatial(p.points1[mask], b.shape),
                   candidate_support_A=spatial(p.points0, a.shape), candidate_support_B=spatial(p.points1, b.shape),
                   feature_seconds=t1-total, matching_seconds=t2-t1, geometry_seconds=t3-t2, total_inference_seconds=t3-total,
                   peak_allocated_MiB=torch.cuda.max_memory_allocated(self.device)/2**20 if self.device.type=='cuda' else None,
                   geometry_diagnostics=diag, independent_GT_available=False, ground_truth_used=False, geographic_prior_used=False)
        if H is not None:
            q = np.array([[0, 0], [a.shape[1]-1, 0], [a.shape[1]-1, a.shape[0]-1], [0, a.shape[0]-1]], float)
            disp = np.linalg.norm(perspective(H, q)-q, axis=1)
            normH = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None
            rec.update(H_identity_frobenius=float(np.linalg.norm(normH-np.eye(3))) if normH is not None else None, corner_mean_displacement_px=float(np.mean(disp)), corner_displacements_px=disp)
        else:
            rec.update(H_identity_frobenius=None, corner_mean_displacement_px=None, corner_displacements_px=None)
        arrays = dict(keypoints_A=ks[0], keypoints_B=ks[1], keypoint_scores_A=fs[0]['keypoint_scores'][0].cpu().numpy(), keypoint_scores_B=fs[1]['keypoint_scores'][0].cpu().numpy(), matches_A=p.points0, matches_B=p.points1, match_indices=ij, confidence=conf, inliers=mask, H_AB=H if H is not None else np.empty((0, 3)), accepted=np.array(e.accepted))
        return clean(rec), arrays
