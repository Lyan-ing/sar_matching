"""Frozen SAR architecture and exact formal native-coordinate extraction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ._vendor.superpoint import SuperPoint, simple_nms, sample_descriptors


class Structure(nn.Module):
    def __init__(self):
        super().__init__()
        for i, sigma in enumerate([1., 2., 4.]):
            r = int(3 * sigma)
            x = torch.arange(-r, r + 1, dtype=torch.float32)
            g = torch.exp(-x * x / (2 * sigma * sigma)); g /= g.sum()
            self.register_buffer(f'g{i}', (g[:, None] * g[None])[None, None])
        self.register_buffer('sx', torch.tensor([[-3., 0, 3], [-10, 0, 10], [-3, 0, 3]])[None, None] / 32)
        self.register_buffer('sy', self.sx.transpose(-1, -2).clone())
        self.encode = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 256, 1))

    def responses(self, image):
        x = torch.log1p(9 * image) / math.log(10); rs = []
        for i in range(3):
            g = getattr(self, f'g{i}'); r = g.shape[-1] // 2
            y = F.conv2d(F.pad(x, (r, r, r, r), mode='replicate'), g)
            gx = F.conv2d(F.pad(y, (1, 1, 1, 1), mode='replicate'), self.sx)
            gy = F.conv2d(F.pad(y, (1, 1, 1, 1), mode='replicate'), self.sy)
            rs.append((torch.sqrt(gx.square() + gy.square() + 1e-12) - 1e-6).clamp_min(0))
        return torch.cat(rs, 1)


def score_map(logits):
    s = logits.softmax(1)[:, :64].permute(0, 2, 3, 1); b, h, w, _ = s.shape
    return s.reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)


class SARModel(nn.Module):
    def __init__(self, structure=True):
        super().__init__()
        self.sp = SuperPoint(max_num_keypoints=1024, detection_threshold=.0005)
        if structure:
            self.structure = Structure()

    def shared(self, image):
        s = self.sp
        x = s.relu(s.conv1a(image)); x = s.relu(s.conv1b(x)); x = s.pool(x)
        x = s.relu(s.conv2a(x)); x = s.relu(s.conv2b(x)); x = s.pool(x)
        x = s.relu(s.conv3a(x)); x = s.relu(s.conv3b(x)); x = s.pool(x)
        return s.relu(s.conv4b(s.relu(s.conv4a(x))))

    @torch.no_grad()
    def extract(self, image, budget=1024):
        x = self.shared(image); s = self.sp
        logits = s.convPb(s.relu(s.convPa(x)))
        d = s.convDa(x)
        if hasattr(self, 'structure'):
            d = d + self.structure.encode(F.avg_pool2d(self.structure.responses(image), 8, 8))
        dense = F.normalize(s.convDb(s.relu(d)).float(), dim=1)
        scores = simple_nms(score_map(logits), 4)
        scores[:, :4] = -1; scores[:, -4:] = -1; scores[:, :, :4] = -1; scores[:, :, -4:] = -1
        _, y, x = torch.where(scores > .0005)
        v = scores[0, y, x]
        ids = torch.argsort(v, descending=True, stable=True)[:budget]
        k = torch.stack([x[ids], y[ids]], 1).float()
        # Empty descriptor sampling is valid in torch and preserves the native interface.
        desc = sample_descriptors(k[None], dense, 8).transpose(1, 2) if len(k) else dense.new_empty((1, 0, 256))
        return dict(keypoints=k[None], keypoint_scores=v[ids][None], descriptors=desc,
                    image_size=torch.tensor([[image.shape[-1], image.shape[-2]]], device=image.device))
