import os
import math
import glob
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class FeatureExtractor:
    """
    CNN(ResNet-18)으로 이미지 임베딩을 추출.
    - 입력: PIL.Image
    - 출력: (D,) numpy vector (L2-normalized)
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = self._build_model_and_transform()
        self.model.eval()

    def _build_model_and_transform(self):
        # 가볍고 빠른 ResNet-18의 풀링 직전 feature를 사용
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 마지막 FC 제거 -> 글로벌풀링 출력 사용
        backbone = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        for p in backbone.parameters():
            p.requires_grad = False

        # ImageNet 표준 mean/std 값 직접 사용
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return backbone, tfm

    @torch.inference_mode()
    def embed(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img).unsqueeze(0).to(self.device)  # (1,3,224,224)
        feat = self.model(x)  # (1,512,1,1)
        feat = feat.view(feat.size(0), -1)  # (1,512)
        v = feat[0].detach().cpu().numpy().astype("float32")
        # L2 정규화 (클러스터링 안정화)
        norm = np.linalg.norm(v) + 1e-12
        return (v / norm)


class ImageClusterSelector:
    """
    디렉토리의 여러 이미지를 임베딩 → KMeans → 최대 클러스터 메도이드 선택.
    """
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.extractor = FeatureExtractor()

    def _list_images(self, directory: str) -> List[str]:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.gif"]
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(directory, e)))
        return sorted(set(paths))

    def _safe_open(self, path: str) -> Optional[Image.Image]:
        try:
            img = Image.open(path).convert("RGB")
            return img
        except (UnidentifiedImageError, OSError):
            return None

    def _auto_k(self, n: int) -> int:
        if self.n_clusters is not None:
            return max(1, min(self.n_clusters, n))
        if n <= 2:
            return 1
        k = int(math.sqrt(n))
        k = max(1, min(k, 8))
        return k

    def select(self, directory: str) -> Tuple[str, List[str]]:
        paths = self._list_images(directory)
        if not paths:
            raise ValueError(f"No image files found in: {directory}")

        images = []
        valid_paths = []
        for p in paths:
            img = self._safe_open(p)
            if img is not None:
                images.append(img)
                valid_paths.append(p)

        if not images:
            raise ValueError(f"Images exist but none could be opened: {directory}")

        feats = np.stack([self.extractor.embed(img) for img in images], axis=0)

        if feats.shape[0] == 1:
            return valid_paths[0], valid_paths

        k = self._auto_k(feats.shape[0])
        if k == 1:
            center = feats.mean(axis=0, keepdims=True)
            idx, _ = pairwise_distances_argmin_min(center, feats, metric="euclidean")
            return valid_paths[int(idx[0])], valid_paths

        km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
        labels = km.fit_predict(feats)

        sizes = np.bincount(labels)
        largest = int(np.argmax(sizes))

        cluster_idx = np.where(labels == largest)[0]
        cluster_feats = feats[cluster_idx]
        centroid = km.cluster_centers_[largest][None, :]
        closest, _ = pairwise_distances_argmin_min(centroid, cluster_feats, metric="euclidean")
        rep_global_idx = cluster_idx[int(closest[0])]

        return valid_paths[rep_global_idx], valid_paths


if __name__ == "__main__":
    sample_dir = os.path.join(os.path.dirname(__file__), "../../app/sample/uploads")

    selector = ImageClusterSelector()
    try:
        rep_path, all_paths = selector.select(sample_dir)
        print(f"\n총 {len(all_paths)}개의 이미지 중 대표 이미지:")
        print(f"대표 이미지 경로: {rep_path}")
        print("\n전체 이미지 목록:")
        for p in all_paths:
            print(f" - {p}")
    except ValueError as e:
        print(f"에러: {e}")
