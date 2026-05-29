"""
Phasor dataset for EsViT self-supervised pre-training.

전제: preprocess_phasor.py 로 MAT → npy 변환이 완료된 상태여야 합니다.

입력 디렉토리 구조:
    data_path/
        PALA_InVivoRatBrain_001_phasor.npy   shape: (2, H, W, T), float32
        PALA_InVivoRatBrain_002_phasor.npy     [0] = magnitude, [1] = phase
        ...
        stats.json   {"mag_mean", "mag_std", "phase_mean", "phase_std"}

출력 (한 샘플):
    (20, H_pad, W_pad) 정규화된 float32 텐서
    채널 순서: [mag_0, phase_0, mag_1, phase_1, ..., mag_9, phase_9]
"""

import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


# ── 샘플링 상수 ────────────────────────────────────────────────────────────────
N_FRAMES     = 10   # 샘플 하나에 들어갈 프레임 수
FRAME_STRIDE = 50   # 선택 프레임 사이 간격
START_STRIDE = 5    # 시작 프레임 후보 사이 간격
# ─────────────────────────────────────────────────────────────────────────────


def pad_to_multiple(tensor, multiple=4):
    """
    (C, H, W) 텐서의 H, W를 multiple의 배수로 zero-padding합니다.
    patch_size=4, window_size=4의 배수여야 모델에 정상 입력됩니다.
    """
    _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor


class PhasorDataset(Dataset):
    """
    전처리된 Phasor npy 파일을 읽어 EsViT 학습용 샘플을 반환합니다.

    __getitem__ 처리 순서:
        mmap으로 프레임 슬라이스 읽기
        → 20채널 텐서 조립 [mag_0, phase_0, ..., mag_9, phase_9]
        → center crop (옵션, bolus 등 크기가 다른 경우)
        → 4의 배수로 padding
        → 정규화 (stats.json 기반, 짝수채널=magnitude, 홀수채널=phase)
        → transform (멀티크롭 증강)
    """

    def __init__(self, data_path, transform=None, center_crop_hw=None,
                 n_frames=N_FRAMES, frame_stride=FRAME_STRIDE,
                 start_stride=START_STRIDE):
        """
        Args:
            data_path:       *_phasor.npy 와 stats.json 이 있는 디렉토리.
                             문자열 하나 또는 여러 디렉토리의 리스트 모두 가능.
                             예: 'USdataset/phasor/perfusion'
                             예: ['USdataset/phasor/perfusion', 'USdataset/phasor/kidney']
            transform:       DataAugmentationPhasor 인스턴스 (None이면 증강 없음)
            center_crop_hw:  (H, W) center crop 크기.
            n_frames:        샘플당 프레임 수 (기본 10)
            frame_stride:    선택 프레임 간격 (기본 50)
            start_stride:    시작 프레임 후보 간격 (기본 5)
        """
        self.transform    = transform
        self.crop_hw      = center_crop_hw
        self.n_frames     = n_frames
        self.frame_stride = frame_stride
        self.start_stride = start_stride

        # data_path가 문자열이면 리스트로 변환
        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = list(data_path)

        # mmap 열기 + 샘플 인덱스 빌드 + stats 로드
        self.mmap        = []
        self.samples     = []   # (file_idx, start_frame) 목록
        self.file_stats  = []   # file_idx → stats dict (정규화용)

        for data_dir in data_paths:
            # stats.json 로드
            stats_path = os.path.join(data_dir, 'stats.json')
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    dir_stats = json.load(f)
                print(f"정규화 통계값 로드: {stats_path}")
            else:
                dir_stats = None
                print(f"[경고] stats.json 없음: {stats_path}")

            npy_files = sorted(glob.glob(os.path.join(data_dir, '*_phasor.npy')))
            if not npy_files:
                raise FileNotFoundError(
                    f"*_phasor.npy 파일 없음: {data_dir}\n"
                    "먼저 preprocess_phasor.py 를 실행하세요."
                )

            for fpath in npy_files:
                file_idx = len(self.mmap)
                arr = np.load(fpath, mmap_mode='r')   # (2, H, W, T)
                T   = arr.shape[-1]
                self.mmap.append(arr)
                self.file_stats.append(dir_stats)

                last_start = T - (n_frames - 1) * frame_stride - 1
                for s in range(0, last_start + 1, start_stride):
                    self.samples.append((file_idx, s))

        # 하위 호환: self.stats = 첫 번째 디렉토리 통계
        self.stats = self.file_stats[0] if self.file_stats else None

        total = sum(
            len(sorted(glob.glob(os.path.join(d, '*_phasor.npy')))) for d in data_paths
        )
        print(f"총 {total}개 파일, {len(self.samples)}개 샘플 로드 완료")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start_frame = self.samples[idx]
        frame_indices = [start_frame + i * self.frame_stride
                         for i in range(self.n_frames)]

        # mmap에서 필요한 프레임만 읽기
        arr = self.mmap[file_idx]   # (2, H, W, T), 실제 I/O는 아래에서 발생
        # (2, H, W, N_FRAMES) — 필요한 프레임만 인덱싱해 메모리에 올림
        frames = arr[:, :, :, frame_indices]

        # 20채널 텐서 조립: [mag_0, phase_0, mag_1, phase_1, ...]
        channels = []
        for i in range(self.n_frames):
            channels.append(frames[0, :, :, i])   # magnitude
            channels.append(frames[1, :, :, i])   # phase
        tensor = torch.from_numpy(
            np.stack(channels, axis=0).copy()
        ).float()   # (20, H, W)

        # Center crop (bolus: 107×128 → 78×118)
        if self.crop_hw is not None:
            ch, cw = self.crop_hw
            _, H, W = tensor.shape
            h0 = (H - ch) // 2
            w0 = (W - cw) // 2
            tensor = tensor[:, h0:h0 + ch, w0:w0 + cw]

        # 4의 배수로 패딩 (78→80, 118→120)
        tensor = pad_to_multiple(tensor, multiple=4)

        # 정규화: 파일별 stats 사용 (폴더마다 다른 통계값 적용)
        stats = self.file_stats[file_idx]
        if stats is not None:
            tensor[0::2] = (tensor[0::2] - stats['mag_mean'])   \
                           / (stats['mag_std']   + 1e-8)
            tensor[1::2] = (tensor[1::2] - stats['phase_mean']) \
                           / (stats['phase_std'] + 1e-8)

        if self.transform is not None:
            return self.transform(tensor), 0   # label 불필요 (SSL)
        return tensor, 0


# ── 증강 구성요소 ──────────────────────────────────────────────────────────────

class RandomResizedCropTensor:
    """
    (C, H, W) 텐서에서 랜덤 영역을 crop한 뒤 output_size로 리사이즈.
    scale: 원본 면적 대비 crop 비율 범위.
    """
    def __init__(self, output_size, scale=(0.4, 1.0)):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.scale = scale

    def __call__(self, x):
        C, H, W = x.shape
        area = H * W
        for _ in range(10):
            target_area = area * (self.scale[0] +
                          torch.rand(1).item() * (self.scale[1] - self.scale[0]))
            aspect = 0.75 + torch.rand(1).item() * 0.5   # 0.75 ~ 1.25
            crop_h = int(round((target_area / aspect) ** 0.5))
            crop_w = int(round((target_area * aspect) ** 0.5))
            if crop_h <= H and crop_w <= W:
                top  = torch.randint(0, H - crop_h + 1, (1,)).item()
                left = torch.randint(0, W - crop_w + 1, (1,)).item()
                cropped = x[:, top:top + crop_h, left:left + crop_w]
                return F.interpolate(
                    cropped.unsqueeze(0), size=self.output_size,
                    mode='bilinear', align_corners=False,
                ).squeeze(0)
        return F.interpolate(
            x.unsqueeze(0), size=self.output_size,
            mode='bilinear', align_corners=False,
        ).squeeze(0)


class GaussianBlur2d:
    """확률 p로 Gaussian blur 적용. (C, H, W) 텐서 입력."""
    def __init__(self, p=0.5, kernel_sizes=(3, 5, 7)):
        self.p = p
        self.kernel_sizes = kernel_sizes

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            k = self.kernel_sizes[torch.randint(len(self.kernel_sizes), (1,)).item()]
            x = F.avg_pool2d(
                x.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2
            ).squeeze(0)
        return x


class SpeckleNoise:
    """
    초음파 특유의 speckle noise를 시뮬레이션합니다.
    Speckle은 곱셈형 잡음: x_noisy = x * (1 + σ·N(0,1))
    magnitude 채널(짝수)에만 적용합니다.
    phase 채널에는 적용하지 않습니다 (speckle은 진폭 현상).

    sigma_range: 잡음 강도 범위. (0, 0.1)이면 0~10% 수준의 잡음.
    """
    def __init__(self, p=0.5, sigma_range=(0.0, 0.1)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            sigma = self.sigma_range[0] + \
                    torch.rand(1).item() * (self.sigma_range[1] - self.sigma_range[0])
            noise = 1.0 + sigma * torch.randn(x[0::2].shape)
            x = x.clone()
            x[0::2] = x[0::2] * noise
        return x


class RandomRotation5deg:
    """
    -5도 ~ +5도 사이에서 랜덤 회전합니다.
    초음파 영상에서는 탐촉자(probe) 기울기 변화를 시뮬레이션합니다.
    전체 채널에 동일하게 적용합니다.
    """
    def __init__(self, p=0.5, max_angle=5.0):
        self.p = p
        self.max_angle = max_angle

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            angle = (torch.rand(1).item() * 2 - 1) * self.max_angle  # -5 ~ +5
            # TF.rotate: (C, H, W) 텐서 입력 가능 (torchvision >= 0.8)
            x = TF.rotate(x, angle=angle,
                          interpolation=TF.InterpolationMode.BILINEAR,
                          fill=0.0)
        return x


class BrightnessChange:
    """
    magnitude 채널 전체를 균일하게 스케일합니다.
    초음파 기기의 gain(증폭) 설정 변화를 시뮬레이션합니다.
    정규화된 magnitude에 곱셈 적용이므로 상대적 밝기만 변화합니다.

    scale_range: 스케일 범위. (0.7, 1.3)이면 ±30% 밝기 변화.
    """
    def __init__(self, p=0.5, scale_range=(0.7, 1.3)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            scale = self.scale_range[0] + \
                    torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
            x = x.clone()
            x[0::2] = x[0::2] * scale   # magnitude 채널만
        return x


class AttenuationChange:
    """
    초음파 감쇠(attenuation)를 시뮬레이션합니다.
    초음파는 조직을 통과할수록 에너지가 감소하므로
    깊이(H 방향, 위→아래)가 깊어질수록 magnitude가 작아집니다.

    exp(-α × depth) 형태의 감쇠를 magnitude 채널에 적용합니다.
    α가 클수록 감쇠가 강합니다.

    alpha_range: 감쇠 계수 범위. 정규화된 데이터 기준으로
                 0.0~0.5 사이면 아래쪽이 최대 60% 감소.
    """
    def __init__(self, p=0.5, alpha_range=(0.0, 0.5)):
        self.p = p
        self.alpha_range = alpha_range

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            alpha = self.alpha_range[0] + \
                    torch.rand(1).item() * (self.alpha_range[1] - self.alpha_range[0])
            H = x.shape[1]
            # depth: 0(얕은 곳) → 1(깊은 곳), shape (1, H, 1) → 브로드캐스트
            depth = torch.linspace(0, 1, H).view(1, H, 1)
            decay = torch.exp(-alpha * depth)   # (1, H, 1)
            x = x.clone()
            x[0::2] = x[0::2] * decay           # magnitude 채널만
        return x


# ── 멀티크롭 증강 파이프라인 ──────────────────────────────────────────────────

class DataAugmentationPhasor:
    """
    Phasor 텐서용 멀티크롭 증강 파이프라인.

    글로벌 crop × 2: 원본의 40~100% 영역 → (80, 120)으로 리사이즈
    로컬  crop × N: 원본의  5~40% 영역  → (40,  60)으로 리사이즈

    각 crop에 순서대로 적용되는 증강:
        1. RandomResizedCrop   (crop + resize)
        2. GaussianBlur        (흐리기)
        3. SpeckleNoise        (초음파 speckle 잡음, magnitude에만)
        4. RandomRotation5deg  (미세 회전)
        5. BrightnessChange    (gain 변화, magnitude에만)
        6. AttenuationChange   (깊이별 감쇠, magnitude에만)

    제거된 항목 (RGB 전용):
        ColorJitter, RandomGrayscale, Solarization, RandomHorizontalFlip
    """

    GLOBAL_SIZE = (80, 120)
    LOCAL_SIZE  = (40,  60)

    def __init__(self, global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4),
                 local_crops_number=(8,),
                 global_size=None, local_size=None):
        self.local_crops_number = list(local_crops_number)

        g_size = global_size if global_size is not None else self.GLOBAL_SIZE
        l_size = local_size  if local_size  is not None else self.LOCAL_SIZE

        self.global_crop = RandomResizedCropTensor(g_size, scale=global_crops_scale)
        self.local_crop  = RandomResizedCropTensor(l_size, scale=local_crops_scale)

        # 글로벌 crop 1: 강한 blur
        # 글로벌 crop 2: 약한 blur
        # 로컬  crop  : 중간 blur
        self.blur_strong = GaussianBlur2d(p=1.0)
        self.blur_weak   = GaussianBlur2d(p=0.1)
        self.blur_local  = GaussianBlur2d(p=0.5)

        # 공통 적용 증강 (crop 후에 적용)
        self.speckle     = SpeckleNoise(p=0.5,  sigma_range=(0.0, 0.1))
        self.rotation    = RandomRotation5deg(p=0.5, max_angle=5.0)
        self.brightness  = BrightnessChange(p=0.5,   scale_range=(0.7, 1.3))
        self.attenuation = AttenuationChange(p=0.5,  alpha_range=(0.0, 0.5))

    def _apply_common(self, x):
        """crop + blur 이후 공통 증강."""
        x = self.speckle(x)
        x = self.rotation(x)
        x = self.brightness(x)
        x = self.attenuation(x)
        return x

    def __call__(self, x):
        """
        x: (20, H, W) 정규화된 float 텐서

        반환: [global_1, global_2, local_0, ..., local_N-1]
              각 원소 shape: global=(20,80,120), local=(20,40,60)
        """
        g1 = self._apply_common(self.blur_strong(self.global_crop(x)))
        g2 = self._apply_common(self.blur_weak(self.global_crop(x)))
        crops = [g1, g2]

        for n in self.local_crops_number:
            for _ in range(n):
                lc = self._apply_common(self.blur_local(self.local_crop(x)))
                crops.append(lc)
        return crops
