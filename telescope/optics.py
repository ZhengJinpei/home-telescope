"""光学计算：倍率、视场、出瞳、分辨率、极限星等，以及目镜和相机的搭配建议。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

EYE_PUPIL_MM = 7.0  # 暗适应后的人眼瞳孔，年纪大了会小一些
MOON_DIAMETER_DEG = 0.52
SEEING_LIMIT_X = 300  # 城市大气抖动下，超过这个倍率基本只会更糊


@dataclass(frozen=True)
class Telescope:
    aperture_mm: float  # 口径
    focal_length_mm: float  # 焦距

    def __post_init__(self):
        if self.aperture_mm <= 0 or self.focal_length_mm <= 0:
            raise ValueError("口径和焦距必须大于 0")

    @property
    def focal_ratio(self) -> float:
        return self.focal_length_mm / self.aperture_mm

    @property
    def max_useful_magnification(self) -> float:
        """经验值：口径毫米数的 2 倍，再受大气限制。"""
        return min(2.0 * self.aperture_mm, SEEING_LIMIT_X)

    @property
    def min_useful_magnification(self) -> float:
        """低于这个倍率，出瞳比人眼瞳孔还大，多出来的光浪费了。"""
        return self.aperture_mm / EYE_PUPIL_MM

    @property
    def dawes_limit_arcsec(self) -> float:
        """能分开的两颗星的最小角距离（角秒）。"""
        return 116.0 / self.aperture_mm

    @property
    def limiting_magnitude(self) -> float:
        """理想夜空下能看到的最暗恒星（近似）。城市里通常要减 1–2 等。"""
        return 7.7 + 5.0 * math.log10(self.aperture_mm / 10.0)

    @property
    def light_gathering_vs_eye(self) -> float:
        return (self.aperture_mm / EYE_PUPIL_MM) ** 2


@dataclass(frozen=True)
class Eyepiece:
    focal_length_mm: float
    apparent_fov_deg: float = 52.0  # 普通 Plössl 目镜约 50–52°
    barlow: float = 1.0  # 巴洛镜倍数，没有就是 1


@dataclass
class EyepieceView:
    magnification: float
    true_fov_deg: float
    exit_pupil_mm: float
    warnings: list[str] = field(default_factory=list)

    @property
    def moon_fits(self) -> bool:
        return self.true_fov_deg >= MOON_DIAMETER_DEG


def view(scope: Telescope, eyepiece: Eyepiece) -> EyepieceView:
    mag = scope.focal_length_mm * eyepiece.barlow / eyepiece.focal_length_mm
    v = EyepieceView(
        magnification=mag,
        true_fov_deg=eyepiece.apparent_fov_deg / mag,
        exit_pupil_mm=scope.aperture_mm / mag,
    )
    if mag > scope.max_useful_magnification:
        v.warnings.append(
            f"{mag:.0f} 倍超过了这台望远镜的有效上限（约 {scope.max_useful_magnification:.0f} 倍），"
            "画面会又暗又糊。倍率不是越高越好！"
        )
    if v.exit_pupil_mm > EYE_PUPIL_MM:
        v.warnings.append(f"出瞳 {v.exit_pupil_mm:.1f} mm 比人眼瞳孔还大，有一部分光进不了眼睛。")
    if v.exit_pupil_mm < 0.5:
        v.warnings.append(f"出瞳只有 {v.exit_pupil_mm:.2f} mm，眼睛里的小飞蚊会很明显。")
    return v


def suggest_eyepieces(scope: Telescope) -> dict[str, float]:
    """按出瞳推荐三支目镜焦距（mm）：找目标用的低倍、常用中倍、看行星的高倍。"""
    f = scope.focal_ratio
    high_pupil = max(0.7, scope.aperture_mm / scope.max_useful_magnification)
    return {
        # 1.25 寸目镜最长约 32 mm，再长就要换 2 寸接口
        "低倍（找目标、看星团星云）": round(min(f * 5.0, 32.0), 1),
        "中倍（月面、星团）": round(f * 2.0, 1),
        "高倍（行星、双星）": round(f * high_pupil, 1),
    }


@dataclass(frozen=True)
class Camera:
    """接在望远镜上拍照用的相机传感器。"""

    name: str
    pixel_um: float
    width_mm: float
    height_mm: float
    # 插进调焦座后，传感器比抽筒顶端高出多少 mm（直焦拍摄时焦点要落在这里）。None 表示镜头拆不下来
    back_focus_mm: float | None = None
    mass_kg: float = 0.0  # 连转接件

    @property
    def diagonal_mm(self) -> float:
        return math.hypot(self.width_mm, self.height_mm)


# 树莓派 HQ 相机：CS 口法兰距 12.5 mm，加 C-CS 转接环和 1.25 寸 C 口接筒后约 17.5 mm
PI_HQ_CAMERA = Camera("树莓派 HQ 相机 (IMX477)", 1.55, 6.287, 4.712, back_focus_mm=17.5, mass_kg=0.15)
PI_CAMERA_3 = Camera("树莓派相机模块 3 (IMX708)", 1.40, 6.45, 3.63)
# 佳能 EOS 80D：EF 卡口法兰距 44 mm + T 环 11 mm = T 口标准后截距 55 mm，
# 再加 2 寸接筒台阶到 T 口约 8 mm。机身约 730 g，T 环和接筒约 120 g
CANON_80D = Camera("佳能 EOS 80D（APS-C）", 3.72, 22.3, 14.9, back_focus_mm=63.0, mass_kg=0.85)

CAMERAS = {"hq": PI_HQ_CAMERA, "v3": PI_CAMERA_3, "80d": CANON_80D}


@dataclass
class CameraView:
    arcsec_per_pixel: float
    fov_width_deg: float
    fov_height_deg: float
    sampling: str

    @property
    def moon_fits(self) -> bool:
        return min(self.fov_width_deg, self.fov_height_deg) >= MOON_DIAMETER_DEG


def camera_view(scope: Telescope, camera: Camera) -> CameraView:
    scale = 206.265 * camera.pixel_um / scope.focal_length_mm
    w = math.degrees(2 * math.atan(camera.width_mm / (2 * scope.focal_length_mm)))
    h = math.degrees(2 * math.atan(camera.height_mm / (2 * scope.focal_length_mm)))
    # 像素尺度和望远镜分辨率比：每个分辨单元 2 个像素左右比较合适
    ratio = scope.dawes_limit_arcsec / scale
    if ratio < 1.0:
        sampling = "欠采样：像素太大，细节被抹掉了，拍星云星团没问题，拍行星建议加巴洛镜"
    elif ratio > 4.0:
        sampling = "过采样：每个像素分到的光太少，可以去掉巴洛镜或用 2×2 合并像素"
    else:
        sampling = "采样合适"
    return CameraView(scale, w, h, sampling)
