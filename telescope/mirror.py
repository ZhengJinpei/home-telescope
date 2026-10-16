"""主镜和副镜的外协加工参数：交给光学厂之前要说清楚的尺寸、面形和检验数据。

约定：面形误差（surface）是镜面本身的误差；波面误差（wavefront）是光反射后的误差，
反射镜正入射时波面误差 = 2 × 面形误差。检验波长按氦氖激光干涉仪的 632.8 nm。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TEST_WAVELENGTH_NM = 632.8
VISUAL_WAVELENGTH_NM = 550.0
GLASS_DENSITY_KG_MM3 = 2.5e-6


def sphere_error(aperture_mm: float, focal_length_mm: float, wavelength_nm: float = VISUAL_WAVELENGTH_NM):
    """如果用球面代替抛物面，在最佳焦点处的波面误差。

    抛物面和同顶点球面在边缘的面形差 Δ = r⁴ / (8R³)，波面差 2Δ 呈 ρ⁴ 分布；
    调焦补偿后 PV = 2Δ / 4，RMS = 2Δ / (6√5)。返回 (PV 波长数, RMS 波长数, 斯特列尔比)。
    """
    r = aperture_mm / 2
    big_r = 2 * focal_length_mm
    w = 2 * r ** 4 / (8 * big_r ** 3) * 1e6  # 波面差，nm
    pv = w / 4 / wavelength_nm
    rms = w / (6 * math.sqrt(5)) / wavelength_nm
    return pv, rms, math.exp(-(2 * math.pi * rms) ** 2)


def min_focal_ratio_for_sphere(aperture_mm: float, strehl: float = 0.8) -> float:
    """球面镜要达到这个斯特列尔比（0.8 = 衍射极限），焦比至少要多少。"""
    f = 3.0
    while sphere_error(aperture_mm, f * aperture_mm)[2] < strehl:
        f += 0.1
    return round(f, 1)


@dataclass(frozen=True)
class FoucaultZone:
    inner_mm: float
    outer_mm: float
    reading_mm: float  # 刀口相对中心带区要后退的距离（光源随刀口移动）


def foucault_zones(aperture_mm: float, focal_length_mm: float, zones: int = 4) -> list[FoucaultZone]:
    """傅科检验的带区读数：等面积分带，光源和刀口一起移动时读数 = r² / R。"""
    r_max = aperture_mm / 2
    big_r = 2 * focal_length_mm
    edges = [r_max * math.sqrt(k / zones) for k in range(zones + 1)]
    out = []
    for inner, outer in zip(edges, edges[1:]):
        mean = (inner + outer) / 2
        out.append(FoucaultZone(inner, outer, mean ** 2 / big_r))
    return out


@dataclass(frozen=True)
class PrimarySpec:
    diameter_mm: float
    focal_length_mm: float
    thickness_mm: float

    @property
    def radius_of_curvature_mm(self) -> float:
        return 2 * self.focal_length_mm

    @property
    def focal_ratio(self) -> float:
        return self.focal_length_mm / self.diameter_mm

    @property
    def sagitta_mm(self) -> float:
        """镜面中心比边缘凹下去多深。"""
        return (self.diameter_mm / 2) ** 2 / (4 * self.focal_length_mm)

    @property
    def parabolizing_nm(self) -> float:
        """从最接近的球面修成抛物面，镜面要改动多少（面形，纳米）。"""
        return sphere_error(self.diameter_mm, self.focal_length_mm)[0] * VISUAL_WAVELENGTH_NM / 2

    @property
    def clear_aperture_mm(self) -> float:
        return self.diameter_mm - 4

    @property
    def mass_kg(self) -> float:
        return math.pi * (self.diameter_mm / 2) ** 2 * self.thickness_mm * GLASS_DENSITY_KG_MM3


@dataclass(frozen=True)
class SecondarySpec:
    minor_axis_mm: float

    @property
    def major_axis_mm(self) -> float:
        """45° 放置的椭圆平面镜：长轴 = 短轴 × √2。"""
        return self.minor_axis_mm * math.sqrt(2)

    @property
    def thickness_mm(self) -> float:
        """至少短轴的 1/6，取整到常见厚度。"""
        return max(6.0, 2 * math.ceil(self.minor_axis_mm / 6 / 2))
