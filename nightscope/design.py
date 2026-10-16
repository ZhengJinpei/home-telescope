"""数码夜视望远镜的光学和电源设计计算。

结构：镜头 → 星光级传感器 → 树莓派处理 → 目镜里的微型屏幕（同时 Wi-Fi 推到手机）。
晚上用红外灯照亮目标。这里只用标准库，方便在电脑上先把参数算清楚再买零件。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Sensor:
    name: str
    width_px: int
    height_px: int
    pixel_um: float
    note: str = ""

    @property
    def width_mm(self) -> float:
        return self.width_px * self.pixel_um / 1000

    @property
    def height_mm(self) -> float:
        return self.height_px * self.pixel_um / 1000

    @property
    def diagonal_mm(self) -> float:
        return math.hypot(self.width_mm, self.height_mm)


# 星光级传感器：近红外灵敏度高，有月光时不开灯也有彩色画面
IMX462 = Sensor("索尼 IMX462（1/2.8 寸星光级）", 1920, 1080, 2.9, "近红外特别灵敏，适合配 850 nm 红外灯")
IMX585 = Sensor("索尼 IMX585（1/1.2 寸星光级 2 代）", 3840, 2160, 2.9, "画面大一倍多，更贵，镜头也要更大")
SENSORS = {"imx462": IMX462, "imx585": IMX585}


@dataclass(frozen=True)
class Lens:
    focal_mm: float
    f_number: float

    @property
    def aperture_mm(self) -> float:
        return self.focal_mm / self.f_number


@dataclass(frozen=True)
class Viewer:
    """目镜里的微型屏幕。"""

    name: str
    width_px: int
    height_px: int
    width_mm: float
    eyepiece_focal_mm: float

    @property
    def apparent_fov_deg(self) -> float:
        """眼睛看到屏幕有多宽（度）。"""
        return math.degrees(2 * math.atan(self.width_mm / 2 / self.eyepiece_focal_mm))


# 0.39 寸 1024×768 硅基 OLED，像素 7.8 µm，配 15 mm 目镜
MICRO_OLED = Viewer("0.39 寸 1024×768 微型 OLED", 1024, 768, 8.0, 15.0)

# 判断距离用约翰逊准则：目标最窄的那一边上要有多少个像素（每个“周期”算 2 个像素）
JOHNSON_PX = {"发现": 2.0, "认出种类": 8.0, "看清细节": 13.0}
TARGETS = {  # 目标最窄的一边（米）
    "鹿、羊": 0.9,
    "人": 0.75,
    "狐狸、狗": 0.35,
    "野鸡、鸭子": 0.25,
    "小鸟": 0.10,
}


@dataclass(frozen=True)
class IlluminatorRef:
    """红外灯射程的参照点：这套配置在晴朗夜里大约能看清多远。

    换算规律：目标被照得多亮 ∝ 功率 / 光束立体角 / 距离²，
    每个像素收到多少光 ∝ 1 / 光圈数²，所以射程 ∝ √(功率 / 立体角) / 光圈数。
    参照值是估计值，实测一次后改这里就能校准。
    """

    power_w: float = 3.0
    beam_deg: float = 10.0
    f_number: float = 1.4
    range_m: float = 250.0


DEFAULT_IR_REF = IlluminatorRef()


def _solid_angle(full_angle_deg: float) -> float:
    return 2 * math.pi * (1 - math.cos(math.radians(full_angle_deg / 2)))


def field_of_view_deg(sensor: Sensor, lens: Lens) -> tuple[float, float, float]:
    """(水平, 垂直, 对角线) 视场角，度。"""
    def fov(size_mm: float) -> float:
        return math.degrees(2 * math.atan(size_mm / 2 / lens.focal_mm))
    return fov(sensor.width_mm), fov(sensor.height_mm), fov(sensor.diagonal_mm)


def magnification(sensor: Sensor, lens: Lens, viewer: Viewer, digital_zoom: float = 1.0) -> float:
    """从目镜里看，东西比肉眼看大几倍。"""
    hfov = field_of_view_deg(sensor, lens)[0] / digital_zoom
    return viewer.apparent_fov_deg / hfov


def lossless_zoom(sensor: Sensor, viewer: Viewer) -> float:
    """传感器像素比屏幕多，这么多倍以内的数码变倍不损失清晰度。"""
    return max(1.0, sensor.width_px / viewer.width_px)


def pixels_on_target(sensor: Sensor, lens: Lens, size_m: float, distance_m: float) -> float:
    return size_m / distance_m * lens.focal_mm / (sensor.pixel_um / 1000)


def resolution_range_m(sensor: Sensor, lens: Lens, size_m: float, pixels: float) -> float:
    """镜头和传感器的分辨率决定的距离：再远，目标就分不到这么多像素了。"""
    return size_m * lens.focal_mm / (pixels * sensor.pixel_um / 1000)


def ir_range_m(power_w: float, beam_deg: float, f_number: float, ref: IlluminatorRef = DEFAULT_IR_REF) -> float:
    """红外灯在晴朗的夜里能让你看清多远（估计）。"""
    scale = (power_w / ref.power_w) * (_solid_angle(ref.beam_deg) / _solid_angle(beam_deg))
    return ref.range_m * math.sqrt(scale) * ref.f_number / f_number


def recommended_beam_deg(sensor: Sensor, lens: Lens) -> float:
    """红外光束刚好盖住画面对角线，再留 20% 余量；更宽浪费光，更窄画面四角是黑的。"""
    return field_of_view_deg(sensor, lens)[2] * 1.2


@dataclass(frozen=True)
class PowerBudget:
    pi_w: float = 2.0  # 树莓派 Zero 2 W 采集、处理、推流
    display_w: float = 0.8  # 微型 OLED 和驱动板
    wifi_w: float = 0.5  # 开热点推流到手机
    ir_full_w: float = 3.0  # 红外灯满功率（电功率）
    boost_efficiency: float = 0.88  # 电池升压到 5 V
    ir_driver_efficiency: float = 0.85
    battery_wh: float = 2 * 3.0 * 3.6  # 两节 3000 mAh 的 18650 并联


IR_LEVELS = (0.0, 0.25, 0.5, 1.0)  # 红外灯四档：关、弱、中、强


def battery_hours(ir_level: int, phone: bool = True, budget: PowerBudget = PowerBudget()) -> float:
    logic = budget.pi_w + budget.display_w + (budget.wifi_w if phone else 0.0)
    ir = budget.ir_full_w * IR_LEVELS[ir_level] / budget.ir_driver_efficiency
    return budget.battery_wh / (logic / budget.boost_efficiency + ir)
