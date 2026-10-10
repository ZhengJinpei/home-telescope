"""地平式（上下 + 左右）电动支架：自动寻星、跟踪，以及最重要的——不对准太阳。

没有硬件时用 SimulatedDriver，所有逻辑都能在电脑上跑。
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from . import astro
from .catalog import Target, find
from .sky import Observer, locate

# 两个轴默认：NEMA17 步进电机 200 步 × 16 细分 × 同步带 1:5 减速 = 16000 步/圈（约 81 角秒/步）
DEFAULT_STEPS_PER_REV = 200 * 16 * 5


class UnsafeTarget(Exception):
    """目标不安全或不可达，支架拒绝转动。"""


class MotorDriver(Protocol):
    def move(self, axis: str, steps: int) -> None:
        """axis 为 "alt" 或 "az"；steps 正负表示方向。"""


class SimulatedDriver:
    """模拟电机：只记录指令，不需要任何硬件。"""

    def __init__(self):
        self.log: list[tuple[str, int]] = []

    def move(self, axis: str, steps: int) -> None:
        if steps:
            self.log.append((axis, steps))


class StepDirDriver:
    """树莓派 + A4988 / TMC2209 这类“步进/方向”驱动板。

    需要 `pip install gpiozero`。尚未在真机上验证，第一次通电请先拆下镜筒、低速测试。
    """

    def __init__(self, pins: dict[str, tuple[int, int]], step_delay_s: float = 0.0005):
        from gpiozero import DigitalOutputDevice  # 只在树莓派上导入

        self._pins = {
            axis: (DigitalOutputDevice(step), DigitalOutputDevice(direction))
            for axis, (step, direction) in pins.items()
        }
        self._delay = step_delay_s

    def move(self, axis: str, steps: int) -> None:
        step_pin, dir_pin = self._pins[axis]
        dir_pin.value = steps > 0
        for _ in range(abs(steps)):
            step_pin.on()
            _time.sleep(self._delay)
            step_pin.off()
            _time.sleep(self._delay)


@dataclass
class GotoPlan:
    target: Target
    when: datetime
    alt: float  # 要指向的高度（已加大气折射）
    az: float
    alt_steps: int
    az_steps: int
    warnings: list[str] = field(default_factory=list)


class AltAzMount:
    def __init__(
        self,
        driver: MotorDriver,
        observer: Observer,
        *,
        alt_steps_per_rev: int = DEFAULT_STEPS_PER_REV,
        az_steps_per_rev: int = DEFAULT_STEPS_PER_REV,
        min_alt: float = 10.0,
        sun_avoid_deg: float = 20.0,
    ):
        self.driver = driver
        self.observer = observer
        self.alt_spr = alt_steps_per_rev
        self.az_spr = az_steps_per_rev
        self.min_alt = min_alt
        self.sun_avoid_deg = sun_avoid_deg
        # 用整数步数记录位置，避免反复换算累积误差。开机默认：水平指向正北
        self._alt_pos = 0
        self._az_pos = 0

    # -------------------------------------------------------------- 位置

    @property
    def alt(self) -> float:
        return self._alt_pos * 360.0 / self.alt_spr

    @property
    def az(self) -> float:
        return astro.norm360(self._az_pos * 360.0 / self.az_spr)

    def sync(self, alt: float, az: float) -> None:
        """告诉支架“现在镜筒指着这里”，不转动电机。"""
        self._alt_pos = round(alt / 360.0 * self.alt_spr)
        self._az_pos = round(astro.norm360(az) / 360.0 * self.az_spr)

    def align(self, target: Target | str, when: datetime) -> None:
        """单星校准：先手动把一颗亮星（比如织女星）放到视野中央，再调用这个。"""
        target = find(target) if isinstance(target, str) else target
        s = locate(target, when, self.observer)
        self.sync(s.alt + astro.refraction(s.alt), s.az)

    # -------------------------------------------------------------- 寻星

    def plan(self, target: Target | str, when: datetime) -> GotoPlan:
        """算出要转多少步，并做安全检查。不安全时抛出 UnsafeTarget。"""
        target = find(target) if isinstance(target, str) else target
        if target.key == "sun":
            raise UnsafeTarget("不能把望远镜对准太阳！直接看会让眼睛永久失明。")

        s = locate(target, when, self.observer)
        alt = s.alt + astro.refraction(s.alt)
        if alt < self.min_alt:
            where = "在地平线下面" if alt < 0 else f"太低了（高度 {alt:.0f}°）"
            raise UnsafeTarget(f"{target.name}现在{where}，看不到。")

        sun = astro.sun_position(when)
        sun_h = astro.to_horizontal(sun.ra, sun.dec, when, self.observer.lat, self.observer.lon)
        warnings = []
        if sun_h.alt > -2:
            sep = astro.angular_separation(s.az, alt, sun_h.az, sun_h.alt)
            if sep < self.sun_avoid_deg:
                raise UnsafeTarget(f"{target.name}离太阳只有 {sep:.0f}°，太危险，不转动。")
            if self._path_near_sun(alt, s.az, sun_h):
                raise UnsafeTarget("转过去的路上会扫过太阳。请盖上镜头盖，或者手动换个方向转。")
            warnings.append("现在是白天，确认主镜和寻星镜的盖子都盖好了！")
        if alt > 85:
            warnings.append("目标接近头顶，地平式支架在这里左右转得很快，跟踪会不太稳。")

        alt_steps = round(alt / 360.0 * self.alt_spr) - self._alt_pos
        d_az = (s.az - self.az + 180.0) % 360.0 - 180.0  # 走最短的方向
        az_steps = round(d_az / 360.0 * self.az_spr)
        return GotoPlan(target, when, alt, s.az, alt_steps, az_steps, warnings)

    def _path_near_sun(self, alt: float, az: float, sun: astro.Horizontal, samples: int = 60) -> bool:
        d_az = (az - self.az + 180.0) % 360.0 - 180.0
        for k in range(samples + 1):
            f = k / samples
            a = self.alt + (alt - self.alt) * f
            z = self.az + d_az * f
            if astro.angular_separation(z, a, sun.az, sun.alt) < self.sun_avoid_deg:
                return True
        return False

    def goto(self, target: Target | str, when: datetime) -> GotoPlan:
        plan = self.plan(target, when)
        self.driver.move("az", plan.az_steps)
        self.driver.move("alt", plan.alt_steps)
        self._az_pos += plan.az_steps
        self._alt_pos += plan.alt_steps
        return plan

    def track(
        self, target: Target | str, start: datetime, minutes: float, interval_s: float = 10.0
    ) -> list[GotoPlan]:
        """星星会随地球自转慢慢移动，每隔 interval_s 秒修正一次。

        这里按“模拟时间”计算；接真机时在两次修正之间 sleep 即可。
        """
        plans = []
        n = int(minutes * 60 / interval_s)
        for k in range(n + 1):
            plans.append(self.goto(target, start + timedelta(seconds=k * interval_s)))
        return plans
