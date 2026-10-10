"""某个时刻某个目标在哪里，以及“今晚看什么”。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

from . import astro
from .catalog import ALL_TARGETS, Target
from .optics import Telescope


@dataclass(frozen=True)
class Observer:
    lat: float  # 纬度，北纬为正
    lon: float  # 经度，东经为正
    tz_hours: float = 8.0  # 时区，中国是 UTC+8

    @property
    def tz(self) -> timezone:
        return timezone(timedelta(hours=self.tz_hours))


# 方便孩子直接说城市名
CITIES = {
    "北京": (39.90, 116.40), "上海": (31.23, 121.47), "广州": (23.13, 113.26),
    "深圳": (22.54, 114.06), "天津": (39.13, 117.20), "重庆": (29.56, 106.55),
    "成都": (30.66, 104.07), "杭州": (30.27, 120.15), "南京": (32.06, 118.80),
    "武汉": (30.59, 114.31), "西安": (34.34, 108.94), "长沙": (28.23, 112.94),
    "郑州": (34.75, 113.62), "济南": (36.65, 117.12), "青岛": (36.07, 120.38),
    "沈阳": (41.80, 123.43), "哈尔滨": (45.80, 126.53), "昆明": (25.04, 102.71),
    "厦门": (24.48, 118.09), "福州": (26.07, 119.30), "合肥": (31.82, 117.23),
    "南宁": (22.82, 108.37), "贵阳": (26.65, 106.63), "兰州": (36.06, 103.83),
    "乌鲁木齐": (43.83, 87.62), "拉萨": (29.65, 91.13), "呼和浩特": (40.84, 111.75),
    "海口": (20.04, 110.20), "香港": (22.32, 114.17), "台北": (25.03, 121.57),
}


def observer_for_city(city: str) -> Observer:
    try:
        lat, lon = CITIES[city]
    except KeyError:
        raise KeyError(f"还不认识“{city}”，请用 --lat/--lon 直接给出经纬度") from None
    return Observer(lat, lon)


@dataclass(frozen=True)
class Sighting:
    target: Target
    when: datetime
    ra: float  # 观测日期的赤经（度）
    dec: float
    alt: float  # 站在地面看的高度角（未计折射）
    az: float
    magnitude: float | None
    illumination: float | None = None  # 月亮、内行星的亮面比例


def locate(target: Target, when: datetime, obs: Observer) -> Sighting:
    illum = None
    mag = target.magnitude
    if target.key == "sun":
        eq = astro.sun_position(when)
        mag = -26.7
    elif target.key == "moon":
        body = astro.moon_position(when)
        eq, mag, illum = body.equatorial, body.magnitude, body.illumination
    elif target.solar_system:
        body = astro.planet_position(target.key, when)
        eq, mag, illum = body.equatorial, body.magnitude, body.illumination
    else:
        ra, dec = astro.precess_from_j2000(target.ra, target.dec, when)
        eq = astro.Equatorial(ra, dec)

    h = astro.to_horizontal(eq.ra, eq.dec, when, obs.lat, obs.lon)
    alt = h.alt
    if target.key == "moon":
        alt = astro.topocentric_moon_altitude(alt, eq.distance)
    return Sighting(target, when, eq.ra, eq.dec, alt, h.az, mag, illum)


def sun_altitude(when: datetime, obs: Observer) -> float:
    sun = astro.sun_position(when)
    return astro.to_horizontal(sun.ra, sun.dec, when, obs.lat, obs.lon).alt


# ---------------------------------------------------------------- 今晚看什么

_PRIORITY = {"月亮": 0, "行星": 1, "双星": 2, "星团": 3, "星云": 3, "星系": 4, "恒星": 5}
_FAINT_PLANETS = {"uranus", "neptune"}


@dataclass
class TonightItem:
    target: Target
    best_time: datetime
    best_alt: float
    best_az: float
    magnitude: float | None
    difficulty: int  # 1–3 颗星，越多越难
    notes: list[str] = field(default_factory=list)


@dataclass
class NightPlan:
    observer: Observer
    night_of: date
    dusk: datetime | None  # 天黑到能看亮星（太阳在地平线下 6°）
    dark_start: datetime | None  # 完全天黑（太阳在地平线下 12°）
    dark_end: datetime | None
    dawn: datetime | None
    moon_phase: str
    moon_illumination: float
    items: list[TonightItem]


def _difficulty(target: Target, mag: float | None) -> int:
    if target.kind in ("月亮", "行星") and target.key not in _FAINT_PLANETS:
        return 1
    if target.kind in ("恒星", "双星"):
        return 1
    if mag is not None and mag < 5.0 and target.kind == "星团":
        return 1
    if mag is not None and mag < 6.5:
        return 2
    return 3


def plan_tonight(
    night_of: date,
    obs: Observer,
    scope: Telescope | None = None,
    *,
    min_alt: float = 15.0,
    step_minutes: int = 10,
) -> NightPlan:
    """规划 night_of 这天傍晚 17 点到第二天早上 7 点之间能看什么。"""
    start = datetime.combine(night_of, time(17, 0), tzinfo=obs.tz)
    steps = int(14 * 60 / step_minutes) + 1
    times = [start + timedelta(minutes=step_minutes * k) for k in range(steps)]
    sun_alts = [sun_altitude(t, obs) for t in times]

    def first(cond, seq):
        return next((t for t, a in seq if cond(a)), None)

    pairs = list(zip(times, sun_alts))
    dusk = first(lambda a: a < -6, pairs)
    dark_start = first(lambda a: a < -12, pairs)
    dark_end = first(lambda a: a < -12, reversed(pairs))
    dawn = first(lambda a: a < -6, reversed(pairs))

    midnight = datetime.combine(night_of + timedelta(days=1), time(0, 0), tzinfo=obs.tz)
    phase, moon_illum = astro.moon_phase(midnight)

    limit = scope.limiting_magnitude - 1.5 if scope else 6.0  # 默认按肉眼加城市光污染

    items: list[TonightItem] = []
    for target in ALL_TARGETS:
        if target.key == "sun":
            continue
        bright = target.kind in ("月亮", "行星", "恒星", "双星")
        threshold = -6 if bright else -12
        best: Sighting | None = None
        for t, sun_alt in pairs:
            if sun_alt >= threshold:
                continue
            s = locate(target, t, obs)
            if s.alt >= min_alt and (best is None or s.alt > best.alt):
                best = s
        if best is None:
            continue

        mag = best.magnitude
        extended = target.kind in ("星云", "星系")
        if mag is not None and mag > limit - (2.0 if extended else 0.0):
            continue

        notes = []
        if target.key == "moon":
            notes.append(f"{phase}，亮面 {moon_illum:.0%}")
            if moon_illum > 0.9:
                notes.append("满月太亮，可以加月亮滤镜；看环形山最好选上弦或下弦前后")
        if extended and moon_illum > 0.5:
            moon_now = locate(_moon(), best.when, obs)
            if moon_now.alt > 0:
                notes.append("月光会让它变淡，月亮落山后再看更好")
        if target.key in ("mercury", "venus") and best.illumination is not None:
            notes.append(f"像小月亮一样，现在亮面约 {best.illumination:.0%}")
        if best.alt < 25:
            notes.append("位置较低，注意楼房和树木遮挡")

        items.append(TonightItem(target, best.when, best.alt, best.az, mag, _difficulty(target, mag), notes))

    items.sort(key=lambda i: (
        _PRIORITY.get(i.target.kind, 9) + (3 if i.target.key in _FAINT_PLANETS else 0),
        i.magnitude if i.magnitude is not None else 99,
    ))
    return NightPlan(obs, night_of, dusk, dark_start, dark_end, dawn, phase, moon_illum, items)


def _moon() -> Target:
    return next(t for t in ALL_TARGETS if t.key == "moon")
