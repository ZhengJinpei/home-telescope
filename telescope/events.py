"""天象日历：月相、行星合月、月掩星、行星相合、冲日、大距、流星雨、日食和月食。

做法很朴素：每隔一段时间算一次位置，找到“过零点”和“最近点”，再用二分法、
黄金分割法细化到半分钟以内。月亮位置误差约 6 角分，所以日月食的时刻是“约”，
误差一般在几分钟内。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Iterable

from . import astro
from .catalog import Target, find
from .sky import Observer, locate, sun_altitude

# ---------------------------------------------------------------- 结果


@dataclass
class Event:
    when: datetime  # 天象发生的时刻（观测者当地时间）
    kind: str  # 月相 / 合月 / 月掩星 / 相合 / 冲日 / 大距 / 流星雨 / 月食 / 日食
    title: str
    detail: str  # 讲给孩子听的话
    visible: bool  # 在本地能不能看到
    view: str = ""  # 什么时候、往哪看
    importance: int = 1  # 1 普通，2 推荐，3 不要错过


# ---------------------------------------------------------------- 搜索工具

Fn = Callable[[float], float]  # 输入 UTC 时间戳（秒）


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, timezone.utc)


def _wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def _grid(start: float, end: float, step: float) -> list[float]:
    """start 到 end（含两端）每隔 step 秒一个点。"""
    n = math.ceil((end - start) / step)
    return [min(start + k * step, end) for k in range(n + 1)]


def _bisect(f: Fn, a: float, b: float, tol: float = 30.0) -> float:
    fa = f(a)
    while b - a > tol:
        m = (a + b) / 2
        fm = f(m)
        if (fm < 0) == (fa < 0):
            a, fa = m, fm
        else:
            b = m
    return (a + b) / 2


def _golden_min(f: Fn, a: float, b: float, tol: float = 30.0) -> float:
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def _crossings(f: Fn, ts: list[float], values: list[float] | None = None) -> list[float]:
    """f 是 (-180, 180] 内的角度差，找它穿过 0 的时刻（跳过 ±180 处的回绕）。"""
    vs = values if values is not None else [f(t) for t in ts]
    out = []
    for t0, t1, v0, v1 in zip(ts, ts[1:], vs, vs[1:]):
        if (v0 < 0) != (v1 < 0) and abs(v1 - v0) < 90:
            out.append(_bisect(f, t0, t1))
    return out


def _minima(f: Fn, ts: list[float], values: list[float] | None = None) -> list[tuple[float, float]]:
    """返回 [(时刻, 最小值)]。"""
    vs = values if values is not None else [f(t) for t in ts]
    out = []
    for k in range(1, len(ts) - 1):
        if vs[k] <= vs[k - 1] and vs[k] < vs[k + 1]:
            t = _golden_min(f, ts[k - 1], ts[k + 1])
            out.append((t, f(t)))
    return out


# ---------------------------------------------------------------- 位置


def _radec(target: Target, when: datetime) -> astro.Equatorial:
    """地心赤道坐标（月亮的 distance 为地球半径）。"""
    if target.key == "sun":
        return astro.sun_position(when)
    if target.key == "moon":
        return astro.moon_position(when).equatorial
    if target.solar_system:
        return astro.planet_position(target.key, when).equatorial
    ra, dec = astro.precess_from_j2000(target.ra, target.dec, when)
    return astro.Equatorial(ra, dec)


def _sep(a: astro.Equatorial, b: astro.Equatorial) -> float:
    return astro.angular_separation(a.ra, a.dec, b.ra, b.dec)


def _ecl_lon(target: Target, when: datetime) -> float:
    eq = _radec(target, when)
    return astro.equatorial_to_ecliptic(eq.ra, eq.dec, when)[0]


def _moon_radius(distance_earth_radii: float) -> float:
    """月亮视半径（度）。"""
    return math.degrees(math.asin(0.2725 / distance_earth_radii))


def _sun_radius(distance_au: float) -> float:
    return 0.2666 / distance_au


def _best_view(
    targets: list[Target],
    center: datetime,
    obs: Observer,
    *,
    window_h: float = 12.0,
    sun_limit: float = -6.0,
    min_alt: float = 8.0,
    penalty_per_hour: float = 0.0,
) -> tuple[datetime, list[float], list[float]] | None:
    """在 center 前后 window_h 小时内，找天黑且所有目标都够高的最佳时刻。

    返回 (时刻, 各目标高度, 各目标方位)，找不到时返回 None。
    penalty_per_hour > 0 时更偏向离 center 近的时刻（例如合月，月亮跑得快）。
    """
    best = None
    steps = int(window_h * 2 * 6)
    for k in range(steps + 1):
        t = center + timedelta(minutes=10 * k - window_h * 60)
        if sun_altitude(t, obs) >= sun_limit:
            continue
        ss = [locate(tg, t, obs) for tg in targets]
        low = min(s.alt for s in ss)
        if low < min_alt:
            continue
        score = low - penalty_per_hour * abs((t - center).total_seconds()) / 3600
        if best is None or score > best[0]:
            best = (score, t, [s.alt for s in ss], [s.az for s in ss])
    return None if best is None else best[1:]


def _where(alt: float, az: float) -> str:
    return f"{astro.compass(az)}方高 {alt:.0f}°"


def _hm(t: datetime) -> str:
    return t.strftime("%H:%M")


def _hm_near(t: datetime, ref: datetime) -> str:
    """和 ref 不是同一天时带上日期，免得把凌晨和前一晚弄混。"""
    return _hm(t) if t.date() == ref.date() else t.strftime("%m-%d %H:%M")


def _local(t: datetime, obs: Observer) -> datetime:
    return t.astimezone(obs.tz)


# ---------------------------------------------------------------- 月相

_PHASES = [
    ("新月", "月亮躲在太阳旁边，看不见。前后几天没有月光，是看星云和星系的好时机。"),
    ("上弦月", "傍晚就能看到半个月亮。用望远镜看明暗交界线，环形山的影子最清楚。"),
    ("满月", "整晚都能看到圆圆的月亮，但它太亮了，暗的星云星系会看不清。"),
    ("下弦月", "下半夜才升起的半个月亮，早起的话可以在早晨看到它。"),
]


def _moon_phases(ts: list[float]) -> list[tuple[float, int]]:
    out = []
    for k in range(4):
        def f(t, k=k):
            return _wrap180(astro.moon_age(_dt(t)) - 90 * k)
        out += [(t, k) for t in _crossings(f, ts)]
    return sorted(out)


def _phase_events(phases, obs: Observer) -> list[Event]:
    events = []
    for t, k in phases:
        name, detail = _PHASES[k]
        events.append(Event(_local(_dt(t), obs), "月相", name, detail, True))
    return events


# ---------------------------------------------------------------- 月食


def _lunar_eclipse(full_moon: float, obs: Observer) -> Event | None:
    def shadow_sep(t: float) -> float:
        when = _dt(t)
        moon = astro.moon_position(when).equatorial
        sun = astro.sun_position(when)
        return astro.angular_separation(moon.ra, moon.dec, sun.ra + 180.0, -sun.dec)

    t_max = _golden_min(shadow_sep, full_moon - 4 * 3600, full_moon + 4 * 3600)
    when = _dt(t_max)
    moon = astro.moon_position(when).equatorial
    sun = astro.sun_position(when)
    sep = shadow_sep(t_max)
    pi_moon = math.degrees(math.asin(1.0 / moon.distance))
    pi_sun = 8.794 / 3600 / sun.distance
    s_moon = _moon_radius(moon.distance)
    s_sun = _sun_radius(sun.distance)
    umbra = 1.02 * (pi_moon + pi_sun - s_sun)
    penumbra = 1.02 * (pi_moon + pi_sun + s_sun)

    umbral_mag = (umbra + s_moon - sep) / (2 * s_moon)
    penumbral_mag = (penumbra + s_moon - sep) / (2 * s_moon)
    if penumbral_mag <= 0:
        return None

    def contact(radius: float) -> tuple[float, float]:
        def g(t: float) -> float:
            return shadow_sep(t) - radius
        return _bisect(g, t_max - 5 * 3600, t_max), _bisect(g, t_max, t_max + 5 * 3600)

    phases = []  # (名称, 时刻)
    if umbral_mag > 0:
        u1, u4 = contact(umbra + s_moon)
        phases += [("初亏", u1), ("复圆", u4)]
        if umbral_mag >= 1:
            u2, u3 = contact(umbra - s_moon)
            phases += [("食既", u2), ("生光", u3)]
            kind, importance = "月全食", 3
            detail = ("地球的影子把月亮整个挡住了。月亮不会消失，而是变成暗红色，"
                      "因为地球大气把红光折射了进去——那是全世界的日出日落照在月亮上。肉眼就能看，不用任何保护。")
        else:
            kind, importance = "月偏食", 2
            detail = "地球的影子挡住了月亮的一部分，像被咬了一口。肉眼就能看，不用任何保护。"
        span = (u1, u4)
    else:
        kind, importance = "半影月食", 1
        detail = "月亮只穿过地球影子的边缘，只会稍微暗一点，很难看出来。"
        span = contact(penumbra + s_moon)
    phases.append(("食甚", t_max))
    phases.sort(key=lambda p: p[1])

    moon_t = find("moon")
    up = [t for t in _grid(span[0], span[1], 300) if locate(moon_t, _dt(t), obs).alt > 0]
    visible = bool(up)
    parts = "，".join(f"{name} {_hm(_local(_dt(t), obs))}" for name, t in phases)
    at_max = locate(moon_t, when, obs)
    if not visible:
        view = f"{parts}。这段时间月亮在地平线下，本地看不到"
    elif at_max.alt > 0:
        view = f"{parts}。食甚时月亮在{_where(at_max.alt, at_max.az)}"
        if len(up) * 300 < span[1] - span[0] - 300:
            view += f"；本地能看到 {_hm(_local(_dt(up[0]), obs))}–{_hm(_local(_dt(up[-1]), obs))} 这一段"
    else:
        view = f"{parts}。本地只能看到 {_hm(_local(_dt(up[0]), obs))}–{_hm(_local(_dt(up[-1]), obs))} 这一段"
    title = kind + (f"（食分 {umbral_mag:.2f}）" if umbral_mag > 0 else "")
    return Event(_local(when, obs), "月食", title, detail, visible, view, importance if visible else 1)


# ---------------------------------------------------------------- 日食

_SUN_SAFETY = ("⚠ 千万不能直接看太阳，也不能用普通墨镜！要用合格的日食眼镜，或者用小孔成像看投影。"
               "望远镜必须在镜筒前面装正规的太阳滤膜。")


def _solar_eclipse(new_moon: float, obs: Observer) -> Event | None:
    def geo_sep(t: float) -> float:
        when = _dt(t)
        return _sep(astro.moon_position(when).equatorial, astro.sun_position(when))

    t_max = _golden_min(geo_sep, new_moon - 4 * 3600, new_moon + 4 * 3600)
    when = _dt(t_max)
    moon = astro.moon_position(when).equatorial
    sun = astro.sun_position(when)
    pi_moon = math.degrees(math.asin(1.0 / moon.distance))
    pi_sun = 8.794 / 3600 / sun.distance
    s_moon, s_sun = _moon_radius(moon.distance), _sun_radius(sun.distance)
    if geo_sep(t_max) > pi_moon - pi_sun + s_sun + s_moon:
        return None  # 地球上哪里都看不到

    best = None  # (分离, 时刻)
    seen = []
    for t in _grid(t_max - 4 * 3600, t_max + 4 * 3600, 120):
        w = _dt(t)
        if sun_altitude(w, obs) < -0.5:
            continue
        m = astro.moon_position(w).equatorial
        topo = astro.topocentric(m, w, obs.lat, obs.lon)
        s = astro.sun_position(w)
        sep = _sep(topo, s)
        if sep < _sun_radius(s.distance) + _moon_radius(m.distance):
            seen.append(t)
            if best is None or sep < best[0]:
                best = (sep, t)

    if best is None:
        return Event(_local(when, obs), "日食", "日食（本地看不到）",
                     "月亮正好从太阳前面经过，地球上有些地方能看到日食，但我们这里看不到。", False)

    sep, t = best
    w = _dt(t)
    m = astro.moon_position(w).equatorial
    s_moon = _moon_radius(m.distance)
    mag = (s_sun + s_moon - sep) / (2 * s_sun)
    if sep < s_moon - s_sun:
        kind, story = "日全食", "月亮把太阳完全挡住，天会突然黑下来，能看到星星和太阳的“外衣”日冕。一辈子可能只能遇到一两次！"
    elif sep < s_sun - s_moon:
        kind, story = "日环食", "月亮比太阳小一点点，挡不全，太阳会变成一个金色的圆环。"
    else:
        kind, story = "日偏食", "月亮挡住了太阳的一部分，太阳会变成一个缺口的饼干。"
    if abs(sep - abs(s_moon - s_sun)) < 0.1:  # 月亮位置误差约 6 角分，边界附近说不准
        band = "全食" if s_moon > s_sun else "环食"
        maybe = "日偏食" if kind != "日偏食" else f"日{band}"
        story += f"本地在日{band}带的边缘附近，也有可能是{maybe}，请以天文台的预报为准。"
    sun_h = locate(find("sun"), w, obs)
    rise = sun_altitude(_dt(seen[0] - 120), obs) < -0.5
    sets = sun_altitude(_dt(seen[-1] + 120), obs) < -0.5
    begin = f"{'太阳升起时已经开始（带食日出）' if rise else '开始'} {_hm(_local(_dt(seen[0]), obs))}"
    finish = f"{'太阳落山时还没结束（带食日落）' if sets else '结束'} {_hm(_local(_dt(seen[-1]), obs))}"
    view = f"{begin}，最大 {_hm(_local(w, obs))}（太阳在{_where(sun_h.alt, sun_h.az)}），{finish}"
    return Event(_local(w, obs), "日食", f"{kind}（食分 {mag:.2f}）", f"{story}\n{_SUN_SAFETY}", True, view, 3)


# ---------------------------------------------------------------- 合月、月掩星、行星相合

_NAKED_EYE_PLANETS = ("mercury", "venus", "mars", "jupiter", "saturn")
# 月亮经常从旁边经过的亮星和星团
_MOON_NEIGHBORS = ("aldebaran", "regulus", "spica", "antares", "M45", "M44")


def _moon_conjunctions(ts: list[float], obs: Observer) -> list[Event]:
    moon_t = find("moon")
    moons = [astro.moon_position(_dt(t)).equatorial for t in ts]
    events = []
    for key in _NAKED_EYE_PLANETS + _MOON_NEIGHBORS:
        target = find(key)
        # 月亮每个月都会经过这些亮星和星团，只有挨得特别近才值得说
        limit = 5.0 if target.solar_system else 2.0

        def f(t, target=target):
            when = _dt(t)
            return _sep(astro.moon_position(when).equatorial, _radec(target, when))

        values = [_sep(m, _radec(target, _dt(t))) for t, m in zip(ts, moons)]
        for t_min, geo_sep in _minima(f, ts, values):
            if geo_sep > limit + 1.0:  # 站在地面看，月亮位置最多能差 1°
                continue
            ev = _occultation(target, t_min, obs)
            if ev is None:
                ev = _conjunction_event(moon_t, target, t_min, geo_sep, obs, limit)
            if ev is not None:
                events.append(ev)
    return events


def _topo_sep(target: Target, when: datetime, obs: Observer) -> float:
    moon = astro.moon_position(when).equatorial
    return _sep(astro.topocentric(moon, when, obs.lat, obs.lon), _radec(target, when))


def _occultation(target: Target, t_geo: float, obs: Observer) -> Event | None:
    """月亮从星星或行星前面挡过去。只报天黑时、月亮够高、在本地能看到的。"""
    if target.kind == "星团":
        return None
    moon_t = find("moon")
    # 先每 10 分钟粗看一遍（月亮 10 分钟只走 0.1°），离得远就不用细算了
    coarse = [(_topo_sep(target, _dt(t), obs), t) for t in _grid(t_geo - 6 * 3600, t_geo + 6 * 3600, 600)]
    closest, t_close = min(coarse)
    if closest > 0.45:
        return None
    covered = []  # (时刻, 站心角距, 月亮视半径)
    for t in _grid(t_close - 2 * 3600, t_close + 2 * 3600, 60):
        w = _dt(t)
        radius = _moon_radius(astro.moon_position(w).equatorial.distance)
        sep = _topo_sep(target, w, obs)
        if sep < radius:
            covered.append((w, sep, radius))
    if not covered:
        return None
    seen = [w for w, _, _ in covered if sun_altitude(w, obs) < -6 and locate(moon_t, w, obs).alt >= 5]
    if not seen:
        return None
    start, end = _local(covered[0][0], obs), _local(covered[-1][0], obs)
    s = locate(moon_t, seen[0], obs)
    detail = (f"月亮会从{target.name}前面经过，把它挡住一会儿。用望远镜盯着月亮的边缘，"
              f"{target.name}会“啪”地一下突然消失，过一会儿再从另一边冒出来。")
    view = f"约 {_hm(start)} 消失，{_hm(end)} 重新出现，月亮在{_where(s.alt, s.az)}。时刻可能差几分钟，提前 10 分钟开始看"
    if len(seen) < len(covered):
        view += "；有一段发生在天还亮或月亮太低的时候"
    if min(sep - radius for _, sep, radius in covered) > -0.1:
        view += "；月亮只是擦边经过，也可能只擦过不挡住"
    return Event(start, "月掩星", f"月掩{target.name}", detail, True, view, 3)


def _conjunction_event(
    a: Target, b: Target, t: float, geo_sep: float, obs: Observer, limit: float
) -> Event | None:
    """a 和 b 靠得最近的时候。相距超过 limit 度就不算数，返回 None。"""
    when = _dt(t)
    is_moon = a.key == "moon"
    kind = "合月" if is_moon else "相合"
    title = f"{a.name}和{b.name}相会"
    found = _best_view([a, b], when, obs, penalty_per_hour=2.0 if is_moon else 0.5)
    if found is None:
        if geo_sep > limit:
            return None
        return Event(_local(when, obs), kind, f"{title}（相距 {geo_sep:.1f}°）",
                     "它们靠得很近，但本地看的时候不是在白天就是在地平线下。", False)
    t_view, alts, azs = found
    sep = _topo_sep(b, t_view, obs) if is_moon else _sep(_radec(a, t_view), _radec(b, t_view))
    if sep > limit:
        return None
    if not is_moon:
        detail = "两颗行星在天上挨得很近，其实它们相距好几亿公里，只是从地球看过去在同一个方向。"
        importance = 3 if sep < 1 else 2 if sep < 2 else 1
        if sep < 0.7:
            detail += "低倍目镜里能同时看到它们！"
    elif b.kind == "星团":
        detail = f"月亮旁边那一小团星星就是{b.name}。月光会让它变淡，用双筒望远镜看更清楚。"
        importance = 1
    elif b.kind == "行星":
        detail = f"月亮旁边那颗亮亮的就是{b.name}，用眼睛就能找到，是认识{b.name}的好机会。"
        importance = 2 if sep < 4 else 1
    else:
        detail = f"月亮旁边那颗亮星就是{b.name}，用眼睛就能找到。"
        importance = 1
    view = f"{_hm_near(_local(t_view, obs), _local(when, obs))} 在{_where(alts[0], azs[0])}，相距 {sep:.1f}°"
    return Event(_local(when, obs), kind, f"{title}（相距 {sep:.1f}°）", detail, True, view, importance)


def _planet_conjunctions(ts: list[float], obs: Observer) -> list[Event]:
    planets = [find(k) for k in _NAKED_EYE_PLANETS]
    pos = {p.key: [_radec(p, _dt(t)) for t in ts] for p in planets}
    events = []
    for i, a in enumerate(planets):
        for b in planets[i + 1:]:
            def f(t, a=a, b=b):
                when = _dt(t)
                return _sep(_radec(a, when), _radec(b, when))

            values = [_sep(x, y) for x, y in zip(pos[a.key], pos[b.key])]
            for t_min, sep in _minima(f, ts, values):
                ev = _conjunction_event(a, b, t_min, sep, obs, 3.0) if sep <= 3.0 else None
                if ev is not None:
                    events.append(ev)
    return events


# ---------------------------------------------------------------- 冲日、大距

_OPPOSITION_NOTES = {
    "mars": (3, "火星大约两年才冲日一次，这时它最亮、最大，望远镜里有机会看到白色的极冠和暗色的花纹。"),
    "jupiter": (2, "木星一年冲日一次，这时它最亮，整晚都在天上，看木星条纹和四颗大卫星的最好时候。"),
    "saturn": (2, "土星一年冲日一次，这时它最亮，整晚都能看到它的光环。"),
    "uranus": (1, "不过天王星很暗，需要望远镜和星图才能找到这个蓝绿色的小圆点。"),
    "neptune": (1, "不过海王星非常暗，需要望远镜和星图耐心寻找。"),
}


def _oppositions(ts: list[float], obs: Observer) -> list[Event]:
    events = []
    for key, (importance, detail) in _OPPOSITION_NOTES.items():
        planet = find(key)

        def f(t, planet=planet):
            when = _dt(t)
            return _wrap180(_ecl_lon(planet, when) - astro.sun_longitude(when) - 180.0)

        for t in _crossings(f, ts):
            when = _dt(t)
            mag = astro.planet_position(key, when).magnitude
            found = _best_view([planet], when, obs, sun_limit=-12.0)
            title = f"{planet.name}冲日（{mag:+.1f} 等）"
            full = f"{planet.name}和太阳正好在地球的两边：太阳落山时它升起，整晚都能看到。{detail}"
            if found is None:
                events.append(Event(_local(when, obs), "冲日", title, full, False))
                continue
            t_view, alts, azs = found
            view = f"前后几周都适合看，最高时 {_hm(_local(t_view, obs))} 在{_where(alts[0], azs[0])}"
            events.append(Event(_local(when, obs), "冲日", title, full, True, view, importance))
    return events


def _elongations(ts: list[float], obs: Observer) -> list[Event]:
    events = []
    for key in ("mercury", "venus"):
        planet = find(key)

        def neg_elong(t, key=key):
            return -astro.planet_position(key, _dt(t)).elongation

        for t, value in _minima(neg_elong, ts):
            when = _dt(t)
            east = _wrap180(_ecl_lon(planet, when) - astro.sun_longitude(when)) > 0
            side, place = ("东大距", "黄昏后在西方低空") if east else ("西大距", "黎明前在东方低空")
            detail = (f"{planet.name}离太阳最远（{-value:.0f}°），是一年里看{planet.name}最好的机会之一，"
                      f"{place}找它。")
            if key == "mercury":
                detail += "水星总是躲在太阳旁边，很多人一辈子都没见过它！"
            else:
                detail += "望远镜里金星正好是“半个月亮”的样子。"
            found = _best_view([planet], when, obs, min_alt=5.0)
            title = f"{planet.name}{side}"
            if found is None:
                events.append(Event(_local(when, obs), "大距", title, detail, False))
                continue
            t_view, alts, azs = found
            view = f"前后一两周都能看，{_hm(_local(t_view, obs))} 在{_where(alts[0], azs[0])}"
            events.append(Event(_local(when, obs), "大距", title, detail, True, view, 2))
    return events


# ---------------------------------------------------------------- 流星雨


@dataclass(frozen=True)
class MeteorShower:
    name: str
    solar_longitude: float  # 极大时的太阳黄经（J2000，国际流星组织 IMO 数据）
    ra: float  # 辐射点赤经（度，J2000）
    dec: float
    zhr: int  # 理想条件下每小时流星数
    story: str


METEOR_SHOWERS = [
    MeteorShower("象限仪座流星雨", 283.15, 230, 49, 80,
                 "新年第一场流星雨，来自一颗叫 2003 EH1 的小行星。极大只持续几个小时，要碰运气。"),
    MeteorShower("天琴座流星雨", 32.32, 271, 34, 18,
                 "中国古人在 2600 多年前就记录过它（“夜中星陨如雨”），是人类最早记录的流星雨之一。"),
    MeteorShower("宝瓶座η流星雨", 45.5, 338, -1, 50,
                 "这些流星是哈雷彗星留下的碎屑。辐射点黎明前才升起，在北方看到的不多。"),
    MeteorShower("英仙座流星雨", 140.0, 48, 58, 100,
                 "暑假最受欢迎的流星雨，来自斯威夫特-塔特尔彗星。找个暗的地方躺下来看最舒服。"),
    MeteorShower("猎户座流星雨", 208.0, 95, 16, 20,
                 "也是哈雷彗星留下的碎屑，流星飞得很快。"),
    MeteorShower("狮子座流星雨", 235.27, 152, 22, 15,
                 "1833 年和 1966 年它下过真正的“流星暴雨”，一小时几万颗！平常年份只有十几颗。"),
    MeteorShower("双子座流星雨", 262.2, 112, 33, 150,
                 "一年里最壮观的流星雨，来自一颗叫法厄同的小行星。前半夜就能看，最适合小朋友。"),
]


def _sun_longitude_j2000(when: datetime) -> float:
    years = (astro.julian_date(when) - 2451545.0) / 365.25
    return astro.sun_longitude(when) - 0.013969 * years  # 扣掉岁差


def _meteor_showers(ts: list[float], obs: Observer) -> list[Event]:
    events = []
    for shower in METEOR_SHOWERS:
        def f(t, shower=shower):
            return _wrap180(_sun_longitude_j2000(_dt(t)) - shower.solar_longitude)

        for t in _crossings(f, ts):
            peak = _local(_dt(t), obs)
            events.append(_meteor_event(shower, peak, obs))
    return events


def _meteor_event(shower: MeteorShower, peak: datetime, obs: Observer) -> Event:
    # 极大如果在中午以前，就算前一晚
    night = peak.date() - timedelta(days=1) if peak.hour < 12 else peak.date()
    start = datetime.combine(night, time(17, 0), tzinfo=obs.tz)
    radiant = Target("radiant", "辐射点", "恒星", "", shower.ra, shower.dec)
    moon_t = find("moon")
    best = None
    for k in range(14 * 6 + 1):
        t = start + timedelta(minutes=10 * k)
        if sun_altitude(t, obs) > -12:
            continue
        r = locate(radiant, t, obs)
        if best is None or r.alt > best[1].alt:
            best = (t, r)
    title = f"{shower.name}极大"
    detail = f"{shower.story}流星雨用眼睛看就行，不用望远镜。"
    if best is None or best[1].alt < 10:
        return Event(peak, "流星雨", title, detail + "可惜在本地辐射点太低。", False)
    t_best, r = best
    rate = shower.zhr * math.sin(math.radians(r.alt))
    _, illum = astro.moon_phase(t_best)
    moon_up = locate(moon_t, t_best, obs).alt > 0
    if illum > 0.6 and moon_up:
        moon_note = f"月光很亮（亮面 {illum:.0%}），只能看到最亮的流星"
    elif illum > 0.3 and moon_up:
        moon_note = f"有些月光（亮面 {illum:.0%}）"
    else:
        moon_note = "没有月光干扰，好机会"
    best_txt = f" {_hm(t_best)}" if t_best.date() == night else f"凌晨 {_hm(t_best)}"
    view = (f"{night:%m-%d} 晚上，最好{best_txt} 前后，辐射点在{_where(r.alt, r.az)}。"
            f"在郊外暗处每小时约 {rate:.0f} 颗，城里会少很多；{moon_note}")
    good_moon = not (illum > 0.6 and moon_up)
    importance = 3 if shower.zhr >= 50 and good_moon else 2 if shower.zhr >= 50 or good_moon else 1
    return Event(peak, "流星雨", title, detail, True, view, importance)


# ---------------------------------------------------------------- 汇总


def find_events(start: date, days: int, obs: Observer) -> list[Event]:
    """start 当天 0 点起 days 天之内的天象，按时间排序。"""
    t0 = datetime.combine(start, time(0, 0), tzinfo=obs.tz).timestamp()
    t1 = t0 + days * 86400
    pad = 2 * 86400  # 多算两天，免得边界上的极值找不到
    fine = _grid(t0 - pad, t1 + pad, 2 * 3600)
    medium = _grid(t0 - pad, t1 + pad, 6 * 3600)
    coarse = _grid(t0 - pad, t1 + pad, 86400)

    phases = _moon_phases(medium)
    events = _phase_events(phases, obs)
    for t, k in phases:
        if k == 2:
            events.append(_lunar_eclipse(t, obs))
        elif k == 0:
            events.append(_solar_eclipse(t, obs))
    events += _moon_conjunctions(fine, obs)
    events += _planet_conjunctions(_grid(t0 - pad, t1 + pad, 12 * 3600), obs)
    events += _oppositions(coarse, obs)
    events += _elongations(coarse, obs)
    events += _meteor_showers(coarse, obs)

    lo, hi = _dt(t0), _dt(t1)
    return sorted((e for e in events if e is not None and lo <= e.when < hi), key=lambda e: e.when)


def highlights(events: Iterable[Event], min_importance: int = 2) -> list[Event]:
    return [e for e in events if e.visible and e.importance >= min_importance]
