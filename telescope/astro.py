"""天文计算：时间、坐标转换、太阳/月亮/行星位置。

只用标准库。行星和月亮用 Paul Schlyter 的简化轨道算法
（https://stjarnhimlen.se/comp/ppcomp.html），精度约 1–2 角分，
对目视观测和小型望远镜自动寻星已经足够。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

# ---------------------------------------------------------------- 角度工具


def _sind(x: float) -> float:
    return math.sin(math.radians(x))


def _cosd(x: float) -> float:
    return math.cos(math.radians(x))


def _atan2d(y: float, x: float) -> float:
    return math.degrees(math.atan2(y, x))


def _asind(x: float) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, x))))


def norm360(x: float) -> float:
    return x % 360.0


def angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """两点之间的角距离（度）。输入可以是 赤经/赤纬，也可以是 方位/高度。"""
    c = _sind(dec1) * _sind(dec2) + _cosd(dec1) * _cosd(dec2) * _cosd(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def hms(hours: float) -> str:
    """12.5 -> '12h30m00s'"""
    total = round(hours * 3600) % (24 * 3600)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def dms(deg: float) -> str:
    """-16.716 -> '-16°42′58″'"""
    sign = "-" if deg < 0 else "+"
    total = round(abs(deg) * 3600)
    d, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{d:02d}°{m:02d}′{s:02d}″"


def compass(az: float) -> str:
    """方位角转成中文方向，方便孩子理解。"""
    names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return names[int((norm360(az) + 22.5) // 45) % 8]


# ---------------------------------------------------------------- 时间


def to_utc(when: datetime) -> datetime:
    if when.tzinfo is None:
        raise ValueError("时间必须带时区，例如 datetime(..., tzinfo=timezone.utc)")
    return when.astimezone(timezone.utc)


def julian_date(when: datetime) -> float:
    t = to_utc(when)
    return 2440587.5 + t.timestamp() / 86400.0


def _day_number(when: datetime) -> float:
    """Schlyter 算法使用的日数：从 1999-12-31 0h UT 起算。"""
    return julian_date(when) - 2451543.5


def obliquity(when: datetime) -> float:
    """黄赤交角（度）。"""
    return 23.4393 - 3.563e-7 * _day_number(when)


def local_sidereal_time(when: datetime, lon: float) -> float:
    """地方恒星时（度）。lon 东经为正。"""
    jd = julian_date(when)
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
    return norm360(gmst + lon)


# ---------------------------------------------------------------- 坐标


@dataclass(frozen=True)
class Equatorial:
    ra: float  # 赤经，度
    dec: float  # 赤纬，度
    distance: float = 0.0  # 天文单位（月亮为地球半径）；恒星为 0


@dataclass(frozen=True)
class Horizontal:
    alt: float  # 高度角，度（地平线以上为正）
    az: float  # 方位角，度（北 0，东 90）


def precess_from_j2000(ra: float, dec: float, when: datetime) -> tuple[float, float]:
    """把 J2000 星表坐标进动到观测日期（IAU 1976）。"""
    t = (julian_date(when) - 2451545.0) / 36525.0
    zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t**3) / 3600.0
    z = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t**3) / 3600.0
    theta = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t**3) / 3600.0
    a = _cosd(dec) * _sind(ra + zeta)
    b = _cosd(theta) * _cosd(dec) * _cosd(ra + zeta) - _sind(theta) * _sind(dec)
    c = _sind(theta) * _cosd(dec) * _cosd(ra + zeta) + _cosd(theta) * _sind(dec)
    return norm360(_atan2d(a, b) + z), _asind(c)


def refraction(alt: float) -> float:
    """大气折射（度），Bennett 公式。望远镜实际要指向的高度 = 几何高度 + 折射。"""
    if alt < -1.0:
        return 0.0
    return 1.0 / math.tan(math.radians(alt + 7.31 / (alt + 4.4))) / 60.0


def to_horizontal(
    ra: float, dec: float, when: datetime, lat: float, lon: float, *, apply_refraction: bool = False
) -> Horizontal:
    ha = local_sidereal_time(when, lon) - ra
    x = _cosd(ha) * _cosd(dec)
    y = _sind(ha) * _cosd(dec)
    z = _sind(dec)
    xhor = x * _sind(lat) - z * _cosd(lat)
    zhor = x * _cosd(lat) + z * _sind(lat)
    az = norm360(_atan2d(y, xhor) + 180.0)
    alt = _asind(zhor)
    if apply_refraction:
        alt += refraction(alt)
    return Horizontal(alt=alt, az=az)


def _ecliptic_to_equatorial(lon: float, lat: float, dist: float, when: datetime) -> Equatorial:
    ecl = obliquity(when)
    x = _cosd(lon) * _cosd(lat)
    y = _sind(lon) * _cosd(lat)
    z = _sind(lat)
    ye = y * _cosd(ecl) - z * _sind(ecl)
    ze = y * _sind(ecl) + z * _cosd(ecl)
    return Equatorial(ra=norm360(_atan2d(ye, x)), dec=_atan2d(ze, math.hypot(x, ye)), distance=dist)


# ---------------------------------------------------------------- 轨道


def _solve_kepler(m: float, e: float) -> float:
    """解开普勒方程，返回偏近点角 E（度）。"""
    big_e = m + math.degrees(e * _sind(m) * (1.0 + e * _cosd(m)))
    for _ in range(20):
        delta = (big_e - math.degrees(e * _sind(big_e)) - m) / (1.0 - e * _cosd(big_e))
        big_e -= delta
        if abs(delta) < 1e-7:
            break
    return big_e


def _orbit_position(n, i, w, a, e, m):
    """返回黄道直角坐标 (x, y, z) 和距离 r。"""
    big_e = _solve_kepler(norm360(m), e)
    xv = a * (_cosd(big_e) - e)
    yv = a * math.sqrt(1.0 - e * e) * _sind(big_e)
    v = _atan2d(yv, xv)
    r = math.hypot(xv, yv)
    x = r * (_cosd(n) * _cosd(v + w) - _sind(n) * _sind(v + w) * _cosd(i))
    y = r * (_sind(n) * _cosd(v + w) + _cosd(n) * _sind(v + w) * _cosd(i))
    z = r * _sind(v + w) * _sind(i)
    return x, y, z, r


def _sun_elements(d: float):
    w = 282.9404 + 4.70935e-5 * d
    e = 0.016709 - 1.151e-9 * d
    m = 356.0470 + 0.9856002585 * d
    return w, e, m


def _sun_ecliptic(d: float) -> tuple[float, float, float]:
    """太阳的地心黄经（度）、距离（AU）和平近点角。"""
    w, e, m = _sun_elements(d)
    x, y, _, r = _orbit_position(0.0, 0.0, w, 1.0, e, m)
    return norm360(_atan2d(y, x)), r, m


# 行星轨道根数：N, i, w, a, e, M，每项是 (常数, 每日变化)
_PLANET_ELEMENTS = {
    "mercury": ((48.3313, 3.24587e-5), (7.0047, 5.00e-8), (29.1241, 1.01444e-5),
                (0.387098, 0.0), (0.205635, 5.59e-10), (168.6562, 4.0923344368)),
    "venus": ((76.6799, 2.46590e-5), (3.3946, 2.75e-8), (54.8910, 1.38374e-5),
              (0.723330, 0.0), (0.006773, -1.302e-9), (48.0052, 1.6021302244)),
    "mars": ((49.5574, 2.11081e-5), (1.8497, -1.78e-8), (286.5016, 2.92961e-5),
             (1.523688, 0.0), (0.093405, 2.516e-9), (18.6021, 0.5240207766)),
    "jupiter": ((100.4542, 2.76854e-5), (1.3030, -1.557e-7), (273.8777, 1.64505e-5),
                (5.20256, 0.0), (0.048498, 4.469e-9), (19.8950, 0.0830853001)),
    "saturn": ((113.6634, 2.38980e-5), (2.4886, -1.081e-7), (339.3939, 2.97661e-5),
               (9.55475, 0.0), (0.055546, -9.499e-9), (316.9670, 0.0334442282)),
    "uranus": ((74.0005, 1.3978e-5), (0.7733, 1.9e-8), (96.6612, 3.0565e-5),
               (19.18171, -1.55e-8), (0.047318, 7.45e-9), (142.5905, 0.011725806)),
    "neptune": ((131.7806, 3.0173e-5), (1.7700, -2.55e-7), (272.8461, -6.027e-6),
                (30.05826, 3.313e-8), (0.008606, 2.15e-9), (260.2471, 0.005995147)),
}

# 星等公式：基准星等, 相位角系数（Schlyter）
_PLANET_MAG = {
    "mercury": -0.36, "venus": -4.34, "mars": -1.51, "jupiter": -9.25,
    "saturn": -9.0, "uranus": -7.15, "neptune": -6.90,
}

PLANETS = tuple(_PLANET_ELEMENTS)


def _elements(name: str, d: float):
    return tuple(c + rate * d for c, rate in _PLANET_ELEMENTS[name])


@dataclass(frozen=True)
class BodyPosition:
    equatorial: Equatorial
    magnitude: float
    elongation: float  # 与太阳的角距离（度）
    illumination: float = 1.0  # 被照亮比例 0–1（月亮、金星、水星有意义）


def sun_position(when: datetime) -> Equatorial:
    lon, r, _ = _sun_ecliptic(_day_number(when))
    return _ecliptic_to_equatorial(lon, 0.0, r, when)


def planet_position(name: str, when: datetime) -> BodyPosition:
    d = _day_number(when)
    n, i, w, a, e, m = _elements(name, d)
    xh, yh, zh, r = _orbit_position(n, i, w, a, e, m)

    if name in ("jupiter", "saturn", "uranus"):
        # 木星、土星、天王星之间的相互摄动
        mj = _elements("jupiter", d)[5]
        ms = _elements("saturn", d)[5]
        mu = _elements("uranus", d)[5]
        lon = _atan2d(yh, xh)
        lat = _atan2d(zh, math.hypot(xh, yh))
        if name == "jupiter":
            lon += (-0.332 * _sind(2 * mj - 5 * ms - 67.6) - 0.056 * _sind(2 * mj - 2 * ms + 21)
                    + 0.042 * _sind(3 * mj - 5 * ms + 21) - 0.036 * _sind(mj - 2 * ms)
                    + 0.022 * _cosd(mj - ms) + 0.023 * _sind(2 * mj - 3 * ms + 52)
                    - 0.016 * _sind(mj - 5 * ms - 69))
        elif name == "saturn":
            lon += (0.812 * _sind(2 * mj - 5 * ms - 67.6) - 0.229 * _cosd(2 * mj - 4 * ms - 2)
                    + 0.119 * _sind(mj - 2 * ms - 3) + 0.046 * _sind(2 * mj - 6 * ms - 69)
                    + 0.014 * _sind(mj - 3 * ms + 32))
            lat += -0.020 * _cosd(2 * mj - 4 * ms - 2) + 0.018 * _sind(2 * mj - 6 * ms - 49)
        else:
            lon += (0.040 * _sind(ms - 2 * mu + 6) + 0.035 * _sind(ms - 3 * mu + 33)
                    - 0.015 * _sind(mj - mu + 20))
        xh = r * _cosd(lon) * _cosd(lat)
        yh = r * _sind(lon) * _cosd(lat)
        zh = r * _sind(lat)

    sun_lon, sun_r, _ = _sun_ecliptic(d)
    xg = xh + sun_r * _cosd(sun_lon)
    yg = yh + sun_r * _sind(sun_lon)
    zg = zh
    dist = math.sqrt(xg * xg + yg * yg + zg * zg)
    geo_lon = _atan2d(yg, xg)
    geo_lat = _asind(zg / dist)

    # 相位角：从行星看太阳和地球的夹角
    cos_fv = (r * r + dist * dist - sun_r * sun_r) / (2 * r * dist)
    fv = math.degrees(math.acos(max(-1.0, min(1.0, cos_fv))))
    mag = _PLANET_MAG[name] + 5 * math.log10(r * dist)
    if name == "mercury":
        mag += 0.027 * fv + 2.2e-13 * fv**6
    elif name == "venus":
        mag += 0.013 * fv + 4.2e-7 * fv**3
    elif name == "mars":
        mag += 0.016 * fv
    elif name == "jupiter":
        mag += 0.014 * fv
    elif name == "saturn":
        mag += 0.044 * fv  # 未计入光环倾角，实际可能更亮 0–1 等
    else:
        mag += 0.001 * fv

    eq = _ecliptic_to_equatorial(geo_lon, geo_lat, dist, when)
    elong = angular_separation(eq.ra, eq.dec, *_radec(sun_position(when)))
    return BodyPosition(eq, mag, elong, (1 + _cosd(fv)) / 2)


def _radec(eq: Equatorial) -> tuple[float, float]:
    return eq.ra, eq.dec


def moon_position(when: datetime) -> BodyPosition:
    """月亮的地心位置。distance 单位为地球半径。"""
    d = _day_number(when)
    n = 125.1228 - 0.0529538083 * d
    i = 5.1454
    w = 318.0634 + 0.1643573223 * d
    mm = 115.3654 + 13.0649929509 * d
    x, y, z, r = _orbit_position(n, i, w, 60.2666, 0.054900, mm)
    lon = _atan2d(y, x)
    lat = _atan2d(z, math.hypot(x, y))

    sw, _, ms = _sun_elements(d)
    ls = ms + sw
    lm = mm + w + n
    dd = lm - ls
    f = lm - n
    lon += (-1.274 * _sind(mm - 2 * dd) + 0.658 * _sind(2 * dd) - 0.186 * _sind(ms)
            - 0.059 * _sind(2 * mm - 2 * dd) - 0.057 * _sind(mm - 2 * dd + ms)
            + 0.053 * _sind(mm + 2 * dd) + 0.046 * _sind(2 * dd - ms) + 0.041 * _sind(mm - ms)
            - 0.035 * _sind(dd) - 0.031 * _sind(mm + ms) - 0.015 * _sind(2 * f - 2 * dd)
            + 0.011 * _sind(mm - 4 * dd))
    lat += (-0.173 * _sind(f - 2 * dd) - 0.055 * _sind(mm - f - 2 * dd)
            - 0.046 * _sind(mm + f - 2 * dd) + 0.033 * _sind(f + 2 * dd) + 0.017 * _sind(2 * mm + f))
    r += -0.58 * _cosd(mm - 2 * dd) - 0.46 * _cosd(2 * dd)

    eq = _ecliptic_to_equatorial(norm360(lon), lat, r, when)
    sun = sun_position(when)
    elong = angular_separation(eq.ra, eq.dec, sun.ra, sun.dec)
    illum = (1 - _cosd(elong)) / 2
    # 满月约 -12.7 等，越接近新月越暗（近似）
    mag = -12.7 + 0.026 * (180 - elong) + 4e-9 * (180 - elong) ** 4
    return BodyPosition(eq, mag, elong, illum)


def equatorial_to_ecliptic(ra: float, dec: float, when: datetime) -> tuple[float, float]:
    """赤道坐标转黄道坐标 (黄经, 黄纬)，单位度，春分点取观测日期。"""
    ecl = obliquity(when)
    x = _cosd(ra) * _cosd(dec)
    y = _sind(ra) * _cosd(dec)
    z = _sind(dec)
    lon = norm360(_atan2d(y * _cosd(ecl) + z * _sind(ecl), x))
    lat = _asind(z * _cosd(ecl) - y * _sind(ecl))
    return lon, lat


def sun_longitude(when: datetime) -> float:
    """太阳的地心黄经（度，春分点取观测日期）。"""
    return _sun_ecliptic(_day_number(when))[0]


def moon_age(when: datetime) -> float:
    """月亮黄经减太阳黄经（度）：0 新月，90 上弦，180 满月，270 下弦。"""
    moon = moon_position(when).equatorial
    moon_lon, _ = equatorial_to_ecliptic(moon.ra, moon.dec, when)
    return norm360(moon_lon - sun_longitude(when))


def topocentric(eq: Equatorial, when: datetime, lat: float, lon: float) -> Equatorial:
    """把月亮的地心坐标换成站在地面上看到的坐标（Meeus 第 40 章），distance 需为地球半径。

    月亮离地球很近，北京和悉尼看到的月亮位置能差 1° 以上；算日食必须用它。
    """
    u = math.atan(0.99664719 * math.tan(math.radians(lat)))
    rho_sin = 0.99664719 * math.sin(u)
    rho_cos = math.cos(u)
    sin_par = 1.0 / eq.distance
    ha = local_sidereal_time(when, lon) - eq.ra
    denom = _cosd(eq.dec) - rho_cos * sin_par * _cosd(ha)
    d_ra = _atan2d(-rho_cos * sin_par * _sind(ha), denom)
    dec = _atan2d((_sind(eq.dec) - rho_sin * sin_par) * _cosd(d_ra), denom)
    return Equatorial(norm360(eq.ra + d_ra), dec, eq.distance)


def moon_phase(when: datetime) -> tuple[str, float]:
    """返回 (月相名称, 被照亮比例)。"""
    age = moon_age(when)
    illum = (1 - _cosd(age)) / 2
    names = ["新月", "蛾眉月", "上弦月", "盈凸月", "满月", "亏凸月", "下弦月", "残月"]
    return names[int((age + 22.5) // 45) % 8], illum


def topocentric_moon_altitude(geo_alt: float, distance_earth_radii: float) -> float:
    """月亮离地球很近，站在地面上看要扣掉视差（最多约 1°）。"""
    parallax = _asind(1.0 / distance_earth_radii)
    return geo_alt - parallax * _cosd(geo_alt)
