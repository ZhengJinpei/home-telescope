from datetime import datetime, timedelta, timezone

import pytest

from telescope import astro
from telescope.catalog import find
from telescope.sky import Observer, locate

BEIJING = Observer(39.9, 116.4)
# 参考值来自 astropy（内置星历，无大气折射），时间为 2026-10-10 21:00 北京时间
WHEN = datetime(2026, 10, 10, 21, 0, tzinfo=BEIJING.tz)


def sep(alt1, az1, alt2, az2):
    return astro.angular_separation(az1, alt1, az2, alt2)


@pytest.mark.parametrize(
    "target, alt, az",
    [
        ("土星", 37.108, 125.559),
        ("木星", -34.730, 7.919),
        ("月亮", -42.362, 296.800),
        ("太阳", -37.667, 296.825),
        ("织女星", 51.032, 285.388),
    ],
)
def test_positions_match_astropy(target, alt, az):
    s = locate(find(target), WHEN, BEIJING)
    assert sep(s.alt, s.az, alt, az) < 0.1  # 6 角分以内


@pytest.mark.parametrize(
    "day, illumination",
    [("2026-10-10", 0.001), ("2026-10-18", 0.499), ("2026-10-26", 0.995), ("2026-11-01", 0.521)],
)
def test_moon_illumination(day, illumination):
    when = datetime.fromisoformat(day).replace(hour=16, tzinfo=timezone.utc)
    _, illum = astro.moon_phase(when)
    assert illum == pytest.approx(illumination, abs=0.02)


def test_moon_phase_names():
    assert astro.moon_phase(datetime(2026, 10, 10, 16, tzinfo=timezone.utc))[0] == "新月"
    assert astro.moon_phase(datetime(2026, 10, 18, 16, tzinfo=timezone.utc))[0] == "上弦月"
    assert astro.moon_phase(datetime(2026, 10, 26, 16, tzinfo=timezone.utc))[0] == "满月"
    assert astro.moon_phase(datetime(2026, 11, 2, 16, tzinfo=timezone.utc))[0] == "下弦月"


def test_polaris_altitude_equals_latitude():
    # 北极星的高度约等于当地纬度，这是认北极星的好方法
    for hour in range(0, 24, 3):
        s = locate(find("北极星"), WHEN + timedelta(hours=hour), BEIJING)
        assert s.alt == pytest.approx(BEIJING.lat, abs=1.0)


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        astro.julian_date(datetime(2026, 1, 1))


def test_formatting():
    assert astro.hms(12.5) == "12h30m00s"
    assert astro.dms(-16.7161) == "-16°42′58″"
    assert astro.compass(0) == "北"
    assert astro.compass(350) == "北"
    assert astro.compass(135) == "东南"


def test_refraction_near_horizon():
    assert astro.refraction(0) == pytest.approx(0.57, abs=0.02)  # 约 34 角分：看到太阳落山时它其实已经在地平线下
    assert astro.refraction(45) == pytest.approx(1 / 60, abs=0.005)
    assert astro.refraction(-5) == 0.0
