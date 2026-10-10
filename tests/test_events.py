from datetime import date, datetime, timedelta, timezone

import pytest

from telescope import astro
from telescope.events import METEOR_SHOWERS, find_events, highlights
from telescope.sky import Observer, observer_for_city

BEIJING = observer_for_city("北京")
SYDNEY = Observer(-33.87, 151.21, 11)  # 3 月是夏令时


@pytest.fixture(scope="module")
def year_2026():
    return find_events(date(2026, 1, 1), 365, BEIJING)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def near(t: datetime, ref: datetime, minutes: float) -> bool:
    return abs((t - ref).total_seconds()) <= minutes * 60


def only(events, kind):
    return [e for e in events if e.kind == kind]


def test_moon_phases_match_published_times():
    phases = {e.title: e.when for e in only(find_events(date(2026, 3, 1), 20, BEIJING), "月相")}
    assert near(phases["满月"], utc(2026, 3, 3, 11, 38), 10)
    assert near(phases["下弦月"], utc(2026, 3, 11, 9, 38), 10)
    assert near(phases["新月"], utc(2026, 3, 19, 1, 23), 10)


def test_total_lunar_eclipse_2026_03_03():
    # 参考：NASA 月食表，食甚 11:33 UT，本影食分 1.15，北京月出时带食
    (e,) = only(find_events(date(2026, 3, 2), 3, BEIJING), "月食")
    assert e.title.startswith("月全食")
    assert e.visible and e.importance == 3
    assert near(e.when, utc(2026, 3, 3, 11, 33), 10)
    assert "食既" in e.view and "生光" in e.view
    (sydney,) = only(find_events(date(2026, 3, 2), 3, SYDNEY), "月食")
    assert sydney.visible


def test_partial_lunar_eclipse_below_horizon_in_beijing():
    # 2026-08-28 月偏食发生在北京时间中午，月亮在地平线下
    (e,) = only(find_events(date(2026, 8, 27), 3, BEIJING), "月食")
    assert e.title.startswith("月偏食")
    assert not e.visible


def test_solar_eclipses():
    # astropy：2030-06-01 北京日偏食，食甚 15:46，食分 0.68
    (e,) = only(find_events(date(2030, 5, 31), 3, BEIJING), "日食")
    assert e.visible and e.title.startswith("日偏食")
    assert near(e.when, datetime(2030, 6, 1, 15, 46, tzinfo=BEIJING.tz), 5)
    assert float(e.title.split("食分 ")[1].rstrip("）")) == pytest.approx(0.68, abs=0.03)
    assert "日食眼镜" in e.detail

    # 2027-08-02 日全食在北非和中东，中国看不到
    (e,) = only(find_events(date(2027, 8, 1), 3, BEIJING), "日食")
    assert not e.visible

    # 2035-09-02 北京在全食带边缘：要提醒可能是全食
    (e,) = only(find_events(date(2035, 9, 1), 3, BEIJING), "日食")
    assert e.visible and "全食" in e.title + e.detail


def test_no_eclipse_on_ordinary_full_moon():
    assert not only(find_events(date(2026, 10, 20), 10, BEIJING), "月食")


def test_oppositions_2026(year_2026):
    events = only(year_2026, "冲日")
    dates = {e.title[:3]: e.when.date() for e in events}
    assert dates["木星冲"] == date(2026, 1, 10)
    assert dates["土星冲"] == date(2026, 10, 4)
    assert dates["海王星"] == date(2026, 9, 26)
    assert "火星冲" not in dates  # 火星下一次冲日在 2027 年 2 月


def test_venus_greatest_elongation():
    events = only(find_events(date(2026, 8, 1), 30, BEIJING), "大距")
    venus = next(e for e in events if e.title.startswith("金星"))
    assert venus.title == "金星东大距"
    assert abs(venus.when.date() - date(2026, 8, 15)) <= timedelta(days=1)
    assert "西方" in venus.view  # 黄昏后在西方


def test_occultation_of_regulus():
    # astropy：2026-03-02 北京 19:05 轩辕十四被月亮挡住，19:59 重新出现
    (e,) = only(find_events(date(2026, 3, 2), 1, BEIJING), "月掩星")
    assert e.title == "月掩轩辕十四"
    assert near(e.when, datetime(2026, 3, 2, 19, 5, tzinfo=BEIJING.tz), 5)


def test_venus_jupiter_conjunction():
    events = only(find_events(date(2026, 6, 5), 10, BEIJING), "相合")
    (e,) = [e for e in events if "金星" in e.title and "木星" in e.title]
    assert e.visible and e.importance >= 2
    assert "西方" in e.view


def test_moon_planet_conjunction_view_uses_local_separation():
    # astropy：2026-11-03 05:47 北京看月亮和木星相距 0.78°
    events = only(find_events(date(2026, 11, 2), 3, BEIJING), "合月")
    (e,) = [e for e in events if "木星" in e.title]
    assert "0.8°" in e.title


def test_meteor_showers(year_2026):
    events = only(year_2026, "流星雨")
    assert len(events) == len(METEOR_SHOWERS)
    peaks = {e.title.removesuffix("极大"): e for e in events}
    assert peaks["双子座流星雨"].when.date() == date(2026, 12, 14)
    perseids = peaks["英仙座流星雨"]
    assert perseids.when.date() == date(2026, 8, 13)
    assert "没有月光" in perseids.view  # 2026 年英仙座流星雨正好赶上新月
    assert perseids.importance == 3
    assert "月光很亮" in peaks["象限仪座流星雨"].view  # 赶上满月


def test_events_sorted_and_in_range():
    start = date(2026, 10, 1)
    events = find_events(start, 31, BEIJING)
    times = [e.when for e in events]
    assert times == sorted(times)
    lo = datetime(2026, 10, 1, tzinfo=BEIJING.tz)
    assert all(lo <= t < lo + timedelta(days=31) for t in times)
    assert len(only(events, "月相")) == 4
    assert all(e.visible and e.importance >= 2 for e in highlights(events))


def test_topocentric_moon():
    when = utc(2026, 10, 20, 12)
    geo = astro.moon_position(when).equatorial
    topo = astro.topocentric(geo, when, BEIJING.lat, BEIJING.lon)
    shift = astro.angular_separation(geo.ra, geo.dec, topo.ra, topo.dec)
    assert 0.3 < shift < 1.0  # 月亮视差最多约 1°
    h_geo = astro.to_horizontal(geo.ra, geo.dec, when, BEIJING.lat, BEIJING.lon)
    h_topo = astro.to_horizontal(topo.ra, topo.dec, when, BEIJING.lat, BEIJING.lon)
    assert h_topo.alt == pytest.approx(astro.topocentric_moon_altitude(h_geo.alt, geo.distance), abs=0.05)
