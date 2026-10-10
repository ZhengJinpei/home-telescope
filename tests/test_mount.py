from datetime import datetime, timedelta

import pytest

from telescope.catalog import find
from telescope.mount import AltAzMount, SimulatedDriver, UnsafeTarget
from telescope.sky import locate, observer_for_city, sun_altitude

BEIJING = observer_for_city("北京")
NIGHT = datetime(2026, 10, 10, 21, 0, tzinfo=BEIJING.tz)
NOON = datetime(2026, 10, 10, 12, 0, tzinfo=BEIJING.tz)


def make_mount():
    driver = SimulatedDriver()
    return AltAzMount(driver, BEIJING), driver


def test_goto_points_at_target():
    mount, driver = make_mount()
    plan = mount.goto("土星", NIGHT)
    s = locate(find("土星"), NIGHT, BEIJING)
    step = 360 / mount.alt_spr
    assert mount.alt == pytest.approx(s.alt, abs=0.1)  # 含折射，约 1 角分
    assert mount.az == pytest.approx(s.az, abs=step)
    assert driver.log == [("az", plan.az_steps), ("alt", plan.alt_steps)]


def test_az_takes_shortest_way():
    mount, _ = make_mount()
    mount.sync(30, 350)
    plan = mount.plan("北极星", NIGHT)  # 北极星在方位 0° 附近
    assert 0 < plan.az_steps < mount.az_spr / 18  # 顺时针转十来度，不应该倒着绕一大圈


def test_refuses_sun():
    mount, driver = make_mount()
    with pytest.raises(UnsafeTarget, match="太阳"):
        mount.goto("太阳", NOON)
    assert driver.log == []


def test_refuses_below_horizon():
    mount, driver = make_mount()
    with pytest.raises(UnsafeTarget, match="地平线"):
        mount.goto("木星", NIGHT)  # 21:00 木星还没升起
    assert driver.log == []


def test_refuses_target_near_sun():
    # 水星离太阳最远也只有 28° 左右，避让角设成 30° 时白天一定拒绝
    mount = AltAzMount(SimulatedDriver(), BEIJING, sun_avoid_deg=30)
    assert sun_altitude(NOON, BEIJING) > 0
    with pytest.raises(UnsafeTarget, match="离太阳只有"):
        mount.plan("水星", NOON)


def test_refuses_path_through_sun():
    mount, _ = make_mount()
    when = NOON
    sun = locate(find("太阳"), when, BEIJING)
    # 镜筒放在太阳一侧 30°，目标在另一侧：直线转过去会经过太阳
    mount.sync(sun.alt, sun.az - 30)
    other = next(
        t for t in ("织女星", "天津四", "牛郎星", "大角星", "木星", "火星", "土星")
        if (s := locate(find(t), when, BEIJING)).alt > 15
        and 30 < ((s.az - sun.az) % 360) < 120
        and abs(s.alt - sun.alt) < 15
    )
    with pytest.raises(UnsafeTarget, match="扫过太阳"):
        mount.plan(other, when)


def test_daytime_warning():
    mount, _ = make_mount()
    plan = mount.plan("木星", NOON)  # 中午木星在西方天空，离太阳足够远
    assert any("盖子" in w for w in plan.warnings)


def test_align_then_goto():
    mount, _ = make_mount()
    mount.align("织女星", NIGHT)
    vega = locate(find("织女星"), NIGHT, BEIJING)
    assert mount.az == pytest.approx(vega.az, abs=0.05)
    plan = mount.goto("天津四", NIGHT)
    assert abs(plan.az_steps) < mount.az_spr / 4  # 两颗星都在夏季大三角里，离得不远


def test_tracking_follows_sky():
    mount, driver = make_mount()
    plans = mount.track("土星", NIGHT, minutes=10, interval_s=60)
    assert len(plans) == 11
    # 第一次是大幅转动，之后每分钟只是小修正（恒星每分钟最多移动 15 角分）
    corrections = [abs(p.az_steps) + abs(p.alt_steps) for p in plans[1:]]
    assert max(corrections) < mount.az_spr * 0.5 / 360
    end = locate(find("土星"), NIGHT + timedelta(minutes=10), BEIJING)
    assert mount.az == pytest.approx(end.az, abs=0.05)
