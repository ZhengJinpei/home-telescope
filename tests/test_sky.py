from datetime import date

import pytest

from telescope.catalog import ALL_TARGETS, find
from telescope.optics import Telescope
from telescope.sky import Observer, observer_for_city, plan_tonight

BEIJING = observer_for_city("北京")


def test_unknown_city():
    with pytest.raises(KeyError):
        observer_for_city("火星基地")


def test_find_by_name_key_alias():
    assert find("土星").key == "saturn"
    assert find("Saturn").key == "saturn"
    assert find("启明星").key == "venus"
    assert find("m42").name == "猎户座大星云"
    assert find("M 042").name == "猎户座大星云"
    with pytest.raises(KeyError):
        find("M99")


def test_catalog_is_complete():
    keys = [t.key for t in ALL_TARGETS]
    assert len(keys) == len(set(keys))
    for t in ALL_TARGETS:
        assert t.story
        if not t.solar_system:
            assert 0 <= t.ra < 360 and -90 <= t.dec <= 90


def test_tonight_october_beijing():
    plan = plan_tonight(date(2026, 10, 10), BEIJING, Telescope(70, 700))
    names = [i.target.name for i in plan.items]
    assert plan.moon_phase == "新月"
    assert "太阳" not in names
    assert "土星" in names  # 2026 年秋天土星整晚可见
    assert "月亮" not in names  # 新月看不到
    assert plan.dusk < plan.dark_start < plan.dark_end < plan.dawn
    for item in plan.items:
        assert item.best_alt >= 15
        assert 1 <= item.difficulty <= 3
    # 行星排在恒星前面
    assert names.index("土星") < names.index("织女星")


def test_full_moon_warns_about_deep_sky():
    plan = plan_tonight(date(2026, 10, 26), BEIJING, Telescope(70, 700))
    moon = next(i for i in plan.items if i.target.key == "moon")
    assert any("满月" in n for n in moon.notes)
    assert any("月光" in n for i in plan.items for n in i.notes if i.target.kind in ("星云", "星系"))


def test_small_scope_skips_faint_targets():
    eye = plan_tonight(date(2026, 10, 10), BEIJING)  # 不给望远镜参数：按肉眼加光污染
    names = {i.target.key for i in eye.items}
    assert "M57" not in names
    assert "neptune" not in names
    assert "saturn" in names


def test_southern_hemisphere():
    sydney = Observer(-33.87, 151.21, 10)
    plan = plan_tonight(date(2026, 10, 10), sydney)
    names = {i.target.key for i in plan.items}
    assert "polaris" not in names
    assert "fomalhaut" in names
