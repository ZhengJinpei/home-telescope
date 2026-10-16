import pytest

from telescope import mirror
from telescope.cli import main


def test_primary_150_f8():
    p = mirror.PrimarySpec(150, 1200, 25)
    assert p.radius_of_curvature_mm == 2400
    assert p.focal_ratio == 8
    assert p.sagitta_mm == pytest.approx(1.172, abs=0.001)  # 75² / (4 × 1200)
    assert p.parabolizing_nm == pytest.approx(72, abs=1)
    assert p.mass_kg == pytest.approx(1.1, abs=0.05)


def test_sphere_vs_paraboloid():
    # 球面代替抛物面：焦比越大越接近；150 mm 到 f/8 附近才达到衍射极限
    pv8, _, s8 = mirror.sphere_error(150, 1200)
    _, _, s10 = mirror.sphere_error(150, 1500)
    assert pv8 == pytest.approx(0.26, abs=0.01)
    assert s8 < 0.8 < s10
    assert mirror.min_focal_ratio_for_sphere(150) == pytest.approx(8.1, abs=0.1)
    assert mirror.min_focal_ratio_for_sphere(200) > mirror.min_focal_ratio_for_sphere(150)
    _, _, s200 = mirror.sphere_error(200, 1200)
    assert s200 < 0.5  # 200 mm f/6 做成球面完全不能用


def test_foucault_zones():
    zones = mirror.foucault_zones(150, 1200)
    assert len(zones) == 4
    assert zones[0].inner_mm == 0 and zones[-1].outer_mm == pytest.approx(75)
    # 等面积分带
    areas = [z.outer_mm ** 2 - z.inner_mm ** 2 for z in zones]
    assert max(areas) == pytest.approx(min(areas))
    edge = zones[-1]
    assert edge.reading_mm == pytest.approx(((65.0 + 75) / 2) ** 2 / 2400, abs=0.01)
    assert [z.reading_mm for z in zones] == sorted(z.reading_mm for z in zones)


def test_secondary():
    s = mirror.SecondarySpec(35)
    assert s.major_axis_mm == pytest.approx(49.5, abs=0.05)
    assert s.thickness_mm >= 35 / 6


def test_cli_mirror(capsys):
    main(["mirror"])
    out = capsys.readouterr().out
    assert "抛物面" in out and "K = -1" in out
    assert "凹深 1.172 mm" in out
    assert "短轴 35 mm，长轴 49.5 mm" in out
    assert "建议修抛物面" in out
