import math
import struct

import pytest

from telescope import spotting as sp
from telescope import stl
from telescope.cli import main


@pytest.fixture(scope="module")
def design():
    return sp.design_spotting()


def test_eyepiece_table(design):
    rows = {r.eyepiece.focal_mm: r for r in design.rows}
    assert rows[40].magnification == 12 and rows[32].magnification == 15
    assert rows[8].magnification == 60
    assert rows[40].exit_pupil_mm == pytest.approx(6.67, abs=0.01)
    # 1.25 寸目镜视场光阑最大 27 mm：480 mm 焦距下真视场约 3.2°
    assert rows[32].true_fov_deg == pytest.approx(math.degrees(27 / 480), abs=0.01)
    assert rows[32].width_at_100m == pytest.approx(5.6, abs=0.1)
    # 黄昏系数 √(口径 × 倍率)
    assert rows[32].twilight_factor == pytest.approx(math.sqrt(80 * 15))
    assert any("傍晚" in n for n in rows[40].notes)
    assert not any("傍晚" in n for n in rows[8].notes)


def test_dusk_brightness():
    # 出瞳比瞳孔大：和肉眼一样亮（只扣透过率）；出瞳小一半：亮度剩四分之一
    assert sp.relative_brightness(7, 6) == pytest.approx(sp.TRANSMISSION)
    assert sp.relative_brightness(3, 6) == pytest.approx(sp.TRANSMISSION / 4)


def test_focus(design):
    # 薄透镜：看 10 m 外，480 mm 物镜的焦点后移约 24 mm
    assert sp.focus_extension_mm(480, 10) == pytest.approx(24.2, abs=0.1)
    assert sp.nearest_focus_m(480, sp.focus_extension_mm(480, 8)) == pytest.approx(8)
    assert sp.nearest_focus_m(480, 0) == math.inf
    assert 4 < design.min_focus_m < 8


def test_tube_length_adds_up(design):
    s = design.spec
    total = s.cell_rear_mm + design.tube_length_mm + s.focuser_length_mm + s.focus_margin_mm + s.prism_path_mm
    assert total == pytest.approx(s.focal_length_mm)
    assert not design.warnings


def test_prism_size_controls_edge_light():
    big = sp.design_spotting()
    path, clear, to_focus = sp.PRISMS["1.25"]
    small = sp.design_spotting(sp.SpottingSpec(prism_path_mm=path, prism_clear_mm=clear, prism_to_focus_mm=to_focus))
    assert all(r.edge_light > 0.95 for r in big.rows)
    low_power = small.rows[0]
    assert low_power.edge_light < 0.7 and any("2 寸" in n for n in low_power.notes)
    assert small.rows[-1].edge_light > 0.95  # 高倍视场小，不受影响


def test_baffles_shrink_toward_focus(design):
    holes = [b.hole_mm for b in design.baffles]
    assert len(holes) == 3 and holes == sorted(holes, reverse=True)
    for b in design.baffles:
        cone = sp.light_cone_mm(80, 480, design.spec.cell_rear_mm + b.position_mm)
        assert b.hole_mm > cone  # 不挡成像的光


def test_chromatic_index():
    assert sp.chromatic_index(80, 480) == pytest.approx(1.905, abs=0.01)
    achro = sp.design_spotting(sp.SpottingSpec(ed_glass=False))
    assert "紫边" in achro.color_note and "ED" in achro.color_note


def test_bad_specs():
    with pytest.raises(ValueError):
        sp.SpottingSpec(aperture_mm=80, focal_length_mm=300)  # f/3.75
    with pytest.raises(ValueError):
        sp.SpottingSpec(aperture_mm=90, tube_od_mm=90)


def test_balance_is_toward_rear(design):
    assert design.tube_length_mm / 2 < design.balance_from_cell_mm < design.tube_length_mm + 100
    assert 1.0 < design.total_kg < 3.0


# ---------------------------------------------------------------- STL


@pytest.mark.parametrize("profile", [
    stl.tube_profile(43, 45, 120),
    stl.sleeve_with_baffle(41.8, 43, 70, 30, 2),
    stl.cap_profile(47, 2, 15, 2),
])
def test_revolve_is_closed_with_right_volume(profile):
    mesh = stl.revolve(profile)
    assert mesh.is_watertight()
    # 128 边形比真圆小 0.04%
    assert mesh.volume() == pytest.approx(stl.profile_volume(profile), rel=0.001)


def test_revolve_rejects_bad_profiles():
    with pytest.raises(ValueError):
        stl.revolve([(0, 0), (1, 0)])
    with pytest.raises(ValueError):
        stl.revolve([(-1, 0), (1, 0), (1, 1)])


def test_profile_volume_matches_formula():
    assert stl.profile_volume(stl.tube_profile(10, 20, 5)) == pytest.approx(math.pi * (400 - 100) * 5)


def test_write_stl(tmp_path):
    mesh = stl.revolve(stl.tube_profile(10, 12, 20), segments=16)
    path = tmp_path / "a.stl"
    stl.write_stl(mesh, path, "test")
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    assert count == len(mesh.triangles) == 16 * 4 * 2
    assert len(data) == 84 + 50 * count


def test_printed_parts_fit_together(design):
    parts = sp.printed_parts(design)
    names = [p.name for p in parts]
    assert names[:2] == ["遮光罩", "镜头盖"] and len(parts) == 2 + len(design.baffles) + 1
    # 内衬筒加起来正好是镜筒长度
    liners = [p for p in parts if p.filename.startswith("liner")]
    assert sum(max(z for _, z in p.profile) for p in liners) == pytest.approx(design.tube_length_mm)
    tube_id = design.spec.tube_od_mm - 2 * design.spec.tube_wall_mm
    for p in liners:
        assert max(r for r, _ in p.profile) < tube_id / 2
        assert stl.revolve(p.profile).is_watertight()
    shield = stl.revolve(parts[0].profile)
    assert shield.is_watertight()


def test_cli_spotting(tmp_path, capsys):
    main(["spotting", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "12×" in out and "60×" in out
    assert "物镜外协要求" in out and "ED" in out
    assert "切成 285 mm 长" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "dew_shield.stl", "lens_cap.stl", "liner_1.stl", "liner_2.stl", "liner_3.stl", "liner_4.stl"]


def test_cli_spotting_small_prism(capsys):
    main(["spotting", "--prism", "1.25", "--achromat"])
    out = capsys.readouterr().out
    assert "视场边缘亮度" in out
    assert "紫边" in out
