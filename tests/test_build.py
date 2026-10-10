import math

import pytest

from telescope import build
from telescope.cli import main
from telescope.mount import AltAzMount, SimulatedDriver
from telescope.sky import observer_for_city


@pytest.fixture(scope="module")
def newt():
    return build.design_newtonian()


@pytest.fixture(scope="module")
def dob(newt):
    return build.design_dobsonian(newt)


def test_focus_reachable_by_camera_and_eyepieces(newt):
    travel = newt.spec.focuser_travel_mm
    assert 0 <= newt.camera_racked_out <= travel
    lo, hi = newt.eyepiece_racked_out
    assert 0 <= lo <= hi <= travel
    assert not newt.warnings


def test_short_focuser_warns():
    n = build.design_newtonian(build.NewtonianSpec(focuser_travel_mm=15))
    assert any("延长筒" in w for w in n.warnings)


def test_secondary_size(newt):
    # 经典公式：短轴 = 照明视场 + (口径 - 照明视场) × 副镜到焦点距离 / 焦距
    s = newt.spec
    field_mm = build.MIN_FULL_FIELD_MM
    expected = field_mm + (s.aperture_mm - field_mm) * newt.secondary_to_focus / s.focal_length_mm
    assert newt.secondary_min_needed == pytest.approx(expected)
    assert newt.secondary_minor_axis in build.SECONDARY_SIZES
    assert newt.secondary_minor_axis >= newt.secondary_min_needed
    assert newt.fully_illuminated_mm >= field_mm
    assert newt.corner_illumination >= build.MIN_CORNER_LIGHT
    assert newt.obstruction < 0.3
    # f/8 的副镜偏移约 1 mm
    assert newt.secondary_offset == pytest.approx(1.1, abs=0.1)


def test_optical_path_adds_up(newt):
    # 主镜到副镜 + 副镜到焦点 = 焦距
    s = newt.spec
    assert (newt.focuser_pos - newt.mirror_face_pos) + newt.secondary_to_focus == pytest.approx(s.focal_length_mm)
    assert newt.focuser_pos < newt.tube_length
    assert 0 < newt.balance_pos < newt.tube_length / 2  # 主镜在尾端，重心偏后
    assert newt.mirror_kg == pytest.approx(1.1, abs=0.1)  # 150 mm × 25 mm 玻璃


def test_bad_specs_rejected():
    with pytest.raises(ValueError):
        build.NewtonianSpec(aperture_mm=150, focal_length_mm=400)  # f/2.7
    with pytest.raises(ValueError):
        build.NewtonianSpec(aperture_mm=200, tube_od_mm=200)  # 镜筒太细


def test_dobsonian_clearances(newt, dob):
    od = newt.spec.tube_od_mm
    # 托架板转起来不超出侧轴承圆盘
    assert math.hypot(dob.cradle_length / 2, dob.cradle_height) < dob.bearing_radius
    # 镜筒尾端转到最低处也碰不到底板
    assert dob.pivot_height - math.hypot(newt.balance_pos, od / 2) >= 25
    # 指向地平线时镜筒在前板上方
    assert dob.front_board_height < dob.pivot_height - od / 2
    # 摇臂箱侧板的半圆缺口两边还有木头
    assert dob.rocker_side_width > 2 * (dob.bearing_radius + build.PAD_MM) + 50
    assert dob.cradle_width - od >= 40
    # 侧轴承圆盘和摇臂箱侧板在同一平面，圆盘边缘才能压在垫片上
    assert dob.rocker_inner_width - dob.cradle_width <= 4
    # 摇臂箱转到任何角度，底板都压得住地板上的三块垫片
    assert dob.az_pad_radius + 15 < min(dob.floor_width, dob.rocker_side_width) / 2
    assert 700 < dob.eyepiece_height_horizon < dob.eyepiece_height_zenith < 1400


def test_drive_steps_feed_the_mount(dob):
    # GT2 同步带绕在 150 mm 半径的驱动盘上，20 齿轮：约 23.6 : 1
    assert build.belt_steps_per_rev(150) == round(200 * 16 * 2 * math.pi * 150 / 40)
    assert dob.alt_steps_per_rev == build.belt_steps_per_rev(dob.bearing_radius - build.DRIVE_INSET_MM)
    assert 1296000 / dob.alt_steps_per_rev < 30  # 每步不到 30 角秒，低倍视场里看不出来
    mount = AltAzMount(SimulatedDriver(), observer_for_city("北京"),
                       alt_steps_per_rev=dob.alt_steps_per_rev, az_steps_per_rev=dob.az_steps_per_rev)
    mount.sync(45.0, 180.0)
    assert mount.alt == pytest.approx(45.0, abs=0.01)


def _closed(cuts) -> bool:
    """首尾相连的线段和圆弧围成封闭轮廓（CNC 才能切）。"""
    pts = []
    for c in cuts:
        if c[0] == "line":
            pts.append(((c[1], c[2]), (c[3], c[4])))
        elif c[0] == "arc":
            _, cx, cy, r, a0, a1 = c
            # DXF 圆弧逆时针画；这里只记录端点，方向无关
            p0 = (cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)))
            p1 = (cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)))
            pts.append((p0, p1))
    ends = [p for seg in pts for p in seg]
    for p in ends:
        if sum(math.dist(p, q) < 1e-6 for q in ends) % 2:
            return False
    return True


def test_cut_parts_are_closed_and_fit(newt, dob):
    parts = build.cut_parts(newt, dob)
    assert sum(p.count for p in parts) == 12
    for p in parts:
        outline = [c for c in p.cuts if c[0] != "circle"]
        assert _closed(outline), p.name
        assert p.width < 1220 and p.height < 1220  # 一张 1220 × 2440 的多层板放得下


def test_dxf_and_svg(tmp_path, newt, dob):
    parts = build.cut_parts(newt, dob)
    build.write_dxf(parts, tmp_path / "a.dxf")
    build.write_svg(parts, tmp_path / "a.svg", 18)
    dxf = (tmp_path / "a.dxf").read_text(encoding="gbk")
    assert dxf.startswith("0\nSECTION")
    assert dxf.rstrip().endswith("EOF")
    # 底座：2 个圆盘 + 驱动盘 + 地板 + 2 个转轴孔 + 3 个定位孔
    # 主镜座：后板 + 3 螺丝孔 + 3 通风孔；托板 + 中心孔 + 3 螺母孔 + 3 软垫定位孔
    assert dxf.count("\nCIRCLE\n") == 9 + 3 + 7 + 8  # 地板还有 3 个垫片定位孔
    assert dxf.count("\nARC\n") == 4
    assert "托架板" in dxf
    svg = (tmp_path / "a.svg").read_text(encoding="utf-8")
    assert svg.count("<path") == 4 and "<svg" in svg


def test_cli_build(tmp_path, capsys):
    main(["build", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "副镜：短轴 35 mm" in out
    assert "切割清单" in out
    assert (tmp_path / "telescope_150mm.dxf").exists()


def test_mirror_cell_fits_tube(newt):
    back, front = build.cell_parts(newt)
    tube_id = newt.spec.tube_od_mm - 2 * newt.spec.tube_wall_mm
    assert tube_id - 2 < back.width < tube_id
    assert front.width < newt.spec.aperture_mm  # 托板比主镜小，不挡侧面的镜夹
    # 校准螺丝孔在两块板上位置一致
    bolt_holes = lambda p: sorted((round(c[1] - p.width / 2, 3), round(c[2] - p.height / 2, 3))
                                  for c in p.cuts if c[0] == "circle" and c[3] in (3.5, 5.0))
    assert bolt_holes(back) == bolt_holes(front)
    assert build.cell_depth(18) == 50


def test_dslr_design():
    from telescope.optics import CANON_80D

    spec = build.NewtonianSpec(aperture_mm=200, focal_length_mm=1200, mirror_thickness_mm=33,
                               tube_od_mm=250, tube_wall_mm=4.9, camera=CANON_80D)
    n = build.design_newtonian(spec)
    # 单反后截距 55 mm 以上，焦点要比树莓派相机往外推很多；目镜靠延长筒对焦
    assert n.focal_plane_above_drawtube > 60
    assert build.CAMERA_MARGIN_MM <= n.camera_racked_out <= spec.focuser_travel_mm
    assert n.extension_tube_mm in build.EXTENSION_TUBES
    lo, hi = n.eyepiece_racked_out
    assert 0 <= lo <= hi <= spec.focuser_travel_mm
    # 副镜要大一号才能照亮 APS-C 画面四角
    hq = build.design_newtonian(build.NewtonianSpec(aperture_mm=200, focal_length_mm=1200, mirror_thickness_mm=33,
                                                    tube_od_mm=250, tube_wall_mm=4.9))
    assert n.secondary_minor_axis > hq.secondary_minor_axis
    assert n.corner_illumination >= build.MIN_CORNER_LIGHT
    assert n.obstruction <= 0.3
    # f/6 配 APS-C：四角彗差明显，中间只有约 3 mm 没有彗差
    assert n.coma_free_mm == pytest.approx(3.1, abs=0.1)
    assert n.corner_coma_arcsec > 10
    # 相机比目镜重，重心往前移
    assert n.balance_camera_pos > n.balance_pos
    dob = build.design_dobsonian(n)
    assert dob.pivot_height - math.hypot(n.balance_camera_pos, spec.tube_od_mm / 2) >= 25


def test_overlap_fraction():
    assert build._overlap_fraction(5, 10, 0) == 1.0
    assert build._overlap_fraction(5, 10, 20) == 0.0
    assert build._overlap_fraction(5, 5, 0) == pytest.approx(1.0)
    half = build._overlap_fraction(5, 1000, 1000)  # 大圆边缘近似直线：一半在里面
    assert half == pytest.approx(0.5, abs=0.01)


def test_camera_without_removable_lens_rejected():
    from telescope.optics import PI_CAMERA_3

    with pytest.raises(ValueError):
        build.NewtonianSpec(camera=PI_CAMERA_3)


def test_cli_build_dslr(capsys):
    main(["build", "--aperture", "200", "--focal", "1200", "--thickness", "33", "--tube-od", "250",
          "--tube-wall", "4.9", "--tube-kg", "5.5", "--camera", "80d"])
    out = capsys.readouterr().out
    assert "50 mm 的 2 寸延长筒" in out
    assert "彗差改正镜" in out
    assert "80D" in out
