import pytest

from telescope import optics

SCOPE = optics.Telescope(70, 700)  # 常见的 70 mm 入门折射镜


def test_basic_numbers():
    assert SCOPE.focal_ratio == 10
    assert SCOPE.max_useful_magnification == 140
    assert SCOPE.min_useful_magnification == 10
    assert SCOPE.dawes_limit_arcsec == pytest.approx(1.657, abs=0.01)
    assert SCOPE.light_gathering_vs_eye == 100
    assert SCOPE.limiting_magnitude == pytest.approx(11.9, abs=0.1)


def test_seeing_caps_big_scopes():
    assert optics.Telescope(300, 1500).max_useful_magnification == optics.SEEING_LIMIT_X


def test_eyepiece_view():
    v = optics.view(SCOPE, optics.Eyepiece(25))
    assert v.magnification == 28
    assert v.true_fov_deg == pytest.approx(52 / 28)
    assert v.exit_pupil_mm == 2.5
    assert v.moon_fits
    assert v.warnings == []


def test_barlow_and_over_magnification_warning():
    v = optics.view(SCOPE, optics.Eyepiece(4, barlow=2))
    assert v.magnification == 350
    assert not v.moon_fits
    assert any("有效上限" in w for w in v.warnings)


def test_wasted_light_warning():
    v = optics.view(optics.Telescope(200, 1000), optics.Eyepiece(40))
    assert v.exit_pupil_mm == 8
    assert any("瞳孔" in w for w in v.warnings)


def test_suggestions_are_buyable():
    s = optics.suggest_eyepieces(SCOPE)
    low, mid, high = s.values()
    assert low <= 32
    assert low > mid > high > 0


def test_camera_view():
    cv = optics.camera_view(SCOPE, optics.PI_HQ_CAMERA)
    assert cv.arcsec_per_pixel == pytest.approx(0.457, abs=0.01)
    assert cv.fov_width_deg == pytest.approx(0.51, abs=0.01)
    assert not cv.moon_fits
    # 短焦镜头能拍下整个月亮
    assert optics.camera_view(optics.Telescope(50, 300), optics.PI_HQ_CAMERA).moon_fits


def test_invalid_scope():
    with pytest.raises(ValueError):
        optics.Telescope(0, 700)
