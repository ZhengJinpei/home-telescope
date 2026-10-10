import pytest

from telescope.cli import main


def run(capsys, *argv):
    main(list(argv))
    return capsys.readouterr().out


def test_optics(capsys):
    out = run(capsys, "optics", "--aperture", "70", "--focal", "700", "--eyepiece", "25", "4", "--camera", "hq")
    assert "28 倍" in out
    assert "有效上限" in out
    assert "树莓派 HQ" in out


def test_tonight(capsys):
    out = run(capsys, "tonight", "--city", "上海", "--date", "2026-10-10", "--top", "5")
    assert "土星" in out
    assert "--top 0" in out


def test_where(capsys):
    out = run(capsys, "where", "土星", "--time", "2026-10-10 21:00")
    assert "东南" in out
    assert "光环" in out


def test_goto_refuses_sun(capsys):
    with pytest.raises(SystemExit) as e:
        main(["goto", "太阳"])
    assert e.value.code == 1
    assert "太阳" in capsys.readouterr().out


def test_unknown_target(capsys):
    with pytest.raises(SystemExit) as e:
        main(["where", "不存在的星"])
    assert e.value.code == 2
    assert "找不到" in capsys.readouterr().out


def test_list(capsys):
    out = run(capsys, "list")
    assert "行星" in out and "星云" in out
