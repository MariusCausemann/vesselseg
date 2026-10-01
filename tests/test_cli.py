import os

import pytest

from vesselseg import cli, hfm, pipeline


def test_to_argv():
    argv = pipeline.to_argv({"workers": 64, "t2_only": True, "verbose": False, "lns_box": [20, 40]})
    assert argv == ["--workers", "64", "--t2_only", "--lns_box", "20", "40"]
    assert pipeline.to_argv(None) == []


def _exe(d):
    d.mkdir(parents=True)
    exe = d / hfm.EXE
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    return exe


def test_find_hfm_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VESSELSEG_HFM_BIN", raising=False)
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(tmp_path / "ws"))
    with pytest.raises(FileNotFoundError, match="build-hfm"):
        hfm.find_hfm()
    _exe(tmp_path / "home" / ".cache" / "vesselseg" / "hfm" / "build")
    assert hfm.find_hfm().parts[-4:] == (".cache", "vesselseg", "hfm", "build")
    _exe(tmp_path / "ws" / ".vesselseg" / "hfm" / "build")
    assert hfm.find_hfm().parts[-4] == "ws"
    exe = _exe(tmp_path / "env")
    monkeypatch.setenv("VESSELSEG_HFM_BIN", str(exe))  # the executable itself
    assert hfm.find_hfm() == tmp_path / "env"
    _exe(tmp_path / "explicit")
    assert hfm.find_hfm(tmp_path / "explicit") == tmp_path / "explicit"


def test_find_hfm_needs_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PIXI_PROJECT_ROOT", raising=False)
    exe = _exe(tmp_path / "d")
    os.chmod(exe, 0o644)
    monkeypatch.setenv("VESSELSEG_HFM_BIN", str(tmp_path / "d"))
    with pytest.raises(FileNotFoundError):
        hfm.find_hfm()


def test_cli_dispatch(capsys):
    cli.main([])
    assert "fit-radius" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="unknown command"):
        cli.main(["nope"])
    with pytest.raises(SystemExit) as e:
        cli.main(["extract", "--help"])
    assert e.value.code == 0
    assert "--lns_box" in capsys.readouterr().out
