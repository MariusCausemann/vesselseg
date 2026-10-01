import pytest

from vesselseg import cli, pipeline


def test_to_argv():
    argv = pipeline.to_argv({"workers": 64, "t2_only": True, "verbose": False, "lns_box": [20, 40]})
    assert argv == ["--workers", "64", "--t2_only", "--lns_box", "20", "40"]
    assert pipeline.to_argv(None) == []


def test_cli_dispatch(capsys):
    cli.main([])
    assert "fit-radius" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="unknown command"):
        cli.main(["nope"])
    with pytest.raises(SystemExit) as e:
        cli.main(["extract", "--help"])
    assert e.value.code == 0
    assert "--lns_box" in capsys.readouterr().out
