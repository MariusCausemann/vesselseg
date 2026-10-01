"""`vesselseg segment` -> brainseg's risc-mi container call, without running it."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from brainseg.tools import riscmi

from vesselseg import segment


@pytest.fixture
def image(tmp_path):
    aff = np.diag([-0.5, 0.5, 0.5, 1.0])  # LAS: not what risc-mi wants
    path = tmp_path / "in" / "T2w.nii.gz"
    path.parent.mkdir()
    nib.save(nib.Nifti1Image(np.random.rand(8, 9, 10).astype(np.float32), aff), path)
    return path


@pytest.fixture
def fake_container(tmp_path, monkeypatch):
    """Capture the apptainer command and write what the container would."""
    calls = []

    def run_command(cmd, description):
        calls.append(cmd)
        binds = dict(cmd[i + 1].split(":")[::-1] for i, c in enumerate(cmd) if c == "--bind")
        host = lambda p: Path(binds[str(Path(p).parent)]) / Path(p).name
        src = host(cmd[cmd.index("-i") + 1])
        im = sitk.ReadImage(str(src))
        orient = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines
        assert orient(im.GetDirection()) == "RAI"
        assert im.GetPixelID() == sitk.sitkFloat32
        sitk.WriteImage(im, str(host(cmd[cmd.index("--prob") + 1])))
        sitk.WriteImage(sitk.Cast(im > 0.5, sitk.sitkUInt8), str(host(cmd[cmd.index("-o") + 1])))

    monkeypatch.setattr(riscmi, "run_command", run_command)
    monkeypatch.setattr(riscmi, "get_container_runtime", lambda: "apptainer")
    sif = tmp_path / "brainseg_riscmi.sif"
    sif.touch()
    return sif, calls


def test_segment_command(tmp_path, image, fake_container):
    sif, calls = fake_container
    out = tmp_path / "riscmi" / "t2_prob.nii.gz"
    segment.segment(image, out, container=sif, gpu=True, tta=False, workers=4)
    (cmd,) = calls
    assert cmd[:3] == ["apptainer", "exec", "--cleanenv"] and "--nv" in cmd
    assert str(sif) in cmd
    assert cmd[cmd.index("--prob") + 1] == "/data_out/t2_prob.nii.gz"
    assert cmd[cmd.index("-o") + 1] == "/data_out/t2_prob_mask.nii.gz"
    assert "--no_tta" in cmd and cmd[cmd.index("--workers") + 1] == "4"
    assert out.exists() and not (out.parent / "t2_prob_mask.nii.gz").exists()
    assert not list(out.parent.glob(".t2_prob_*"))  # the RAI copy is cleaned up
    ref = sitk.DICOMOrient(sitk.ReadImage(str(image)), "RAI")
    assert np.allclose(sitk.ReadImage(str(out)).GetDirection(), ref.GetDirection())


def test_segment_keeps_mask_cpu(tmp_path, image, fake_container):
    sif, calls = fake_container
    out = tmp_path / "t1_prob.nii.gz"
    segment.segment(image, out, container=sif, keep_mask=True)
    assert "--nv" not in calls[0] and "--no_tta" not in calls[0]
    assert (tmp_path / "t1_prob_mask.nii.gz").exists()


@pytest.mark.parametrize(
    "name, prob",
    [("v.nii.gz", "v_prob.nii.gz"), ("v.nii", "v_prob.nii"), ("v", "v_prob.nii.gz")],
)
def test_prob_output_path(name, prob):
    assert riscmi.prob_output_path(Path("/x") / name) == Path("/x") / prob


def test_prob_path_must_be_in_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(riscmi, "run_command", lambda *a: None)
    monkeypatch.setattr(riscmi, "get_container_runtime", lambda: "apptainer")
    with pytest.raises(ValueError):
        riscmi.run_riscmi_arteries(
            tmp_path / "in.nii.gz",
            tmp_path / "a" / "m.nii.gz",
            "x.sif",
            save_prob=True,
            prob_path=tmp_path / "b" / "p.nii.gz",
        )


def test_container_not_pulled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BRAINSEG_CONTAINER_DIR", str(tmp_path / "sifs"))
    with pytest.raises(FileNotFoundError, match="brainseg_riscmi.sif"):
        segment.find_riscmi_container()
    with pytest.raises(FileNotFoundError):
        segment.find_riscmi_container(tmp_path / "missing.sif")
    (tmp_path / "sifs").mkdir()
    (tmp_path / "sifs" / "brainseg_riscmi.sif").touch()
    assert segment.find_riscmi_container() == tmp_path / "sifs" / "brainseg_riscmi.sif"
