"""Unit tests for synthetic BraTS dataset generation."""

from pathlib import Path
import nibabel as nib
import numpy as np

from fedmed.data.synthetic import generate_synthetic_dataset


def test_generate_synthetic_dataset_msd(tmp_path: Path) -> None:
    """Verify MSD 4D stacked NIfTI generation and non-empty tumor labels."""
    patient_ids = generate_synthetic_dataset(
        root_dir=tmp_path,
        num_patients=2,
        spatial_size=(32, 32, 32),
        seed=42,
        layout="msd",
        label_scheme="msd",
    )
    assert len(patient_ids) == 2
    for pid in patient_ids:
        case_dir = tmp_path / pid
        image_p = case_dir / "image.nii.gz"
        label_p = case_dir / "label.nii.gz"
        assert image_p.is_file()
        assert label_p.is_file()

        img_data = nib.load(str(image_p)).get_fdata()
        lbl_data = nib.load(str(label_p)).get_fdata()

        assert img_data.shape == (32, 32, 32, 4)
        assert lbl_data.shape == (32, 32, 32)
        assert np.max(lbl_data) > 0


def test_generate_synthetic_dataset_brats(tmp_path: Path) -> None:
    """Verify BraTS separated 4-modality NIfTI generation."""
    patient_ids = generate_synthetic_dataset(
        root_dir=tmp_path,
        num_patients=1,
        spatial_size=(24, 24, 24),
        seed=123,
        layout="brats",
        label_scheme="brats",
    )
    assert len(patient_ids) == 1
    pid = patient_ids[0]
    case_dir = tmp_path / pid
    for mod in ("t1", "t1ce", "t2", "flair"):
        assert (case_dir / f"{pid}_{mod}.nii.gz").is_file()
    assert (case_dir / f"{pid}_seg.nii.gz").is_file()
