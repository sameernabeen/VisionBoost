
import cv2
import numpy as np
import pytest

from pathlib import Path
from src.optimized_pipeline import run_pipeline, process_image


def create_test_images(folder, count=5):
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        image = np.random.randint(
            0, 256, (100, 100, 3), dtype=np.uint8
        )

        cv2.imwrite(str(folder / f"image_{i}.jpg"), image)


def test_pipeline_processes_all_images(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    create_test_images(input_dir, 5)

    processed, failed, runtime = run_pipeline(
        input_dir, output_dir, workers=2, batch_size=2
    )

    assert processed == 5
    assert failed == 0
    assert runtime > 0
    assert len(list(output_dir.glob("*.jpg"))) == 5


def test_pipeline_handles_incomplete_final_batch(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    create_test_images(input_dir, 5)

    processed, failed, _ = run_pipeline(
        input_dir, output_dir, workers=2, batch_size=3
    )

    assert processed == 5
    assert failed == 0


def test_pipeline_rejects_zero_workers(tmp_path):
    with pytest.raises(ValueError, match="Workers"):
        run_pipeline(
            tmp_path, tmp_path / "output", workers=0, batch_size=2
        )


def test_pipeline_rejects_zero_batch_size(tmp_path):
    with pytest.raises(ValueError, match="Batch size"):
        run_pipeline(
            tmp_path, tmp_path / "output", workers=2, batch_size=0
        )


def test_pipeline_rejects_missing_input(tmp_path):
    with pytest.raises(ValueError, match="Input directory"):
        run_pipeline(
            tmp_path / "missing",
            tmp_path / "output",
            workers=2,
            batch_size=2
        )


def test_pipeline_rejects_empty_input(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="No JPG images"):
        run_pipeline(
            input_dir,
            tmp_path / "output",
            workers=2,
            batch_size=2
        )


def test_process_image_rejects_invalid_image(tmp_path):
    result = process_image(
        Path("invalid.jpg"),
        None,
        tmp_path
    )

    assert result is False


def test_process_image_handles_save_failure(tmp_path, monkeypatch):
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)

    result = process_image(
        Path("test.jpg"),
        image,
        tmp_path
    )

    assert result is False
