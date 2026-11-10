
import subprocess
import sys

import cv2
import numpy as np


SCRIPT = "src/robust_pipeline.py"


def run_pipeline(input_dir, output_dir, workers=2, batch_size=2):
    """Run the robust pipeline with isolated test reports."""

    report_dir = output_dir.parent / "test_reports"

    return subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--workers", str(workers),
            "--batch-size", str(batch_size),
            "--report-dir", str(report_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60
    )


def create_valid_image(folder, filename="test.jpg"):
    """Create a small valid JPEG image."""

    folder.mkdir(parents=True, exist_ok=True)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:] = (100, 150, 200)

    image_path = folder / filename

    success = cv2.imwrite(str(image_path), image)
    assert success

    return image_path


def test_valid_image_processing(tmp_path):
    """A valid JPEG should be processed successfully."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    create_valid_image(input_dir)

    result = run_pipeline(input_dir, output_dir)

    assert result.returncode == 0
    assert "processed: 1" in result.stdout
    assert "failed: 0" in result.stdout
    assert (output_dir / "test.jpg").exists()


def test_corrupted_image_handling(tmp_path):
    """A corrupted JPEG should be retried and recorded in the error CSV."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    report_dir = tmp_path / "test_reports"

    input_dir.mkdir()

    create_valid_image(input_dir, "valid.jpg")

    corrupted_path = input_dir / "corrupted.jpg"
    corrupted_path.write_text(
        "This is not a valid JPEG image",
        encoding="utf-8"
    )

    result = run_pipeline(input_dir, output_dir)

    assert result.returncode == 0
    assert "processed: 1" in result.stdout
    assert "failed: 1" in result.stdout

    # Verify the valid image was processed.
    assert (output_dir / "valid.jpg").exists()

    # Verify the corrupted image was recorded in the CSV.
    error_report = report_dir / "pipeline_errors.csv"

    assert error_report.exists()

    with open(error_report, "r", encoding="utf-8") as file:
        report_content = file.read()

    assert "corrupted.jpg" in report_content
    assert "loading" in report_content
    assert "3" in report_content


def test_missing_input_directory(tmp_path):
    """A missing input directory should produce a clear error."""

    input_dir = tmp_path / "missing"
    output_dir = tmp_path / "output"

    result = run_pipeline(input_dir, output_dir)

    assert result.returncode != 0
    assert "Input directory not found" in result.stderr


def test_empty_input_directory(tmp_path):
    """An empty input directory should be rejected."""

    input_dir = tmp_path / "empty"
    output_dir = tmp_path / "output"

    input_dir.mkdir()

    result = run_pipeline(input_dir, output_dir)

    assert result.returncode != 0
    assert "No JPG images found" in result.stderr


def test_invalid_worker_count(tmp_path):
    """Zero workers should be rejected."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    create_valid_image(input_dir)

    result = run_pipeline(
        input_dir,
        output_dir,
        workers=0
    )

    assert result.returncode != 0
    assert "Workers and batch size must be positive" in result.stderr


def test_invalid_batch_size(tmp_path):
    """Zero batch size should be rejected."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    create_valid_image(input_dir)

    result = run_pipeline(
        input_dir,
        output_dir,
        batch_size=0
    )

    assert result.returncode != 0
    assert "Workers and batch size must be positive" in result.stderr


def test_multiple_valid_images(tmp_path):
    """Multiple valid images should all be processed successfully."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    for i in range(5):
        create_valid_image(input_dir, f"image_{i}.jpg")

    result = run_pipeline(
        input_dir,
        output_dir,
        workers=2,
        batch_size=2
    )

    assert result.returncode == 0
    assert "processed: 5" in result.stdout
    assert "failed: 0" in result.stdout

    for i in range(5):
        assert (output_dir / f"image_{i}.jpg").exists()
