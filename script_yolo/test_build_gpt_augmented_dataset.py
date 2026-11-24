from pathlib import Path

from build_gpt_augmented_dataset import real_test_paths, write_calibration


def test_real_test_paths_use_final_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "landslide_yolo_gpt_clean"
    rows = [
        {
            "status": "kept",
            "source": "reviewed",
            "split": "test",
            "output_path": "images/test/example.jpg",
        },
        {
            "status": "kept",
            "source": "gpt_v1",
            "split": "test",
            "output_path": "images/test/generated.jpg",
        },
    ]

    paths = real_test_paths(rows, output)

    assert paths == [str((output / "images/test/example.jpg").resolve())]
    assert all(".tmp" not in path for path in paths)


def test_calibration_paths_use_final_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "landslide_yolo_gpt_clean"
    staging = tmp_path / ".landslide_yolo_gpt_clean.tmp"
    staging.mkdir()
    rows = []
    for index in range(250):
        rows.extend(
            [
                {
                    "source": "reviewed",
                    "positive": "1",
                    "split": "train",
                    "output_path": f"images/train/positive_{index}.jpg",
                },
                {
                    "source": "negative_captures",
                    "positive": "0",
                    "split": "train",
                    "output_path": f"images/train/negative_{index}.jpg",
                },
            ]
        )

    write_calibration(rows, staging, output)

    paths = (staging / "calibration.txt").read_text(encoding="utf-8").splitlines()
    assert len(paths) == 500
    assert all(path.startswith(str(output.resolve())) for path in paths)
    assert all(".tmp" not in path for path in paths)
