from pathlib import Path

from validate_dataset import resolve_root


def test_resolve_root_defaults_to_yaml_parent_for_relative_data_path(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    data_path = project / "dataset" / "data_portable.yaml"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("train: images/train\n", encoding="utf-8")
    monkeypatch.chdir(project)

    root = resolve_root(Path("dataset/data_portable.yaml"), {})

    assert root == data_path.parent.resolve()
