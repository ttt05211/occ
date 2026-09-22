from causal_se2_occ.io import prepare_output_file


def test_prepare_output_file_creates_nested_parent(tmp_path):
    target = tmp_path / "nested" / "deeper" / "result.json"
    out = prepare_output_file(target)
    assert out == target
    assert target.parent.is_dir()
    assert not target.exists()
