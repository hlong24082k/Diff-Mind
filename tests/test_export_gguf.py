import pytest

from src.training.export_gguf import collect_ggufs


def test_collect_ggufs_renames_requested_quants(tmp_path):
    src, out = tmp_path / "v1-s125-merged_gguf", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    for n in ("unsloth.Q4_K_M.gguf", "unsloth.F16.gguf"):
        (src / n).write_text(n)
    files = collect_ggufs([tmp_path / "missing", tmp_path], "v1-s125", out, ["q4_k_m", "f16"])
    assert [f.name for f in files] == ["v1-s125-q4_k_m.gguf", "v1-s125-f16.gguf"]
    assert files[0].read_text() == "unsloth.Q4_K_M.gguf"


def test_collect_ggufs_fails_when_quant_missing(tmp_path):
    src = tmp_path / "v1-merged_gguf"
    src.mkdir()
    (src / "unsloth.Q4_K_M.gguf").write_text("x")
    with pytest.raises(AssertionError):
        collect_ggufs([tmp_path], "v1", tmp_path, ["f16"])


def test_collect_ggufs_ignores_other_runs(tmp_path):
    other = tmp_path / "v1-s201-merged_gguf"
    other.mkdir()
    (other / "x.Q4_K_M.gguf").write_text("x")
    with pytest.raises(AssertionError):
        collect_ggufs([tmp_path], "v1-s125", tmp_path, ["q4_k_m"])
