from hindi_whisper.agreement import longest_common_prefix
from hindi_whisper.config import ExperimentConfig


def test_prefix_commits_only_the_shared_start():
    assert longest_common_prefix(["maine", "kaha"], ["maine", "suna"]) == ["maine"]
    assert longest_common_prefix(["a", "b"], ["a", "b", "c"]) == ["a", "b"]
    assert longest_common_prefix(["x"], ["y"]) == []
    assert longest_common_prefix([], ["a"]) == []


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("run_name: x\nnot_a_field: 1\n", encoding="utf-8")
    try:
        ExperimentConfig.from_yaml(path)
    except ValueError as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("unknown key should fail")


def test_small_experiment_yaml_loads():
    cfg = ExperimentConfig.from_yaml("experiments/fleurs_small.yaml")
    assert cfg.dataset == "fleurs"
    assert cfg.language == "hindi"
    assert cfg.lora_steps == 150
    assert cfg.model_name == "openai/whisper-small"
