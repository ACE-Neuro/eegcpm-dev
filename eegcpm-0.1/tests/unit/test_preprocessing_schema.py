"""Single preprocessing config schema (B1 — P1-F6; the schema half of C5).

One schema, ``extra='forbid'``: the nested ``params:`` dialect is the
only accepted one. The *flat* dialect (step keys next to ``name``) and
the *keyed* dialect (``{'filter': {...}}``) are hard errors instead of
silent no-ops, unknown step names are rejected, and every shipped
config validates **and** builds a pipeline.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eegcpm.core.config import PreprocessingSpec, load_preprocessing_spec
from eegcpm.modules.preprocessing import PreprocessingPipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATHS = sorted(
    (_REPO_ROOT / "examples" / "configs" / "preprocessing").glob("*.yaml")
) + sorted(
    (_REPO_ROOT / "eegcpm" / "config" / "preprocessing").glob("*.yaml")
)


def test_flat_dialect_rejected():
    """Flat keys next to `name` are forbidden (they used to no-op)."""
    with pytest.raises(ValidationError):
        load_preprocessing_spec(
            {"steps": [{"name": "filter", "l_freq": 1.0}]}
        )


def test_keyed_dialect_rejected():
    """The legacy keyed-section dialect is not a config at all."""
    with pytest.raises(ValidationError):
        load_preprocessing_spec({"filter": {"l_freq": 1.0, "h_freq": 40.0}})


def test_keyed_dialect_inside_steps_rejected():
    with pytest.raises(ValidationError):
        load_preprocessing_spec({"steps": {"filter": {"l_freq": 1.0}}})


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValidationError):
        load_preprocessing_spec(
            {"steps": [{"name": "filter", "params": {}}], "window_criterion": 0.2}
        )


def test_empty_steps_rejected():
    with pytest.raises(ValidationError):
        load_preprocessing_spec({"steps": []})


def test_missing_steps_rejected():
    with pytest.raises(ValidationError):
        load_preprocessing_spec({"stage": "preprocessing"})


def test_unknown_step_name_rejected():
    with pytest.raises(ValidationError):
        load_preprocessing_spec({"steps": [{"name": "not_a_step", "params": {}}]})


def test_project_config_envelope_unwrapped():
    spec = load_preprocessing_spec({
        "preprocessing": {
            "steps": [{"name": "filter", "params": {"l_freq": 1.0, "h_freq": 40.0}}],
        },
        "source": {"method": "dSPM"},
    })
    assert isinstance(spec, PreprocessingSpec)
    assert [s.name for s in spec.steps] == ["filter"]
    assert spec.steps[0].params == {"l_freq": 1.0, "h_freq": 40.0}


def test_named_chain_accepted():
    spec = load_preprocessing_spec({"steps": "hbn_langer"})
    assert spec.pipeline_steps() == "hbn_langer"
    pipeline = PreprocessingPipeline(spec.pipeline_steps())
    assert [s.name for s in pipeline.steps] == [
        "channel_roles", "zapline", "filter", "bad_channel_detection",
        "interpolate", "eog_regression", "robust_pca", "asr",
        "block_rejection", "reference",
    ]


def test_unknown_named_chain_rejected():
    with pytest.raises(ValidationError):
        load_preprocessing_spec({"steps": "no_such_chain"})


def test_variant_hbn_langer_with_step_list_rejected():
    """Only `steps:` selects the chain.

    `variant: hbn_langer` next to an explicit step list silently changed
    meaning when the chain became code-defined, so it is now a hard
    error (older configs relied on the variant alone).
    """
    with pytest.raises(ValidationError):
        load_preprocessing_spec({
            "variant": "hbn_langer",
            "steps": [{"name": "filter", "params": {"l_freq": 1.0}}],
        })


def test_variant_other_name_with_step_list_accepted():
    """A non-chain variant label is metadata and stays allowed."""
    spec = load_preprocessing_spec({
        "variant": "standard",
        "steps": [{"name": "filter", "params": {"l_freq": 1.0}}],
    })
    assert [s.name for s in spec.steps] == ["filter"]


def test_variant_hbn_langer_with_named_chain_accepted():
    """`variant: hbn_langer` + `steps: hbn_langer` is consistent."""
    spec = load_preprocessing_spec(
        {"variant": "hbn_langer", "steps": "hbn_langer"})
    assert spec.pipeline_steps() == "hbn_langer"


def test_envelope_with_top_level_steps_rejected():
    """The `preprocessing:` envelope and a top-level `steps:` key
    together are ambiguous → ValidationError (the shape the CLI
    already handles)."""
    with pytest.raises(ValidationError):
        load_preprocessing_spec({
            "preprocessing": {
                "steps": [{"name": "filter", "params": {"l_freq": 1.0}}],
            },
            "steps": [{"name": "filter", "params": {"l_freq": 1.0}}],
        })


def test_required_default_true_and_override():
    spec = load_preprocessing_spec({
        "steps": [
            {"name": "filter", "params": {"l_freq": 1.0}},
            {"name": "interpolate", "params": {}, "required": False},
        ]
    })
    assert spec.steps[0].required is True
    assert spec.steps[1].required is False

    pipeline = PreprocessingPipeline(spec.pipeline_steps())
    assert pipeline.steps[0].required is True
    assert pipeline.steps[1].required is False


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: p.name)
def test_every_shipped_config_validates_and_constructs(config_path):
    """Every shipped config (examples + package templates) validates
    against the one schema and builds its pipeline."""
    assert config_path.exists(), f"config disappeared: {config_path}"
    with open(config_path) as f:
        data = yaml.safe_load(f)

    spec = load_preprocessing_spec(data)
    pipeline = PreprocessingPipeline(spec.pipeline_steps())
    assert len(pipeline.steps) >= 1
    assert all(s.name for s in pipeline.steps)


def test_legacy_keyed_configs_are_gone():
    """The deleted keyed-dialect files must not come back."""
    legacy_dir = _REPO_ROOT / "config"
    for name in ("preprocessing_config.yaml",
                 "preprocessing_mne_bids_equivalent.yaml",
                 "preprocessing_mne_bids_with_iclabel.yaml"):
        assert not (legacy_dir / name).exists(), f"{name} was restored"
