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
    """The named chain resolves to prebuilt steps + its spec deviations.

    ``pipeline_steps()`` builds the chain itself (so ``chain_params`` can be
    forwarded) and returns a ``ResolvedChainSteps`` — the deviations travel
    with the list, otherwise ``PreprocessingPipeline(spec.pipeline_steps())``
    would drop them and hash differently from
    ``PreprocessingPipeline('hbn_langer')`` (the golden pin).
    """
    from eegcpm.modules.preprocessing.pipeline_builder import ResolvedChainSteps

    spec = load_preprocessing_spec({"steps": "hbn_langer"})
    steps = spec.pipeline_steps()
    assert isinstance(steps, ResolvedChainSteps)
    assert steps.spec_deviations  # P1-F2 deviation travels with the chain
    pipeline = PreprocessingPipeline(steps)
    assert pipeline.spec_deviations == steps.spec_deviations
    assert [s.name for s in pipeline.steps] == [
        "channel_roles", "zapline", "filter", "bad_channel_detection",
        "interpolate", "eog_regression", "asr",
        "block_rejection", "reference",
    ]
    # The spec-resolved chain is the same resolved config as the string path.
    from eegcpm.core.config_hash import resolved_config_hash

    assert resolved_config_hash(pipeline) == resolved_config_hash(
        PreprocessingPipeline("hbn_langer"))


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
    steps = spec.pipeline_steps()
    assert [s.name for s in steps][0] == "channel_roles"
    assert len(steps) == 9


# ---------------------------------------------------------------------------
# chain_params (plan §12.6): per-dataset tuning of the named chain
# ---------------------------------------------------------------------------

def test_chain_params_unknown_key_rejected():
    """`extra='forbid'`: a misspelled chain parameter is a hard error."""
    with pytest.raises(ValidationError):
        load_preprocessing_spec(
            {"steps": "hbn_langer", "chain_params": {"asr_cutof": 20}})


def test_chain_params_wrong_type_rejected():
    with pytest.raises(ValidationError):
        load_preprocessing_spec(
            {"steps": "hbn_langer", "chain_params": {"asr_cutoff": "twenty"}})
    with pytest.raises(ValidationError):
        load_preprocessing_spec(
            {"steps": "hbn_langer", "chain_params": {"asr_max_dims": 1.5}})
    with pytest.raises(ValidationError):
        load_preprocessing_spec(
            {"steps": "hbn_langer",
             "chain_params": {"asr_drift_transition_hz": [1.0, 0.5]}})
    with pytest.raises(ValidationError):
        load_preprocessing_spec(
            {"steps": "hbn_langer", "chain_params": {"asr_calibration": "auto"}})


def test_chain_params_r6_bounds_are_hard_errors():
    """R6 (ASR scoped audit): cutoff and drift-band bounds, not silent no-ops.

    ``asr_cutoff`` is bounded to [1, 200] and ``asr_drift_transition_hz`` must
    satisfy ``0 < low < high`` with ``high <= 5`` Hz. The Nyquist half of the
    drift bound is data-dependent and is enforced at runtime in ``ASRStep``
    (see ``test_asr_timeline.py::test_drift_transition_above_nyquist_refused``).
    """
    for bad in (0.5, 0.0, -3.0, 500.0):
        with pytest.raises(ValidationError, match="asr_cutoff"):
            load_preprocessing_spec(
                {"steps": "hbn_langer", "chain_params": {"asr_cutoff": bad}})
    for ok in (1.0, 5.0, 20.0, 200.0):
        spec = load_preprocessing_spec(
            {"steps": "hbn_langer", "chain_params": {"asr_cutoff": ok}})
        assert spec.chain_params.asr_cutoff == ok

    for bad in ([0.0, 0.75], [1.0, 0.5], [0.25, 0.25], [0.5, 5.5], [0.5, 10.0]):
        with pytest.raises(ValidationError, match="asr_drift_transition_hz"):
            load_preprocessing_spec(
                {"steps": "hbn_langer",
                 "chain_params": {"asr_drift_transition_hz": bad}})
    spec = load_preprocessing_spec(
        {"steps": "hbn_langer",
         "chain_params": {"asr_drift_transition_hz": [0.5, 5.0]}})
    assert spec.chain_params.asr_drift_transition_hz == (0.5, 5.0)
    assert load_preprocessing_spec(
        {"steps": "hbn_langer",
         "chain_params": {"asr_drift_transition_hz": None}}
    ).chain_params.asr_drift_transition_hz is None


def test_chain_params_with_explicit_step_list_rejected():
    """chain_params next to an explicit step list would be silently ignored."""
    with pytest.raises(ValidationError, match="chain_params"):
        load_preprocessing_spec({
            "steps": [{"name": "filter", "params": {"l_freq": 1.0}}],
            "chain_params": {"asr_cutoff": 20},
        })
    # The schema default next to an explicit list is not "given" -> accepted.
    spec = load_preprocessing_spec({
        "steps": [{"name": "filter", "params": {"l_freq": 1.0}}]})
    assert spec.chain_params.asr_cutoff == 20.0


def test_chain_params_defaults_match_the_golden_chain():
    """Omitting chain_params resolves to the pinned (golden) ASR settings."""
    from eegcpm.modules.preprocessing.hbn_langer_chain import (
        build_hbn_langer_chain,
    )

    spec = load_preprocessing_spec({"steps": "hbn_langer"})
    assert spec.chain_params.asr_cutoff == 20.0
    assert spec.chain_params.asr_drift_transition_hz == (0.25, 0.75)
    assert spec.chain_params.asr_max_dims == 0.66
    assert spec.chain_params.asr_calibration == "fixed_first_60s"

    default_chain = build_hbn_langer_chain()
    spec_chain = build_hbn_langer_chain(**spec.chain_params.model_dump())
    assert [s.get_config() for s in spec_chain] == [
        s.get_config() for s in default_chain
    ]


def test_chain_params_accepted_values_reach_the_wrapper():
    """The per-dataset ASR values reach the built chain's ASR wrapper."""
    spec = load_preprocessing_spec({
        "steps": "hbn_langer",
        "chain_params": {
            "asr_cutoff": 30,
            "asr_drift_transition_hz": [0.5, 1.0],
            "asr_max_dims": 0.5,
            "asr_calibration": "data_driven",
        },
    })
    pipeline = PreprocessingPipeline(spec.pipeline_steps())
    asr = pipeline.steps[6]
    assert asr.name == "asr"
    assert asr.cutoff == 30.0
    assert asr.drift_transition_hz == (0.5, 1.0)
    assert asr.max_dims == 0.5
    assert asr.calibration_window == "data_driven"
    cfg = asr.get_config()
    assert cfg["cutoff"] == 30.0
    assert cfg["drift_transition_hz"] == [0.5, 1.0]
    assert cfg["max_dims"] == 0.5
    # Truthful data-driven calibration description (not the fixed-path one).
    assert cfg["calibration_reference"] == "eegprep_clean_windows_selection"
    assert cfg["ref_maxbadchannels"] == 0.075

    # A None transition (non-EEGLAB regime) is expressible per dataset too.
    spec_none = load_preprocessing_spec({
        "steps": "hbn_langer",
        "chain_params": {"asr_drift_transition_hz": None},
    })
    asr_none = PreprocessingPipeline(spec_none.pipeline_steps()).steps[6]
    assert asr_none.drift_transition_hz is None


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
