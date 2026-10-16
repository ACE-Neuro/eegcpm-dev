"""P2-F5: one source config schema — nested sections consumed, flat keys raise.

The source stage has a single pydantic schema (``extra='forbid'``). The legacy
flat dialect (``snr``/``loose``/``depth``/``spacing``/``trial_level`` at the top
level) is a hard error instead of being silently ignored, and the nested
``forward``/``inverse`` sections must actually reach the module attributes.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from eegcpm.core.config import SourceConfig
from eegcpm.modules.source import SourceReconstructionModule

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG_DIRS = (
    REPO_ROOT / "examples" / "configs" / "source",
    REPO_ROOT / "eegcpm" / "config" / "source",
)


# --- flat / legacy dialect is rejected -------------------------------------

def test_flat_snr_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"snr": 2.0})


def test_flat_trial_level_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"trial_level": True})


def test_flat_spacing_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"spacing": "oct5"})


def test_flat_loose_and_depth_raise():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"loose": 0.2, "depth": 0.8})


def test_unknown_output_key_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"output": {"save_stc": True, "nope": 1}})


def test_non_fsaverage_template_raises(tmp_path):
    """F4: ``forward.template`` is fsaverage-only, not a silently-ignored string.

    The source module's template forward/BEM/head->MRI transform and CONN
    labels are all fsaverage-specific and the module never read the field, so
    ``template: colin27`` used to be accepted and then silently ignored. It is
    a ``Literal['fsaverage']`` now: any other value is rejected at validation
    (schema and module-construction paths).
    """
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"forward": {"template": "colin27"}})
    with pytest.raises(ValidationError):
        SourceReconstructionModule(
            {"forward": {"template": "fsaverage-ECoG"}}, tmp_path
        )
    # The valid value (and the default) still validate and reach the config.
    assert SourceConfig.model_validate({}).forward.template == "fsaverage"
    assert (
        SourceConfig.model_validate(
            {"forward": {"template": "fsaverage"}}
        ).forward.template
        == "fsaverage"
    )


def test_module_construction_rejects_flat_keys(tmp_path):
    with pytest.raises(ValidationError):
        SourceReconstructionModule({"snr": 2.0}, tmp_path)


# --- variant / method coherence --------------------------------------------

def test_variant_method_mismatch_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"method": "dSPM", "variant": "sLORETA-CONN32"})


def test_variant_derived_from_method():
    assert SourceConfig.model_validate({"method": "dSPM"}).variant == "dSPM-CONN32"
    assert (
        SourceConfig.model_validate({"method": "dSPM", "variant": "dSPM-CONN32"}).variant
        == "dSPM-CONN32"
    )
    assert SourceConfig.model_validate({}).variant == "sLORETA-CONN32"


def test_non_conn_networks_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"parcellation": "aparc"})


def test_invalid_method_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"method": "custom"})


def test_invalid_mode_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"mode": "trial"})


# --- numeric bounds (B4-N1) -------------------------------------------------

def test_equalize_trials_zero_or_negative_raises():
    """equalize_trials=0 would select zero trials per condition -> reject."""
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"equalize_trials": 0})
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"equalize_trials": -1})
    assert SourceConfig.model_validate({"equalize_trials": 1}).equalize_trials == 1


def test_min_trials_below_one_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"min_trials": 0})


def test_negative_random_seed_raises():
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"random_seed": -1})
    assert SourceConfig.model_validate({"random_seed": 0}).random_seed == 0


def test_bounds_reach_module_construction(tmp_path):
    with pytest.raises(ValidationError):
        SourceReconstructionModule({"equalize_trials": 0}, tmp_path)
    with pytest.raises(ValidationError):
        SourceReconstructionModule({"random_seed": -1}, tmp_path)


# --- nested config reaches the module --------------------------------------

def test_nested_config_reaches_module_attributes(tmp_path):
    module = SourceReconstructionModule(
        {
            "forward": {"spacing": "oct5"},
            "inverse": {"snr": 2.0, "snr_trial": 1.5, "loose": 0.1, "depth": 0.9},
            "roi_radius": 12.0,
            "mode": "both",
            "min_trials": 4,
            "equalize_trials": 3,
            "random_seed": 7,
        },
        tmp_path,
    )
    assert module.method == "sLORETA"
    assert module.parcellation == "conn_networks"
    assert module.spacing == "oct5"
    assert module.snr == 2.0
    assert module.snr_trial == 1.5
    assert module.loose == 0.1
    assert module.depth == 0.9
    assert module.roi_radius == 12.0
    assert module.mode == "both"
    assert module.min_trials == 4
    assert module.equalize_trials == 3
    assert module.random_seed == 7


def test_defaults_unchanged(tmp_path):
    module = SourceReconstructionModule({}, tmp_path)
    assert module.method == "sLORETA"
    assert module.snr == 3.0
    assert module.snr_trial == 1.0
    assert module.loose == 0.2
    assert module.depth == 0.8
    assert module.spacing == "oct6"
    assert module.mode == "evoked"
    assert module.min_trials == 10
    assert module.equalize_trials is None


# --- every shipped source config validates ---------------------------------

def _source_yaml_paths():
    paths = []
    for directory in SOURCE_CONFIG_DIRS:
        if directory.exists():
            paths.extend(sorted(directory.glob("*.yaml")))
    return paths


@pytest.mark.parametrize("path", _source_yaml_paths(), ids=lambda p: p.name)
def test_source_yaml_configs_validate(path):
    cfg = SourceConfig.from_yaml(path)
    assert cfg.variant is not None
    assert cfg.variant.split("-")[0].lower() == cfg.method.lower()
    # the nested sections are actually consumed
    assert cfg.forward.spacing
    assert cfg.inverse.snr > 0
    assert 0.0 <= cfg.inverse.loose <= 1.0
    assert 0.0 <= cfg.inverse.depth <= 1.0


def test_shipped_source_configs_exist():
    paths = _source_yaml_paths()
    assert paths, "no source YAML configs found to validate"
