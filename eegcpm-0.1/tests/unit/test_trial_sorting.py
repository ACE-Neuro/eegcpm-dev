"""Tests for the consolidated trial sorting/filtering API (Pass 1).

Covers:
- ``SelectionRecord`` provenance fields (mask hash, source hash, timestamp)
- ``TrialSorter.select`` / ``sort`` / ``split`` (new API)
- The legacy UI-facing methods keep their historical return shapes
- ``trial_filter.build_mask`` / ``query_mask`` / ``apply_trial_filters``

The sorters are built from the real pydantic ``TaskConfig`` schema — the
cptvis task config on disk plus a synthetic three-condition config.
"""

import hashlib
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from eegcpm.core.task_config import (
    BinSpec,
    ConditionSpec,
    ResponseSpec,
    SelectionRecord,
    TaskConfig,
    TrialSorter,
)
from eegcpm.modules.epochs import trial_filter as tf

CPTVIS_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "eegcpm"
    / "config"
    / "tasks"
    / "cptvis.yaml"
)

N_PER_CONDITION = 60
CONDITIONS = ["target", "nontarget", "distractor"]
TRIAL_SCALE = 1e-6  # epoch data marker: trial i is i * TRIAL_SCALE


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _synthetic_metadata(n_per_condition: int = N_PER_CONDITION, seed: int = 7) -> pd.DataFrame:
    """Behavioral metadata with 3 conditions (60 trials each by default)."""
    rng = np.random.default_rng(seed)
    n = n_per_condition * len(CONDITIONS)
    trial_type = np.repeat(np.array(CONDITIONS), n_per_condition)
    corr = rng.choice([1.0, 0.0, np.nan], size=n, p=[0.65, 0.25, 0.10])
    rt = rng.uniform(0.2, 1.0, size=n)
    rt[np.isnan(corr)] = np.nan  # no response -> no RT
    return pd.DataFrame(
        {
            "trial_type": trial_type,
            "key_resp.corr": corr,
            "key_resp.rt": rt,
            "difficulty": rng.uniform(0.0, 1.0, size=n),
            "trial_index": np.arange(n),
        }
    )


def _synthetic_epochs_with_metadata(
    n_per_condition: int = N_PER_CONDITION,
    seed: int = 7,
    sfreq: float = 100.0,
    event_id=None,
) -> mne.EpochsArray:
    """Synthetic epochs with per-trial metadata and an alignment marker.

    The data of trial ``i`` is constant ``i * TRIAL_SCALE`` on every channel
    and sample, so epoch data can be checked against
    ``metadata['trial_index']`` after any selection/reordering.
    """
    metadata = _synthetic_metadata(n_per_condition=n_per_condition, seed=seed)
    n = len(metadata)
    if event_id is None:
        event_id = {"target": 1, "nontarget": 2, "distractor": 3}
    info = mne.create_info(
        [f"EEG{i:03d}" for i in range(4)], sfreq, ch_types=["eeg"] * 4
    )
    data = np.full((n, 4, 11), TRIAL_SCALE, dtype=float)
    data *= np.arange(n)[:, None, None]
    event_codes = np.array([1, 2, 3])
    codes = np.tile(event_codes, n_per_condition)
    events = np.column_stack(
        [np.arange(1, n + 1) * 25, np.zeros(n, dtype=int), codes]
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=-0.05,
        event_id=event_id,
        metadata=metadata,
        verbose=False,
    )


def _markers(epochs) -> np.ndarray:
    """Recover per-trial marker indices from the epoch data."""
    return np.rint(epochs.get_data()[:, 0, 0] / TRIAL_SCALE).astype(int)


def _assert_aligned(epochs) -> None:
    """Epoch data order, events order, metadata order all agree."""
    assert epochs.metadata is not None
    trial_index = epochs.metadata["trial_index"].to_numpy()
    np.testing.assert_array_equal(_markers(epochs), trial_index)
    # events array rows follow the same metadata rows (events[:, 0] = (i+1)*25)
    np.testing.assert_array_equal(epochs.events[:, 0], (trial_index + 1) * 25)


@pytest.fixture()
def epochs():
    """Fresh synthetic epochs per test (selections must not mutate)."""
    return _synthetic_epochs_with_metadata()


@pytest.fixture(scope="module")
def cptvis_config():
    assert CPTVIS_CONFIG.exists(), f"missing task config: {CPTVIS_CONFIG}"
    return TaskConfig.from_yaml(CPTVIS_CONFIG)


@pytest.fixture(scope="module")
def cptvis_sorter(cptvis_config):
    return TrialSorter(cptvis_config)


@pytest.fixture(scope="module")
def three_condition_config():
    return TaskConfig(
        task_name="synthetic3",
        description="Synthetic three-condition task for sorting tests",
        task_type="event-related",
        conditions=[
            ConditionSpec(name="target", event_codes=["target"]),
            ConditionSpec(name="nontarget", event_codes=["nontarget"]),
            ConditionSpec(name="distractor", event_codes=["distractor"]),
        ],
        response_mapping=ResponseSpec(
            response_column="key_resp.corr",
            categories={"correct": 1, "incorrect": 0, "no_response": "n/a"},
            rt_column="key_resp.rt",
            rt_min=0.1,
            rt_max=1.5,
        ),
        binning=[
            BinSpec(
                name="rt_quartiles",
                method="quantile",
                n_bins=4,
                column="key_resp.rt",
                labels=["fast", "medium_fast", "medium_slow", "slow"],
            )
        ],
    )


@pytest.fixture(scope="module")
def sorter(three_condition_config):
    return TrialSorter(three_condition_config)


# ---------------------------------------------------------------------------
# SelectionRecord
# ---------------------------------------------------------------------------


class TestSelectionRecord:
    def test_fields_present(self, epochs, sorter):
        selected, record = sorter.select(epochs, conditions=["target"])

        assert isinstance(record, SelectionRecord)
        assert set(record.__dataclass_fields__) == {
            "spec",
            "n_before",
            "n_after",
            "mask_hash",
            "source_epochs_hash",
            "timestamp",
        }
        assert record.spec["conditions"] == ["target"]
        assert record.n_before == len(epochs)
        assert record.n_after == len(selected) == N_PER_CONDITION

    def test_hash_format(self, epochs, sorter):
        selected, record = sorter.select(epochs, conditions=["target"])

        assert len(record.mask_hash) == 64
        assert len(record.source_epochs_hash) == 64
        assert all(c in "0123456789abcdef" for c in record.mask_hash)
        assert all(c in "0123456789abcdef" for c in record.source_epochs_hash)

        # mask hash is the sha256 of the exact boolean mask used
        expected_mask = (epochs.metadata["trial_type"] == "target").to_numpy()
        assert record.mask_hash == hashlib.sha256(
            np.ascontiguousarray(expected_mask).tobytes()
        ).hexdigest()

        # Deterministic for an identical mask, different for a different one;
        # the source hash depends only on the input epochs.
        _sel_same, record_same = sorter.select(epochs, conditions=["target"])
        assert record_same.mask_hash == record.mask_hash
        _sel_other, record_other = sorter.select(epochs, conditions=["distractor"])
        assert record_other.mask_hash != record.mask_hash
        assert record_other.source_epochs_hash == record.source_epochs_hash

    def test_timestamp_is_utc_iso(self, epochs, sorter):
        _selected, record = sorter.select(epochs, conditions=["target"])
        parsed = datetime.fromisoformat(record.timestamp)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_sort_record_has_empty_mask_hash(self, epochs, sorter):
        _sorted, record = sorter.sort(epochs, by="difficulty")
        assert record.mask_hash == ""
        assert record.spec == {
            "sort_by": "difficulty",
            "ascending": True,
            "n_trials": len(epochs),
        }
        assert record.n_after == record.n_before == len(epochs)

    def test_source_epochs_hash_none_without_events(self):
        # Defensive: an object with neither filenames nor events still hashes.
        class _FakeEpochs:
            pass

        digest = TrialSorter._source_epochs_hash(_FakeEpochs())
        assert len(digest) == 64


# ---------------------------------------------------------------------------
# TrialSorter.select
# ---------------------------------------------------------------------------


class TestTrialSorterSelect:
    def test_by_conditions_subset(self, epochs, sorter):
        selected, record = sorter.select(epochs, conditions=["target", "distractor"])

        assert len(selected) == 2 * N_PER_CONDITION
        assert set(selected.metadata["trial_type"].unique()) == {
            "target",
            "distractor",
        }
        assert record.n_after == 2 * N_PER_CONDITION
        _assert_aligned(selected)

    def test_by_response_category(self, epochs, sorter):
        selected, _record = sorter.select(epochs, response="correct")
        assert (selected.metadata["key_resp.corr"] == 1).all()
        n_expected = int((epochs.metadata["key_resp.corr"] == 1).sum())
        assert len(selected) == n_expected

        selected_incorrect, _ = sorter.select(epochs, response="incorrect")
        assert (selected_incorrect.metadata["key_resp.corr"] == 0).all()
        _assert_aligned(selected)

    def test_by_accuracy(self, epochs, sorter):
        selected, _ = sorter.select(epochs, accuracy=1)
        assert (selected.metadata["key_resp.corr"] == 1).all()

        selected0, _ = sorter.select(epochs, accuracy=0)
        assert (selected0.metadata["key_resp.corr"] == 0).all()

        assert len(selected) + len(selected0) == int(
            epochs.metadata["key_resp.corr"].notna().sum()
        )

    @pytest.mark.parametrize(
        "bounds",
        [(0.5, None), (None, 0.5), (0.4, 0.6)],
        ids=["lo_only", "hi_only", "both"],
    )
    def test_by_rt_bounds(self, epochs, sorter, bounds):
        lo, hi = bounds
        selected, _ = sorter.select(epochs, rt=bounds)
        rts = selected.metadata["key_resp.rt"].to_numpy()

        expected = np.ones(len(epochs.metadata), dtype=bool)
        if lo is not None:
            assert (rts >= lo).all()
            expected &= (epochs.metadata["key_resp.rt"] >= lo).to_numpy()
        if hi is not None:
            assert (rts <= hi).all()
            expected &= (epochs.metadata["key_resp.rt"] <= hi).to_numpy()
        assert len(selected) == int(expected.sum())
        # missing RTs are excluded when a bound is set
        assert not np.isnan(rts).any()
        _assert_aligned(selected)

    def test_by_where_query_string(self, epochs, sorter):
        selected, record = sorter.select(epochs, where="`key_resp.rt` > 0.5")
        assert (selected.metadata["key_resp.rt"] > 0.5).all()
        assert record.spec["where"] == "`key_resp.rt` > 0.5"

        queried, _ = sorter.select(epochs, where="difficulty < 0.25")
        assert (queried.metadata["difficulty"] < 0.25).all()

    def test_combination(self, epochs, sorter):
        selected, _ = sorter.select(
            epochs, conditions=["target"], rt=(0.4, 0.7)
        )
        meta = selected.metadata
        assert (meta["trial_type"] == "target").all()
        assert ((meta["key_resp.rt"] >= 0.4) & (meta["key_resp.rt"] <= 0.7)).all()

    def test_group_returns_dict_of_bins(self, epochs, sorter):
        binned = sorter.select(epochs, group="rt_quartiles")

        assert isinstance(binned, dict)
        assert list(binned) == ["fast", "medium_fast", "medium_slow", "slow"]
        total = 0
        previous_max = -np.inf
        for label, (bin_epochs, record) in binned.items():
            assert isinstance(bin_epochs, mne.BaseEpochs)
            assert record.spec["group"] == "rt_quartiles"
            assert record.spec["bin"] == label
            assert record.n_before == len(epochs)
            assert record.n_after == len(bin_epochs) > 0
            assert record.mask_hash
            _assert_aligned(bin_epochs)
            total += len(bin_epochs)
            rts = bin_epochs.metadata["key_resp.rt"].to_numpy()
            assert rts.min() >= previous_max - 1e-12  # bins are ordered
            previous_max = rts.max()
        # NaN RTs are excluded from the bins, everything else is covered
        n_valid_rts = int(epochs.metadata["key_resp.rt"].notna().sum())
        assert total == n_valid_rts

    def test_group_with_real_cptvis_config(self, epochs, cptvis_sorter):
        binned = cptvis_sorter.select(epochs, conditions=["target"], group="rt_quartiles")
        assert list(binned) == ["fast", "medium_fast", "medium_slow", "slow"]
        for bin_epochs, _record in binned.values():
            assert (bin_epochs.metadata["trial_type"] == "target").all()

    def test_unknown_condition_raises(self, epochs, sorter):
        with pytest.raises(ValueError, match=r"Unknown condition.*bogus"):
            sorter.select(epochs, conditions=["bogus"])
        with pytest.raises(ValueError, match="target"):
            sorter.select(epochs, conditions=["bogus"])

    def test_bare_string_conditions_rejected(self, epochs, sorter):
        with pytest.raises(ValueError, match="bare string"):
            sorter.select(epochs, conditions="target")

    def test_unknown_response_raises(self, epochs, sorter):
        with pytest.raises(ValueError, match=r"Unknown response category.*bogus"):
            sorter.select(epochs, response="bogus")
        with pytest.raises(ValueError, match="correct"):
            sorter.select(epochs, response="bogus")

    def test_invalid_accuracy(self, epochs, sorter):
        with pytest.raises(ValueError, match="accuracy must be 0, 1 or None"):
            sorter.select(epochs, accuracy=2)

    def test_invalid_rt_spec(self, epochs, sorter):
        with pytest.raises(ValueError, match=r"rt must be a \(lo, hi\)"):
            sorter.select(epochs, rt=0.5)

    def test_unknown_column_raises(self, epochs, sorter):
        with pytest.raises(ValueError, match=r"Unknown column.*no_such_column"):
            sorter.select(epochs, where="no_such_column > 1")
        with pytest.raises(ValueError, match="Available columns"):
            sorter.select(epochs, where="no_such_column > 1")

    def test_empty_selection_raises(self, epochs, sorter):
        with pytest.raises(ValueError, match="produced no trials"):
            # RTs are 0.2..1.0 s, so 100..200 s can never match
            sorter.select(epochs, conditions=["target"], rt=(100.0, 200.0))

    def test_no_metadata_raises(self, sorter):
        info = mne.create_info(["EEG1"], 100.0, ["eeg"])
        plain = mne.EpochsArray(
            np.zeros((3, 1, 11)),
            info,
            events=np.column_stack(
                [np.arange(1, 4) * 25, np.zeros(3, dtype=int), np.ones(3, dtype=int)]
            ),
            tmin=-0.05,
            event_id={"stim": 1},
            verbose=False,
        )
        with pytest.raises(ValueError, match="No trial metadata"):
            sorter.select(plain, conditions=["target"])

    def test_metadata_length_mismatch_raises(self, epochs, sorter):
        with pytest.raises(ValueError, match="rows but epochs has"):
            sorter.select(epochs, metadata=epochs.metadata.iloc[:-1], conditions=["target"])

    def test_group_unknown_spec(self, epochs, sorter):
        with pytest.raises(ValueError, match=r"Unknown binning spec.*no_such_bin"):
            sorter.select(epochs, group="no_such_bin")


# ---------------------------------------------------------------------------
# TrialSorter.sort
# ---------------------------------------------------------------------------


class TestTrialSorterSort:
    def test_ascending(self, epochs, sorter):
        sorted_epochs, record = sorter.sort(epochs, by="difficulty")

        values = sorted_epochs.metadata["difficulty"].to_numpy()
        assert np.all(np.diff(values) >= 0)
        assert len(sorted_epochs) == len(epochs)
        assert record.n_after == record.n_before
        # epochs data follows metadata rows
        _assert_aligned(sorted_epochs)

    def test_descending(self, epochs, sorter):
        sorted_epochs, record = sorter.sort(epochs, by="difficulty", ascending=False)

        values = sorted_epochs.metadata["difficulty"].to_numpy()
        assert np.all(np.diff(values) <= 0)
        assert record.spec["ascending"] is False
        _assert_aligned(sorted_epochs)

    def test_sort_by_rt_nan_last(self, epochs, sorter):
        sorted_epochs, _ = sorter.sort(epochs, by="key_resp.rt")
        rts = sorted_epochs.metadata["key_resp.rt"].to_numpy()
        finite = rts[~np.isnan(rts)]
        assert np.all(np.diff(finite) >= 0)
        # NaNs are placed last (pandas default) and never interleave
        nan_positions = np.flatnonzero(np.isnan(rts))
        if nan_positions.size:
            assert nan_positions[0] > np.flatnonzero(~np.isnan(rts)).max()

    def test_sort_is_stable_for_ties(self, epochs, sorter):
        sorted_epochs, _ = sorter.sort(epochs, by="key_resp.corr")
        ties = sorted_epochs.metadata[sorted_epochs.metadata["key_resp.corr"] == 1]
        indices = ties["trial_index"].to_numpy()
        assert np.all(np.diff(indices) > 0)

    def test_unknown_column_raises(self, epochs, sorter):
        with pytest.raises(ValueError, match=r"Unknown column|not found"):
            sorter.sort(epochs, by="no_such_column")

    def test_sort_does_not_reuse_stale_metadata(self, sorter):
        """Metadata is resolved at call time, not at construction."""
        epochs = _synthetic_epochs_with_metadata()
        assert epochs.metadata is not None
        sorted_epochs, _ = sorter.sort(epochs, by="trial_index", ascending=False)
        assert sorted_epochs.metadata["trial_index"].iloc[0] == len(epochs) - 1


# ---------------------------------------------------------------------------
# TrialSorter.split
# ---------------------------------------------------------------------------


class TestTrialSorterSplit:
    def test_position_halves(self, epochs, sorter):
        parts = sorter.split(epochs, by="position", parts=2)

        assert list(parts) == ["half_1", "half_2"]
        sizes = [len(part_epochs) for part_epochs, _ in parts.values()]
        assert sizes == [len(epochs) // 2, len(epochs) - len(epochs) // 2]
        # first half is the first trials, second half the rest
        assert parts["half_1"][0].metadata["trial_index"].iloc[0] == 0
        assert (
            parts["half_2"][0].metadata["trial_index"].iloc[0]
            == len(epochs) // 2
        )
        for part_epochs, record in parts.values():
            assert record.n_before == len(epochs)
            assert record.n_after == len(part_epochs)
            assert record.mask_hash
            _assert_aligned(part_epochs)

    def test_position_thirds(self, epochs, sorter):
        parts = sorter.split(epochs, by="position", parts=3)
        assert list(parts) == ["half_1", "half_2", "half_3"]
        assert sum(len(part_epochs) for part_epochs, _ in parts.values()) == len(epochs)

    def test_column_quantiles(self, epochs, sorter):
        parts = sorter.split(epochs, by="difficulty", parts=4)

        assert list(parts) == ["bin_1", "bin_2", "bin_3", "bin_4"]
        assert sum(len(part_epochs) for part_epochs, _ in parts.values()) == len(epochs)
        lowers = [
            part_epochs.metadata["difficulty"].min()
            for part_epochs, _ in parts.values()
        ]
        uppers = [
            part_epochs.metadata["difficulty"].max()
            for part_epochs, _ in parts.values()
        ]
        assert lowers == sorted(lowers)
        for earlier_max, later_min in zip(uppers[:-1], lowers[1:]):
            assert earlier_max <= later_min + 1e-12
        for part_epochs, record in parts.values():
            assert record.spec["split_by"] == "difficulty"
            _assert_aligned(part_epochs)

    def test_split_column_with_missing_values(self, epochs, sorter):
        parts = sorter.split(epochs, by="key_resp.rt", parts=2)
        n_valid = int(epochs.metadata["key_resp.rt"].notna().sum())
        assert sum(len(part_epochs) for part_epochs, _ in parts.values()) == n_valid

    def test_split_unknown_column(self, epochs, sorter):
        with pytest.raises(ValueError, match="not found"):
            sorter.split(epochs, by="no_such_column")

    def test_split_parts_exceed_trials(self, sorter):
        small = _synthetic_epochs_with_metadata(n_per_condition=1)
        with pytest.raises(ValueError, match="Cannot split"):
            sorter.split(small, by="position", parts=5)

    def test_split_invalid_parts(self, epochs, sorter):
        with pytest.raises(ValueError, match="positive integer"):
            sorter.split(epochs, by="position", parts=0)


# ---------------------------------------------------------------------------
# legacy UI-facing API (page 9 return shapes)
# ---------------------------------------------------------------------------


class TestLegacyReturnShape:
    def test_sort_by_condition_shape(self, epochs, cptvis_sorter):
        sorted_by_condition = cptvis_sorter.sort_by_condition(epochs)

        assert set(sorted_by_condition) == {"target", "nontarget"}
        for name, info in sorted_by_condition.items():
            assert set(info) >= {"epochs", "n_trials", "description"}
            assert info["n_trials"] == len(info["epochs"]) == N_PER_CONDITION
            assert isinstance(info["selection_record"], SelectionRecord)
            assert info["selection_record"].mask_hash

    def test_sort_by_condition_event_code_fallback(self, cptvis_sorter):
        # event names are annotation-style; cptvis condition event_codes
        # ("target"/"nontarget") match neither the names nor the codes
        epochs = _synthetic_epochs_with_metadata(
            event_id={
                "Stimulus/S 1": 1,
                "Stimulus/S 2": 2,
                "Stimulus/S 3": 3,
            }
        )
        with pytest.raises(ValueError, match="could not be matched"):
            cptvis_sorter.sort_by_condition(epochs)

    def test_sort_by_response_shape(self, epochs, sorter):
        sorted_by_response = sorter.sort_by_response(epochs)

        assert set(sorted_by_response) == {"correct", "incorrect", "no_response"}
        for info in sorted_by_response.values():
            assert set(info) >= {"epochs", "n_trials"}
            assert info["n_trials"] == len(info["epochs"])
        assert (
            sorted_by_response["correct"]["n_trials"]
            + sorted_by_response["incorrect"]["n_trials"]
            == int(epochs.metadata["key_resp.corr"].notna().sum())
        )

    def test_sort_by_response_positional_dataframe(self, epochs, sorter):
        # page 9 calls sort_by_response(epochs, metadata) positionally
        sorted_by_response = sorter.sort_by_response(epochs, epochs.metadata)
        assert sorted_by_response["correct"]["n_trials"] == int(
            (epochs.metadata["key_resp.corr"] == 1).sum()
        )

    def test_sort_by_response_requires_metadata(self, sorter):
        info = mne.create_info(["EEG1"], 100.0, ["eeg"])
        plain = mne.EpochsArray(
            np.zeros((3, 1, 11)),
            info,
            events=np.column_stack(
                [np.arange(1, 4) * 25, np.zeros(3, dtype=int), np.ones(3, dtype=int)]
            ),
            tmin=-0.05,
            event_id={"stim": 1},
            verbose=False,
        )
        with pytest.raises(ValueError, match="No trial metadata"):
            sorter.sort_by_response(plain)

    def test_bin_trials_shape(self, epochs, sorter, three_condition_config):
        bin_spec = three_condition_config.binning[0]
        binned = sorter.bin_trials(epochs, bin_spec=bin_spec)

        assert list(binned) == ["fast", "medium_fast", "medium_slow", "slow"]
        for info in binned.values():
            assert set(info) >= {"epochs", "n_trials"}
            assert info["n_trials"] == len(info["epochs"])

    def test_bin_trials_positional_dataframe(self, epochs, sorter, three_condition_config):
        # page 9 calls bin_trials(epochs, metadata, bin_spec) positionally
        bin_spec = three_condition_config.binning[0]
        binned = sorter.bin_trials(epochs, epochs.metadata, bin_spec)
        assert sum(info["n_trials"] for info in binned.values()) == int(
            epochs.metadata["key_resp.rt"].notna().sum()
        )

    def test_bin_trials_unknown_method_raises(self, epochs, sorter):
        bad_spec = BinSpec(name="bad", method="mystery", n_bins=2, column="difficulty")
        with pytest.raises(ValueError, match="Unknown binning method"):
            sorter.bin_trials(epochs, bin_spec=bad_spec)


# ---------------------------------------------------------------------------
# trial_filter predicate layer
# ---------------------------------------------------------------------------


class TestTrialFilterPredicates:
    def test_build_mask_conditions(self):
        df = _synthetic_metadata(n_per_condition=10)
        mask = tf.build_mask(df, conditions=["target", "nontarget"])
        assert mask.dtype == bool
        assert len(mask) == len(df)
        assert mask.sum() == 20

    def test_build_mask_conditions_numeric_normalisation(self):
        df = pd.DataFrame({"trial_type": [1.0, 2.0, "3", None]})
        mask = tf.build_mask(df, conditions=[1, "2", 3.0])
        np.testing.assert_array_equal(mask, [True, True, True, False])

    def test_build_mask_response(self):
        df = _synthetic_metadata(n_per_condition=10)
        mask = tf.build_mask(
            df, response=1, response_column="key_resp.corr"
        )
        np.testing.assert_array_equal(
            mask, (df["key_resp.corr"] == 1).to_numpy()
        )

    def test_build_mask_accuracy(self):
        df = _synthetic_metadata(n_per_condition=10)
        mask = tf.build_mask(
            df, accuracy=1, accuracy_column="key_resp.corr"
        )
        np.testing.assert_array_equal(
            mask, (df["key_resp.corr"] == 1).to_numpy()
        )

    def test_build_mask_rt_bounds(self):
        df = _synthetic_metadata(n_per_condition=10)
        mask = tf.build_mask(df, rt=(0.4, 0.6), rt_column="key_resp.rt")
        expected = (
            (df["key_resp.rt"] >= 0.4) & (df["key_resp.rt"] <= 0.6)
        ).to_numpy()
        np.testing.assert_array_equal(mask, expected)
        assert not mask[df["key_resp.rt"].isna().to_numpy()].any()

    def test_build_mask_where(self):
        df = _synthetic_metadata(n_per_condition=10)
        mask = tf.build_mask(df, where="`key_resp.rt` > 0.5 and difficulty < 0.9")
        expected = (
            (df["key_resp.rt"] > 0.5) & (df["difficulty"] < 0.9)
        ).to_numpy()
        np.testing.assert_array_equal(mask, expected)

        with pytest.raises(ValueError, match="Unknown column"):
            tf.build_mask(df, where="no_such_column > 1")

    def test_build_mask_combinations(self):
        df = _synthetic_metadata(n_per_condition=10)
        mask = tf.build_mask(
            df,
            conditions=["target"],
            rt=(0.3, 0.9),
            rt_column="key_resp.rt",
        )
        expected = (
            (df["trial_type"] == "target")
            & (df["key_resp.rt"] >= 0.3)
            & (df["key_resp.rt"] <= 0.9)
        ).to_numpy()
        np.testing.assert_array_equal(mask, expected)

    def test_build_mask_combinations_all_predicates(self):
        """where + conditions + response + accuracy + rt are ANDed."""
        df = _synthetic_metadata(n_per_condition=10)
        mask = tf.build_mask(
            df,
            where="difficulty >= 0",
            conditions=["target", "nontarget"],
            response=1,
            response_column="key_resp.corr",
            accuracy=1,
            accuracy_column="key_resp.corr",
            rt=(0.3, 0.8),
            rt_column="key_resp.rt",
        )
        expected = (
            (df["trial_type"].isin(["target", "nontarget"]))
            & (df["key_resp.corr"] == 1)
            & (df["key_resp.rt"] >= 0.3)
            & (df["key_resp.rt"] <= 0.8)
        ).to_numpy()
        np.testing.assert_array_equal(mask, expected)

    def test_build_mask_unknown_trial_type_column(self):
        df = pd.DataFrame({"condition": ["a"]})
        with pytest.raises(ValueError, match="trial_type"):
            tf.build_mask(df, conditions=["a"])

    def test_build_mask_response_without_column(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="response_column is required"):
            tf.build_mask(df, response=1)

    def test_build_mask_rt_without_column(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="rt_column is required"):
            tf.build_mask(df, rt=(0.1, 0.2))

    def test_query_mask_good(self):
        df = _synthetic_metadata(n_per_condition=10)
        mask = tf.query_mask(df, "`key_resp.rt` > 0.5")
        np.testing.assert_array_equal(mask, (df["key_resp.rt"] > 0.5).to_numpy())

    def test_query_mask_gappy_index(self):
        df = _synthetic_metadata(n_per_condition=10).iloc[[0, 5, 27]]
        mask = tf.query_mask(df, "difficulty >= 0")
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_query_mask_unknown_column(self):
        df = _synthetic_metadata(n_per_condition=10)
        with pytest.raises(ValueError, match="Unknown column"):
            tf.query_mask(df, "no_such_column > 1")

    def test_query_mask_syntax_error(self):
        df = _synthetic_metadata(n_per_condition=10)
        with pytest.raises(ValueError, match="Invalid where clause"):
            tf.query_mask(df, "difficulty >")

    def test_query_mask_empty_string(self):
        df = _synthetic_metadata(n_per_condition=10)
        with pytest.raises(ValueError, match="non-empty"):
            tf.query_mask(df, "   ")

    def test_legacy_filters_still_work(self):
        df = _synthetic_metadata(n_per_condition=10)

        rt_filtered = tf.filter_by_rt(
            df, rt_column="key_resp.rt", rt_min=0.3, rt_max=0.8
        )
        assert ((rt_filtered["key_resp.rt"] >= 0.3) & (rt_filtered["key_resp.rt"] <= 0.8)).all()

        correct = tf.filter_by_accuracy(
            df, accuracy_column="key_resp.corr", keep="correct"
        )
        assert (correct["key_resp.corr"] == 1).all()
        with pytest.raises(ValueError, match="Invalid keep value"):
            tf.filter_by_accuracy(df, accuracy_column="key_resp.corr", keep="maybe")

        no_missing = tf.drop_missing_responses(df, rt_column="key_resp.rt")
        assert no_missing["key_resp.rt"].notna().all()
        assert (no_missing["key_resp.rt"] > 0).all()

        trimmed = tf.drop_sequential_positions(df, drop_first_n=5, drop_last_n=5)
        assert len(trimmed) == len(df) - 10

    def test_drop_missing_responses_unknown_column_raises(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            tf.drop_missing_responses(df)

    def test_apply_trial_filters_pipeline(self):
        df = _synthetic_metadata(n_per_condition=10)

        class Config:
            drop_first_n = 3
            drop_last_n = 2
            drop_missing_responses = True
            rt_field = "key_resp.rt"
            rt_min = 0.25
            rt_max = 0.95
            accuracy_filter = "correct"
            accuracy_field = "key_resp.corr"

        filtered = tf.apply_trial_filters(df, Config())

        assert (filtered["key_resp.corr"] == 1).all()
        assert ((filtered["key_resp.rt"] >= 0.25) & (filtered["key_resp.rt"] <= 0.95)).all()
        # sequential filter ran first: result is a subset of the middle slice
        middle = df.iloc[3:-2]
        assert len(filtered) <= len(middle)
        assert set(filtered["trial_index"]).issubset(set(middle["trial_index"]))

    def test_apply_trial_filters_missing_rt_field_raises(self):
        df = _synthetic_metadata(n_per_condition=10)

        class Config:
            drop_first_n = 0
            drop_last_n = 0
            drop_missing_responses = True
            rt_field = "nonexistent_rt"
            rt_min = None
            rt_max = None
            accuracy_filter = None
            accuracy_field = "key_resp.corr"

        with pytest.raises(ValueError, match="nonexistent_rt"):
            tf.apply_trial_filters(df, Config())
