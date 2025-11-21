import pytest

from gemini_batch.aggregation import (
    aggregate_records,
    ListVoteConfig,
    MajorityVoteAggregator,
)


def test_scalar_majority_vote():
    records = [
        {"urkunde": {"nummer": "356"}},
        {"urkunde": {"nummer": "356"}},
        {"urkunde": {"nummer": "355"}},
    ]

    result = aggregate_records(records)

    assert result.aggregated["urkunde"]["nummer"] == "356"
    vote = result.field_votes[("urkunde", "nummer")]
    assert vote.agreement == pytest.approx(2 / 3)
    assert not vote.is_tie


def test_tie_detection_defaults_to_stable_choice():
    records = [
        {"urkunde": {"standesbeamter": "Sigl"}},
        {"urkunde": {"standesbeamter": "Vogel"}},
    ]

    result = aggregate_records(records)

    vote = result.field_votes[("urkunde", "standesbeamter")]
    assert vote.is_tie
    assert vote.value in {"Sigl", "Vogel"}


def test_list_alignment_by_index():
    records = [
        {"beziehungen": [{"name": "Anna", "beziehungstyp": "Mutter"}]},
        {"beziehungen": [{"name": "Anna", "beziehungstyp": "Mutter"}]},
        {"beziehungen": [{"name": "Anna", "beziehungstyp": "Vater"}]},
    ]

    result = aggregate_records(records)

    assert result.aggregated["beziehungen"][0]["beziehungstyp"] == "Mutter"
    vote = result.field_votes[("beziehungen", "[0]", "beziehungstyp")]
    assert vote.agreement == pytest.approx(2 / 3)


def test_list_alignment_with_match_keys():
    records = [
        {"beziehungen": [{"name": "Anna", "beziehungstyp": "Mutter", "wohnort": {"stadt": "München"}}]},
        {"beziehungen": [{"name": "Anna", "beziehungstyp": "Mutter", "wohnort": {"stadt": "Muenchen"}}]},
        {"beziehungen": [{"name": "Anna", "beziehungstyp": "Mutter", "wohnort": {"stadt": "München"}}]},
    ]

    config = {("beziehungen",): ListVoteConfig(match_on=("name", "beziehungstyp"))}

    result = aggregate_records(records, list_configs=config)

    path = ("beziehungen", "[Anna, Mutter]", "wohnort", "stadt")
    vote = result.field_votes[path]
    assert vote.value == "München"
    assert vote.agreement == pytest.approx(2 / 3)


def test_custom_tie_breaker():
    records = [
        {"urkunde": {"standesbeamter": "Sigl"}},
        {"urkunde": {"standesbeamter": "Vogel"}},
    ]

    def prefer_sigl(_path, winners):
        # Custom tie breaker picking Sigl when available.
        for winner in winners:
            if winner == "Sigl":
                return winner
        return winners[0]

    aggregator = MajorityVoteAggregator(tie_breaker=prefer_sigl)
    result = aggregator.aggregate(records)

    vote = result.field_votes[("urkunde", "standesbeamter")]
    assert vote.is_tie
    assert vote.value == "Sigl"
