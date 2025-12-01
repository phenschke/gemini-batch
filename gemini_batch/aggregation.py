"""
WIP.
Utilities for aggregating multiple structured extraction results via majority voting.

The aggregator works on arbitrary JSON-like data (dict/list/scalar) and records
per-field agreement statistics while constructing a consensus object. Nested objects
and list-of-object structures are handled recursively, with configurable strategies
for aligning list items.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import json
import enum

from pydantic import BaseModel

from .types import Path, ValueCount, FieldVote, ListVoteConfig, MajorityVoteResult


class MajorityVoteAggregator:
    """
    Aggregate structured data samples via majority vote.

    Usage:
        aggregator = MajorityVoteAggregator(
            list_configs={
                ("beziehungen",): ListVoteConfig(match_on=("name", "beziehungstyp"))
            }
        )
        result = aggregator.aggregate(records, as_model=config.Sterbeurkunde)
    """

    def __init__(
        self,
        list_configs: Optional[Mapping[Path, ListVoteConfig]] = None,
        tie_breaker: Optional[Callable[[Path, Sequence[Any]], Any]] = None,
    ) -> None:
        self._list_configs: Dict[Path, ListVoteConfig] = {
            tuple(path): cfg for path, cfg in (list_configs or {}).items()
        }
        self._tie_breaker = tie_breaker
        self._n_samples: int = 0
        self._field_votes: Dict[Path, FieldVote] = {}

    # ------------------------------------------------------------------ public API
    def aggregate(
        self,
        records: Sequence[Any],
        *,
        as_model: Optional[Any] = None,
    ) -> MajorityVoteResult:
        """
        Aggregate a sequence of structured records.

        Args:
            records: Sequence of dict / list / scalar structures (or Pydantic models).
            as_model: Optional Pydantic BaseModel subclass to validate the aggregated
                result before returning. The validated model is returned in `result.aggregated`.

        Returns:
            MajorityVoteResult containing the aggregated structure and per-field statistics.
        """
        if not records:
            raise ValueError("No records provided for aggregation.")

        plain_records = [self._to_plain(record) for record in records]
        self._n_samples = len(plain_records)
        self._field_votes = {}

        aggregated, _ = self._aggregate_path((), plain_records)

        if as_model is not None:
            if BaseModel is None:
                raise RuntimeError("Pydantic BaseModel is required for model validation.")
            aggregated = as_model.model_validate(aggregated)

        result = MajorityVoteResult(aggregated=aggregated, field_votes=dict(self._field_votes), n_samples=self._n_samples)

        # Reset mutable state to avoid leaking between runs.
        self._n_samples = 0
        self._field_votes = {}
        return result

    # ----------------------------------------------------------------- internal API
    def _to_plain(self, value: Any) -> Any:
        """Convert supported data types (e.g., BaseModel, Enum) into plain Python types."""
        if BaseModel is not None and isinstance(value, BaseModel):
            return self._to_plain(value.model_dump())
        if isinstance(value, dict):
            return {str(k): self._to_plain(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_plain(item) for item in value]
        if isinstance(value, enum.Enum):
            return value.value
        return value

    def _aggregate_path(self, path: Path, values: Sequence[Any]) -> Tuple[Any, bool]:
        """
        Aggregate values observed at a given path.

        Returns:
            Tuple of (aggregated_value, value_present) where value_present indicates whether
            any non-null/non-empty value was present among the samples.
        """
        non_null = [v for v in values if v is not None]
        if not non_null:
            vote = self._vote_scalar(path, values)
            return vote.value, False

        if all(isinstance(v, dict) for v in non_null):
            aggregated = self._aggregate_dict(path, values)
            return aggregated, True

        if all(isinstance(v, list) for v in non_null):
            aggregated = self._aggregate_list(path, values)
            return aggregated, True

        vote = self._vote_scalar(path, values)
        return vote.value, vote.value is not None

    def _aggregate_dict(self, path: Path, values: Sequence[Any]) -> Dict[str, Any]:
        aggregated: Dict[str, Any] = {}
        keys = set()
        for value in values:
            if isinstance(value, dict):
                keys.update(value.keys())
        for key in sorted(keys):
            child_values = [
                value.get(key) if isinstance(value, dict) else None
                for value in values
            ]
            child_path = path + (key,)
            aggregated_value, present = self._aggregate_path(child_path, child_values)
            if present or any(val is not None for val in child_values):
                aggregated[key] = aggregated_value
        return aggregated

    def _aggregate_list(self, path: Path, values: Sequence[Any]) -> List[Any]:
        config = self._list_configs.get(path)
        aligned: Dict[Tuple[str, Tuple[Any, ...]], List[Any]] = {}
        ordering: Dict[Tuple[str, Tuple[Any, ...]], List[int]] = {}

        for sample_index, sequence in enumerate(values):
            if not isinstance(sequence, list):
                continue
            for position, item in enumerate(sequence):
                key = self._resolve_list_key(config, item, position)
                if key not in aligned:
                    aligned[key] = [None] * self._n_samples
                    ordering[key] = []
                aligned[key][sample_index] = item
                ordering[key].append(position)

        aggregated_items: List[Any] = []
        for key, item_values in sorted(
            aligned.items(),
            key=lambda kv: self._list_sort_key(kv[0], ordering),
        ):
            label = self._format_list_key(key)
            child_path = path + (label,)
            aggregated_value, present = self._aggregate_path(child_path, item_values)
            if present and aggregated_value is not None:
                aggregated_items.append(aggregated_value)

        # Record a vote for the number of items to capture disagreements about list length.
        self._vote_scalar(path + ("#count",), [len(seq) if isinstance(seq, list) else 0 for seq in values])
        # The value returned by length vote is not needed for reconstruction, but we keep the stats.
        return aggregated_items

    def _resolve_list_key(
        self,
        config: Optional[ListVoteConfig],
        item: Any,
        position: int,
    ) -> Tuple[str, Tuple[Any, ...]]:
        if config is None or not isinstance(item, dict) or not config.match_on:
            return ("index", (position,))

        key_values: List[Any] = []
        missing = False
        for field_path in config.match_on:
            value = self._lookup_field(item, field_path)
            key_values.append(value)
            if value in (None, "", []):
                missing = True
        if missing and config.require_all_fields:
            return ("index", (position,))
        if all(value in (None, "", []) for value in key_values):
            return ("index", (position,))
        return ("match", tuple(key_values))

    def _list_sort_key(
        self,
        key: Tuple[str, Tuple[Any, ...]],
        ordering: Dict[Tuple[str, Tuple[Any, ...]], List[int]],
    ) -> Tuple[int, int, Tuple[Any, ...]]:
        source, identity = key
        positions = ordering.get(key, [])
        first_pos = min(positions) if positions else 0
        # Match-aligned items should come before pure index-aligned ones.
        priority = 0 if source == "match" else 1
        return (priority, first_pos, identity)

    def _format_list_key(self, key: Tuple[str, Tuple[Any, ...]]) -> str:
        source, identity = key
        if source == "match":
            identity_str = ", ".join("" if v is None else str(v) for v in identity)
            return f"[{identity_str}]"
        return f"[{identity[0]}]"

    def _vote_scalar(self, path: Path, values: Sequence[Any]) -> FieldVote:
        tally: MutableMapping[str, ValueCount] = {}
        for value in values:
            normalized = self._normalize_value(value)
            if normalized not in tally:
                tally[normalized] = ValueCount(value=value, count=0)
            tally[normalized] = ValueCount(
                value=tally[normalized].value,
                count=tally[normalized].count + 1,
            )

        ranked = sorted(
            tally.values(),
            key=lambda vc: (-vc.count, self._stable_label(vc.value)),
        )

        if not ranked:
            ranked = [ValueCount(value=None, count=0)]

        top_count = ranked[0].count
        winners = [vc.value for vc in ranked if vc.count == top_count]
        if len(winners) > 1 and self._tie_breaker is not None:
            chosen = self._tie_breaker(path, winners)
        else:
            chosen = winners[0]

        vote = FieldVote(path=path, value=chosen, total=self._n_samples, counts=ranked)
        self._field_votes[path] = vote
        return vote

    def _normalize_value(self, value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return self._stable_label(value)

    def _stable_label(self, value: Any) -> str:
        return repr(value)

    def _lookup_field(self, item: Mapping[str, Any], dotted_path: str) -> Any:
        current: Any = item
        for part in dotted_path.split("."):
            if not isinstance(current, Mapping):
                return None
            current = current.get(part)
        return current


def aggregate_records(
    records: Sequence[Any],
    *,
    list_configs: Optional[Mapping[Path, ListVoteConfig]] = None,
    tie_breaker: Optional[Callable[[Path, Sequence[Any]], Any]] = None,
    as_model: Optional[Any] = None,
) -> MajorityVoteResult:
    """
    Convenience wrapper around MajorityVoteAggregator.aggregate().
    """
    aggregator = MajorityVoteAggregator(list_configs=list_configs, tie_breaker=tie_breaker)
    return aggregator.aggregate(records, as_model=as_model)
