"""Type definitions for gemini-batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

# Type aliases
Path = Tuple[str, ...]


@dataclass(frozen=True)
class ValueCount:
    """Count of how many samples voted for a given value."""
    value: Any
    count: int


@dataclass
class FieldVote:
    """Agreement statistics for a specific JSON path."""
    path: Path
    value: Any
    total: int
    counts: List[ValueCount]

    @property
    def agreement(self) -> float:
        if not self.counts or self.total == 0:
            return 0.0
        return self.counts[0].count / self.total

    @property
    def is_tie(self) -> bool:
        if len(self.counts) < 2:
            return False
        top = self.counts[0].count
        return any(c.count == top for c in self.counts[1:])


@dataclass(frozen=True)
class ListVoteConfig:
    """
    Configure how list items should be aligned before voting.

    Attributes:
        match_on: Optional tuple of dotted field paths used to align list items that
            describe the same entity across samples. When omitted, items are aligned
            purely by index order.
        require_all_fields: When True, only use the match key if *all* match_on fields
            are present and non-empty; otherwise fall back to index alignment.
    """
    match_on: Optional[Tuple[str, ...]] = None
    require_all_fields: bool = False


@dataclass
class MajorityVoteResult:
    """Result of aggregating multiple structured samples."""
    aggregated: Any
    field_votes: Dict[Path, FieldVote]
    n_samples: int

    def disagreements(self) -> Dict[Path, FieldVote]:
        """Return only the paths where agreement is not unanimous."""
        return {path: vote for path, vote in self.field_votes.items() if vote.agreement < 1.0}


class TokenStatistics(BaseModel):
    """Token usage statistics aggregated from batch processing metadata."""

    # Request counts
    total_requests: int
    successful_requests: int
    failed_requests: int

    # Total token counts (sum across all successful requests)
    total_prompt_tokens: int
    total_candidates_tokens: int
    total_tokens: int
    total_cached_tokens: int
    total_thoughts_tokens: int

    # Average tokens per successful request (None if no successful requests)
    avg_prompt_tokens: Optional[float]
    avg_candidates_tokens: Optional[float]
    avg_total_tokens: Optional[float]
    avg_cached_tokens: Optional[float]
    avg_thoughts_tokens: Optional[float]
