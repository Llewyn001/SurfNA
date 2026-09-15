"""Batch samplers that never split a pose-ranking group.

Ranking supervision is defined within a complex.  Sampling manifest rows with
an ordinary DataLoader silently destroys listwise and pairwise losses whenever
a group crosses a batch or distributed-rank boundary.  These samplers shuffle
and shard *groups*, then emit every row of each selected group together.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Hashable, Iterable, Iterator, Mapping, Sequence

from torch.utils.data import Sampler


def build_group_index(group_keys: Sequence[Hashable]) -> dict[Hashable, tuple[int, ...]]:
    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        grouped[key].append(index)
    if not grouped:
        raise ValueError("group_keys is empty")
    return {key: tuple(indices) for key, indices in grouped.items()}


class DistributedGroupBatchSampler(Sampler[list[int]]):
    """Shard complete groups across ranks and pack them into batches.

    Parameters
    ----------
    group_keys:
        One stable group key per dataset row.  Use ``pose_group_uid`` rather
        than a process-local integer.
    groups_per_batch:
        Number of complete groups placed in a batch.  ``1`` is recommended for
        the 20-pose curated scorer groups.
    num_replicas, rank:
        Distributed world size and rank.  Groups are assigned by striding a
        common deterministic order, so no group can appear on two ranks.
    drop_rank_tail:
        Training with DDP requires every rank to execute the same number of
        backward calls.  When true, at most ``num_replicas - 1`` groups are
        dropped before sharding.  Keep false for validation/test, where every
        group must be scored exactly once and metrics are gathered as objects.
    """

    def __init__(
        self,
        group_keys: Sequence[Hashable],
        *,
        groups_per_batch: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        drop_rank_tail: bool = True,
        drop_batch_tail: bool = False,
    ) -> None:
        if groups_per_batch < 1:
            raise ValueError("groups_per_batch must be >= 1")
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.group_to_indices = build_group_index(group_keys)
        self.group_keys = tuple(sorted(self.group_to_indices, key=str))
        self.groups_per_batch = int(groups_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_rank_tail = bool(drop_rank_tail)
        self.drop_batch_tail = bool(drop_batch_tail)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ordered_groups(self) -> list[Hashable]:
        groups = list(self.group_keys)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        if self.drop_rank_tail and self.num_replicas > 1:
            usable = len(groups) - (len(groups) % self.num_replicas)
            groups = groups[:usable]
        return groups

    def groups_for_rank(self) -> tuple[Hashable, ...]:
        return tuple(self._ordered_groups()[self.rank :: self.num_replicas])

    def __iter__(self) -> Iterator[list[int]]:
        local_groups = self.groups_for_rank()
        stop = len(local_groups)
        if self.drop_batch_tail:
            stop -= stop % self.groups_per_batch
        for start in range(0, stop, self.groups_per_batch):
            selected = local_groups[start : start + self.groups_per_batch]
            if len(selected) < self.groups_per_batch and self.drop_batch_tail:
                break
            batch: list[int] = []
            for group_key in selected:
                batch.extend(self.group_to_indices[group_key])
            yield batch

    def __len__(self) -> int:
        n_groups = len(self.groups_for_rank())
        if self.drop_batch_tail:
            return n_groups // self.groups_per_batch
        return math.ceil(n_groups / self.groups_per_batch)

    def audit(self) -> dict[str, object]:
        """Return invariants suitable for logging before a run starts."""
        local = self.groups_for_rank()
        emitted = list(iter(self))
        emitted_indices = [index for batch in emitted for index in batch]
        expected_indices = [index for key in local for index in self.group_to_indices[key]]
        return {
            "n_global_groups": len(self.group_keys),
            "n_local_groups": len(local),
            "n_local_batches": len(emitted),
            "n_local_rows": len(emitted_indices),
            "duplicate_local_rows": len(emitted_indices) - len(set(emitted_indices)),
            "missing_local_rows": len(set(expected_indices) - set(emitted_indices)),
            "groups_split_across_batches": self._count_split_groups(emitted),
            "drop_rank_tail": self.drop_rank_tail,
            "drop_batch_tail": self.drop_batch_tail,
            "rank": self.rank,
            "num_replicas": self.num_replicas,
        }

    def _count_split_groups(self, batches: Iterable[Sequence[int]]) -> int:
        batch_by_index = {}
        for batch_number, batch in enumerate(batches):
            for index in batch:
                batch_by_index[index] = batch_number
        split = 0
        for indices in self.group_to_indices.values():
            locations = {batch_by_index[index] for index in indices if index in batch_by_index}
            split += int(len(locations) > 1)
        return split


class GroupBatchSampler(DistributedGroupBatchSampler):
    """Single-process convenience wrapper."""

    def __init__(
        self,
        group_keys: Sequence[Hashable],
        *,
        groups_per_batch: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_batch_tail: bool = False,
    ) -> None:
        super().__init__(
            group_keys,
            groups_per_batch=groups_per_batch,
            shuffle=shuffle,
            seed=seed,
            num_replicas=1,
            rank=0,
            drop_rank_tail=False,
            drop_batch_tail=drop_batch_tail,
        )


def _allocate_exact_counts(
    probabilities: Mapping[str, float], total: int
) -> dict[str, int]:
    """Largest-remainder allocation with deterministic lexical tie breaking."""
    if total < 1:
        raise ValueError("total must be positive")
    values = {str(key): float(value) for key, value in probabilities.items()}
    if not values or any(value < 0 for value in values.values()):
        raise ValueError("sampling probabilities must be non-empty and non-negative")
    normalizer = sum(values.values())
    if normalizer <= 0:
        raise ValueError("sampling probabilities must have positive mass")
    values = {key: value / normalizer for key, value in values.items()}
    exact = {key: value * total for key, value in values.items()}
    counts = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(values, key=lambda key: (-(exact[key] - counts[key]), key))
    for key in order[:remaining]:
        counts[key] += 1
    return counts


class DistributionAwareDistributedGroupBatchSampler(DistributedGroupBatchSampler):
    """Deterministically match a frozen validation difficulty distribution.

    A complete pose group remains the atomic sampling unit.  Rare difficulty
    bins may be sampled more than once per epoch and common bins may be sampled
    less often; duplicates are intentional *group exposures*, never split rows.
    The common global exposure sequence is constructed before DDP striding, so
    every rank executes the same number of backward calls.
    """

    def __init__(
        self,
        group_keys: Sequence[Hashable],
        *,
        difficulty_by_group: Mapping[Hashable, str],
        target_probabilities: Mapping[str, float],
        samples_per_epoch: int | None = None,
        groups_per_batch: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        drop_rank_tail: bool = True,
        drop_batch_tail: bool = False,
    ) -> None:
        super().__init__(
            group_keys,
            groups_per_batch=groups_per_batch,
            shuffle=shuffle,
            seed=seed,
            num_replicas=num_replicas,
            rank=rank,
            drop_rank_tail=drop_rank_tail,
            drop_batch_tail=drop_batch_tail,
        )
        missing = set(self.group_keys) - set(difficulty_by_group)
        if missing:
            raise ValueError(f"missing group difficulty labels: {sorted(map(str, missing))[:5]}")
        self.difficulty_by_group = {
            key: str(difficulty_by_group[key]) for key in self.group_keys
        }
        self.target_probabilities = {
            str(key): float(value) for key, value in target_probabilities.items()
        }
        self.samples_per_epoch = int(samples_per_epoch or len(self.group_keys))
        groups_by_bin: dict[str, list[Hashable]] = defaultdict(list)
        for key in self.group_keys:
            groups_by_bin[self.difficulty_by_group[key]].append(key)
        required_empty = [
            label
            for label, probability in self.target_probabilities.items()
            if probability > 0 and not groups_by_bin.get(label)
        ]
        if required_empty:
            raise ValueError(
                "target distribution assigns mass to absent bins: "
                f"{required_empty}"
            )
        self.target_counts = _allocate_exact_counts(
            self.target_probabilities, self.samples_per_epoch
        )
        self.groups_by_bin = {
            label: tuple(sorted(keys, key=str)) for label, keys in groups_by_bin.items()
        }

    def _ordered_groups(self) -> list[Hashable]:
        rng = random.Random(self.seed + self.epoch)
        exposures: list[Hashable] = []
        for label in sorted(self.target_counts):
            target = self.target_counts[label]
            source = list(self.groups_by_bin.get(label, ()))
            if target <= 0:
                continue
            rng.shuffle(source)
            for index in range(target):
                if index and index % len(source) == 0:
                    rng.shuffle(source)
                exposures.append(source[index % len(source)])
        if self.shuffle:
            rng.shuffle(exposures)
        if self.drop_rank_tail and self.num_replicas > 1:
            usable = len(exposures) - (len(exposures) % self.num_replicas)
            exposures = exposures[:usable]
        return exposures

    def audit(self) -> dict[str, object]:
        global_exposures = self._ordered_groups()
        local = self.groups_for_rank()
        emitted = list(iter(self))
        exposure_counts = Counter(self.difficulty_by_group[key] for key in global_exposures)
        return {
            "sampler": self.__class__.__name__,
            "n_unique_global_groups": len(self.group_keys),
            "n_global_group_exposures": len(global_exposures),
            "n_local_group_exposures": len(local),
            "n_local_batches": len(emitted),
            "groups_split_across_batches": self._count_split_groups(emitted),
            "target_probabilities": self.target_probabilities,
            "target_counts_before_ddp_tail": self.target_counts,
            "actual_counts_after_ddp_tail": dict(exposure_counts),
            "duplicate_global_group_exposures": len(global_exposures)
            - len(set(global_exposures)),
            "drop_rank_tail": self.drop_rank_tail,
            "drop_batch_tail": self.drop_batch_tail,
            "rank": self.rank,
            "num_replicas": self.num_replicas,
        }


class DistributionAwareGroupBatchSampler(DistributionAwareDistributedGroupBatchSampler):
    """Single-process convenience wrapper for distribution-aware training."""

    def __init__(
        self,
        group_keys: Sequence[Hashable],
        *,
        difficulty_by_group: Mapping[Hashable, str],
        target_probabilities: Mapping[str, float],
        samples_per_epoch: int | None = None,
        groups_per_batch: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_batch_tail: bool = False,
    ) -> None:
        super().__init__(
            group_keys,
            difficulty_by_group=difficulty_by_group,
            target_probabilities=target_probabilities,
            samples_per_epoch=samples_per_epoch,
            groups_per_batch=groups_per_batch,
            shuffle=shuffle,
            seed=seed,
            num_replicas=1,
            rank=0,
            drop_rank_tail=False,
            drop_batch_tail=drop_batch_tail,
        )
