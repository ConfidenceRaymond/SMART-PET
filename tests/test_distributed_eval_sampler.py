from dataclasses import dataclass

from smartpet.training.distributed import DistributedEvalSampler


@dataclass
class RuntimeStub:
    rank: int
    world_size: int


def test_eval_sampler_has_no_padding_or_overlap():
    dataset = list(range(10))
    partitions = [list(DistributedEvalSampler(dataset, RuntimeStub(rank, 3))) for rank in range(3)]
    flattened = [item for part in partitions for item in part]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))
