from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STATIC_VERSION = "entity-word-block-rgcn-v1"


@dataclass
class StaticGraph:
    triples: np.ndarray
    num_entities: int
    num_words: int
    num_relations: int
    metadata: dict


def load_static_graph(directory, dataset_metadata):
    path = Path(directory) / "e-w-graph.txt"
    triples = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if triples.shape[1] != 3 or not len(triples) or triples.min() < 0:
        raise ValueError("Static graph must contain nonnegative entity/relation/word triples")
    if triples[:, 0].max() >= dataset_metadata["source_stat_entities"]:
        raise ValueError("Static graph entity ID exceeds stat.txt")
    for column in (1, 2):
        ids = np.unique(triples[:, column])
        if not np.array_equal(ids, np.arange(len(ids))):
            raise ValueError("Static relation and word IDs must be contiguous from zero")
    num_relations = int(triples[:, 1].max()) + 1
    num_words = int(triples[:, 2].max()) + 1
    source_edges = len(triples)
    metadata = {
        "version": STATIC_VERSION,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "num_words": num_words,
        "num_relations": num_relations,
        "source_edges": source_edges,
        "retained_edges": len(triples),
        "normalization": "mean by total incoming degree, functional RReLU, then entity L2",
        "activation": "F.rrelu default training=False (negative slope 11/48)",
        "self_loop": False,
        "static_loss": False,
    }
    return StaticGraph(
        triples, dataset_metadata["num_entities"], num_words, num_relations, metadata
    )


class StaticInit(nn.Module):
    def __init__(self, graph, dimension):
        super().__init__()

        self.num_blocks = math.gcd(dimension, 100)
        self.block_size = dimension // self.num_blocks
        self.num_relations = graph.num_relations
        self.words = nn.Parameter(torch.empty(graph.num_words, dimension))
        self.weight = nn.Parameter(
            torch.empty(2 * graph.num_relations, self.num_blocks * self.block_size**2)
        )
        nn.init.xavier_normal_(self.words)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        self.register_buffer("entity_ids", torch.as_tensor(graph.triples[:, 0].copy()))
        self.register_buffer(
            "relation_ids", torch.as_tensor(graph.triples[:, 1].copy()) + graph.num_relations
        )
        self.register_buffer("word_ids", torch.as_tensor(graph.triples[:, 2].copy()))
        degree = torch.bincount(self.entity_ids, minlength=graph.num_entities)
        self.register_buffer("degree", degree.clamp_min(1)[:, None])

    def forward(self, entity_embeddings):
        vectors = self.words[self.word_ids].reshape(-1, 1, self.block_size)
        weights = self.weight[self.relation_ids].reshape(-1, self.block_size, self.block_size)
        messages = torch.bmm(vectors, weights).reshape(len(self.entity_ids), self.words.shape[1])
        result = entity_embeddings.new_zeros(entity_embeddings.shape).index_add(
            0, self.entity_ids, messages
        )
        result = F.rrelu(result / self.degree)

        return F.normalize(result, p=2, dim=-1) + entity_embeddings * 0
