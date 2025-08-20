import json
from pathlib import Path
import gzip
from typing import Iterator
import numpy as np
from openai import OpenAI
from usearch.index import Index
from .db import AISafetyGraph


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# TODO: make it interactive, tune merging threshold
class Merger:
    def __init__(self):
        self.g = AISafetyGraph()
        self.client = OpenAI()
        self.nodes = self.g.get_nodes()
        # TODO: make this configurable
        self.input_jsonl = "batchinput.jsonl"
        self.create_out = "batch_create.json"
        self.output_jsonl = "batch_out.jsonl"
        self.batch = None

    def create_batch(self):
        with open(self.input_jsonl, "w", encoding="utf-8") as f:
            for i, node in enumerate(self.nodes):
                line = {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "text-embedding-3-small",
                        "input": node["text"],
                        "encoding_format": "float",
                    },
                }
                f.write(json.dumps(line) + "\n")
        uploaded = self.client.files.create(
            file=open(self.input_jsonl, "rb"), purpose="batch"
        )
        self.batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )
        with open(self.create_out, "w", encoding="utf-8") as f:
            json.dump(self.batch.model_dump(), f)

    def retrieve_output(self):
        self.batch = self.client.batches.retrieve(self.batch.id)
        file_response = self.client.files.content(self.batch.output_file_id)
        with open(self.output_jsonl, "w") as f:
            f.write(file_response.text)

    def load_embeddings(self):
        embs = []
        for item in iter_jsonl(self.output_jsonl):
            embs.append(item["response"]["body"]["data"][0]["embedding"])
        return np.asarray(embs, dtype=np.float32)

    def top_duplicate_pairs_with_texts(
        self, embs, nodes, K=30, top_n=30, metric="cos", dtype="f32", exact=False
    ):
        n, d = embs.shape
        keys = np.arange(n, dtype=np.int64)
        index = Index(ndim=d, metric=metric, dtype=dtype)
        index.add(keys, embs)
        batch = index.search(embs, count=K + 1, exact=exact)
        best_pair_dist = {}
        for i in range(len(batch)):
            k = int(keys[i])
            m = batch[i]
            for nk, dist in zip(m.keys, m.distances):
                nk = int(nk)
                if nk == k:
                    continue
                a, b = (k, nk) if k < nk else (nk, k)
                prev = best_pair_dist.get((a, b))
                if prev is None or dist < prev:
                    best_pair_dist[(a, b)] = float(dist)
        top = sorted(best_pair_dist.items(), key=lambda kv: kv[1])[:top_n]
        results = []
        for (a, b), d in top:
            results.append(
                {
                    "id_a": a,
                    "id_b": b,
                    "node_a": nodes[a],
                    "node_b": nodes[b],
                    "distance": d,
                    "similarity": 1.0 - d,
                }
            )
        return results

    def merge(self, pairs):
        for r in pairs:
            self.g.merge_nodes(r["node_a"]["id"], r["node_b"]["id"])

    def run(self):
        self.create_batch()
        self.retrieve_output()
        embs = self.load_embeddings()
        pairs = self.top_duplicate_pairs_with_texts(embs, self.nodes, K=40, top_n=30)
        self.merge(pairs)


if __name__ == "__main__":
    Merger().run()
