#!/usr/bin/env python3
"""ICD10-ICD11 RAG retrieval followed by contrastive reranking.

The script intentionally uses plain files instead of a vector database so the
pre-experiment is reproducible in one command. "RAG" here means dense candidate
retrieval. The retrieved Top-K list is cached and becomes the closed candidate
set for contrastive reranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    description: str = ""
    hierarchy_path: str = ""


def entity_text(entity: Entity, text_mode: str) -> str:
    """Build the encoder text from the fields available in Integration2023."""
    if text_mode == "name":
        return entity.name
    if text_mode == "path":
        return entity.hierarchy_path or entity.name
    parts = [entity.name]
    if entity.hierarchy_path:
        parts.append(f"Hierarchy: {entity.hierarchy_path}")
    if entity.description:
        parts.append(f"Description: {entity.description}")
    return ". ".join(parts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def fallback_name(entity_id: str) -> str:
    value = entity_id.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return value.replace("_", " ").replace("-", " ").strip()


def read_entities(path: str) -> Dict[str, Entity]:
    entities: Dict[str, Entity] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                entity_id = str(
                    row.get("id", row.get("entity_id", row.get("code", "")))
                )
                name = str(
                    row.get("name", row.get("term", row.get("label", "")))
                )
                description = str(
                    row.get("description", row.get("definition", ""))
                )
            else:
                columns = line.split("\t")
                if len(columns) == 1:
                    columns = line.split(None, 1)
                entity_id = columns[0]
                name = columns[1] if len(columns) > 1 else ""
                description = columns[2] if len(columns) > 2 else ""
            if not entity_id:
                raise ValueError(f"{path}:{line_number}: missing entity ID")
            name = name.strip()
            if name.startswith(("http://", "https://")):
                name = fallback_name(name)
            name = name or fallback_name(entity_id)
            entities[entity_id] = Entity(entity_id, name, description.strip())
    if not entities:
        raise ValueError(f"No entities found in {path}")
    return entities


def read_paths(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    paths: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            columns = line.split("\t", 1)
            if not columns[0]:
                raise ValueError(f"{path}:{line_number}: missing entity ID")
            hierarchy = columns[1].strip(" \t,") if len(columns) > 1 else ""
            paths[columns[0]] = hierarchy
    return paths


def attach_paths(
    entities: Dict[str, Entity], paths: Dict[str, str]
) -> Dict[str, Entity]:
    return {
        entity_id: Entity(
            entity.entity_id,
            entity.name,
            entity.description,
            paths.get(entity_id, ""),
        )
        for entity_id, entity in entities.items()
    }


def read_pairs(
    path: str, source: Dict[str, Entity], target: Dict[str, Entity]
) -> List[Tuple[str, str]]:
    pairs = []
    missing_source = set()
    missing_target = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                left = str(
                    row.get("source_id", row.get("left_id", row.get("source", "")))
                )
                right = str(
                    row.get("target_id", row.get("right_id", row.get("target", "")))
                )
            else:
                columns = line.replace(",", "\t").split()
                if len(columns) < 2:
                    raise ValueError(f"{path}:{line_number}: expected two IDs")
                left, right = columns[:2]
            if left not in source:
                missing_source.add(left)
            if right not in target:
                missing_target.add(right)
            if left in source and right in target:
                pairs.append((left, right))
    if missing_source or missing_target:
        print(
            f"Warning: skipped pairs with missing IDs: "
            f"source={len(missing_source)}, target={len(missing_target)}"
        )
    if not pairs:
        raise ValueError(f"No valid pairs found in {path}")
    return pairs


def discover_dataset_files(dataset_dir: str):
    root = Path(dataset_dir)
    choices = {
        "source_entities": [
            "source_entities.tsv",
            "entities_1.tsv",
            "ent_ids_1",
            "ent_ids_1.txt",
        ],
        "target_entities": [
            "target_entities.tsv",
            "entities_2.tsv",
            "ent_ids_2",
            "ent_ids_2.txt",
        ],
        "train_pairs": [
            "sup_pairs",
            "train_pairs.tsv",
            "sup_pairs.tsv",
            "sup_pairs.txt",
        ],
        "test_pairs": [
            "ref_pairs",
            "test_pairs.tsv",
            "ref_pairs.tsv",
            "ref_pairs.txt",
        ],
        "source_paths": ["paths_1", "paths_1.tsv", "paths_1.txt"],
        "target_paths": ["paths_2", "paths_2.tsv", "paths_2.txt"],
    }
    resolved = {}
    for key, names in choices.items():
        resolved[key] = next((root / name for name in names if (root / name).exists()), None)
    required = {"source_entities", "target_entities", "train_pairs", "test_pairs"}
    missing = [key for key in required if resolved[key] is None]
    if missing:
        raise FileNotFoundError(
            f"Could not discover {', '.join(missing)} under {root}. "
            "Pass the corresponding explicit path arguments."
        )
    return {
        key: str(value) if value is not None else None
        for key, value in resolved.items()
    }


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, local_files_only: bool):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1)
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(pooled, dim=-1)


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        chunks.append(
            model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            ).cpu()
        )
    return torch.cat(chunks, dim=0)


def retrieve_topk(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    top_k: int,
    query_chunk_size: int = 256,
):
    all_scores, all_indices = [], []
    target_t = target_embeddings.T
    for start in range(0, len(query_embeddings), query_chunk_size):
        scores = query_embeddings[start : start + query_chunk_size] @ target_t
        values, indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)
        all_scores.append(values)
        all_indices.append(indices)
    return torch.cat(all_scores), torch.cat(all_indices)


def ranking_metrics(
    rankings: Dict[str, List[str]], pairs: Sequence[Tuple[str, str]]
) -> Dict[str, float]:
    hit1 = hit10 = recall20 = 0
    reciprocal_ranks = []
    for source_id, target_id in pairs:
        candidates = rankings.get(source_id, [])
        rank = candidates.index(target_id) + 1 if target_id in candidates else None
        hit1 += int(rank == 1)
        hit10 += int(rank is not None and rank <= 10)
        recall20 += int(rank is not None and rank <= 20)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    total = len(pairs)
    return {
        "recall_at_20": recall20 / total,
        "hits_at_1": hit1 / total,
        "hits_at_10": hit10 / total,
        "mrr": float(np.mean(reciprocal_ranks)),
        "evaluated_pairs": total,
    }


def exact_rankings(
    source: Dict[str, Entity], target: Dict[str, Entity], source_ids: Iterable[str]
):
    lookup: Dict[str, List[str]] = {}
    for entity_id, entity in target.items():
        lookup.setdefault(normalize_name(entity.name), []).append(entity_id)
    return {
        source_id: lookup.get(normalize_name(source[source_id].name), [])
        for source_id in source_ids
    }


def write_candidates(
    path: Path,
    source: Dict[str, Entity],
    target: Dict[str, Entity],
    source_ids: Sequence[str],
    indices: torch.Tensor,
    scores: torch.Tensor,
    target_ids: Sequence[str],
):
    with path.open("w", encoding="utf-8") as handle:
        for row_index, source_id in enumerate(source_ids):
            candidates = [
                {
                    "target_id": target_ids[int(index)],
                    "target_name": target[target_ids[int(index)]].name,
                    "score": float(score),
                }
                for index, score in zip(indices[row_index], scores[row_index])
            ]
            handle.write(
                json.dumps(
                    {
                        "source_id": source_id,
                        "source_name": source[source_id].name,
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        source: Dict[str, Entity],
        target: Dict[str, Entity],
        hard_negatives: Dict[str, List[str]],
        text_mode: str,
    ):
        self.pairs = list(pairs)
        self.source = source
        self.target = target
        self.hard_negatives = hard_negatives
        self.text_mode = text_mode

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source_id, positive_id = self.pairs[index]
        negatives = [
            entity_id
            for entity_id in self.hard_negatives.get(source_id, [])
            if entity_id != positive_id
        ]
        if negatives:
            negative_id = random.choice(negatives)
        else:
            fallback_negatives = [
                entity_id for entity_id in self.target if entity_id != positive_id
            ]
            negative_id = random.choice(fallback_negatives or [positive_id])
        return (
            entity_text(self.source[source_id], self.text_mode),
            entity_text(self.target[positive_id], self.text_mode),
            entity_text(self.target[negative_id], self.text_mode),
        )


class PairCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, batch):
        source, positive, negative = zip(*batch)
        return {
            "source": self._tokenize(source),
            "positive": self._tokenize(positive),
            "negative": self._tokenize(negative),
        }


def train_contrastive(
    model,
    tokenizer,
    dataset,
    device,
    epochs,
    batch_size,
    learning_rate,
    temperature,
    max_length,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PairCollator(tokenizer, max_length),
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"Contrastive {epoch + 1}/{epochs}"):
            optimizer.zero_grad()

            def encode(part):
                return model(
                    batch[part]["input_ids"].to(device),
                    batch[part]["attention_mask"].to(device),
                )

            query = encode("source")
            positive = encode("positive")
            negative = encode("negative")
            candidates = torch.cat([positive, negative], dim=0)
            logits = query @ candidates.T / temperature
            labels = torch.arange(len(query), device=device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} contrastive_loss={np.mean(losses):.4f}")


@torch.no_grad()
def rerank_candidates(
    model,
    tokenizer,
    source,
    target,
    source_ids,
    initial_rankings,
    device,
    batch_size,
    max_length,
    text_mode,
):
    unique_targets = list(
        dict.fromkeys(
            target_id
            for source_id in source_ids
            for target_id in initial_rankings[source_id]
        )
    )
    query_embeddings = encode_texts(
        model,
        tokenizer,
        [entity_text(source[source_id], text_mode) for source_id in source_ids],
        device,
        batch_size,
        max_length,
    )
    target_embeddings = encode_texts(
        model,
        tokenizer,
        [entity_text(target[target_id], text_mode) for target_id in unique_targets],
        device,
        batch_size,
        max_length,
    )
    target_index = {
        target_id: target_embeddings[index]
        for index, target_id in enumerate(unique_targets)
    }
    reranked = {}
    for index, source_id in enumerate(source_ids):
        candidates = initial_rankings[source_id]
        scores = [
            float(query_embeddings[index] @ target_index[target_id])
            for target_id in candidates
        ]
        reranked[source_id] = [
            target_id
            for target_id, _score in sorted(
                zip(candidates, scores), key=lambda item: item[1], reverse=True
            )
        ]
    return reranked


def llm_filter(
    model_name: str,
    source: Dict[str, Entity],
    target: Dict[str, Entity],
    rankings: Dict[str, List[str]],
    device: str,
    keep: int,
    local_files_only: bool,
    limit: Optional[int],
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=local_files_only,
    )
    filtered = {}
    for index, (source_id, candidates) in enumerate(
        tqdm(rankings.items(), desc="LLM filtering")
    ):
        if limit is not None and index >= limit:
            filtered[source_id] = candidates
            continue
        candidate_lines = "\n".join(
            f"{candidate_id}\t{target[candidate_id].name}"
            for candidate_id in candidates
        )
        prompt = (
            "Select the most likely equivalent target terminology concepts for "
            "the source term. Return only target IDs separated by commas, in "
            f"best-first order, at most {keep} IDs.\n\n"
            f"Source: {source[source_id].name}\nCandidates:\n{candidate_lines}\nAnswer:"
        )
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**encoded, max_new_tokens=80, do_sample=False)
        answer = tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        selected = [
            candidate_id
            for candidate_id in candidates
            if candidate_id in [part.strip() for part in answer.split(",")]
        ][:keep]
        filtered[source_id] = selected or candidates[:keep]
    return filtered


def save_results(output_dir: Path, results: Dict[str, dict], config: dict):
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "results": results}, handle, indent=2)
    with (output_dir / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "method",
            "recall_at_20",
            "hits_at_1",
            "hits_at_10",
            "mrr",
            "evaluated_pairs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, values in results.items():
            writer.writerow({"method": method, **values})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "sample_data"))
    parser.add_argument("--source-entities")
    parser.add_argument("--target-entities")
    parser.add_argument("--train-pairs")
    parser.add_argument("--test-pairs")
    parser.add_argument("--source-paths")
    parser.add_argument("--target-paths")
    parser.add_argument(
        "--text-mode",
        choices=("name", "name_path", "path"),
        default="name_path",
        help="Encoder input. Integration2023 has names and hierarchy paths but no definitions.",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument("--output-dir", default="runs/el_icd10_icd11")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-keep", type=int, default=5)
    parser.add_argument("--llm-limit", type=int)
    parser.add_argument("--validate-data-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset_dir:
        discovered = discover_dataset_files(args.dataset_dir)
        args.source_entities = args.source_entities or discovered["source_entities"]
        args.target_entities = args.target_entities or discovered["target_entities"]
        args.train_pairs = args.train_pairs or discovered["train_pairs"]
        args.test_pairs = args.test_pairs or discovered["test_pairs"]
        args.source_paths = args.source_paths or discovered["source_paths"]
        args.target_paths = args.target_paths or discovered["target_paths"]
    required = [
        args.source_entities,
        args.target_entities,
        args.train_pairs,
        args.test_pairs,
    ]
    if not all(required):
        raise ValueError(
            "Provide --dataset-dir or all four explicit entity/pair file paths."
        )
    if args.smoke_test:
        args.limit = args.limit or 32
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.encode_batch_size = min(args.encode_batch_size, 16)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = attach_paths(
        read_entities(args.source_entities), read_paths(args.source_paths)
    )
    target = attach_paths(
        read_entities(args.target_entities), read_paths(args.target_paths)
    )
    train_pairs = read_pairs(args.train_pairs, source, target)
    test_pairs = read_pairs(args.test_pairs, source, target)
    if args.limit:
        train_pairs = train_pairs[: args.limit]
        test_pairs = test_pairs[: args.limit]
        keep_source = {left for left, _right in train_pairs + test_pairs}
        source = {key: value for key, value in source.items() if key in keep_source}
        gold_targets = {right for _left, right in train_pairs + test_pairs}
        extra_targets = [
            key for key in target if key not in gold_targets
        ][: max(args.limit * 4, args.top_k)]
        keep_target = gold_targets | set(extra_targets)
        target = {key: value for key, value in target.items() if key in keep_target}

    audit = {
        "source_entities": len(source),
        "target_entities": len(target),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "source_entities_with_paths": sum(
            bool(entity.hierarchy_path) for entity in source.values()
        ),
        "target_entities_with_paths": sum(
            bool(entity.hierarchy_path) for entity in target.values()
        ),
        "empty_source_names": sum(not entity.name for entity in source.values()),
        "empty_target_names": sum(not entity.name for entity in target.values()),
        "source_name_duplicates": len(source)
        - len({normalize_name(entity.name) for entity in source.values()}),
        "target_name_duplicates": len(target)
        - len({normalize_name(entity.name) for entity in target.values()}),
        "train_test_pair_overlap": len(set(train_pairs) & set(test_pairs)),
        "train_source_overlap_with_test": len(
            {left for left, _right in train_pairs}
            & {left for left, _right in test_pairs}
        ),
        "text_mode": args.text_mode,
    }
    with (output_dir / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))
    if args.validate_data_only:
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    model = TextEncoder(args.model_name, args.local_files_only).to(device)
    target_ids = list(target)
    test_source_ids = list(dict.fromkeys(left for left, _right in test_pairs))
    all_retrieval_source_ids = list(
        dict.fromkeys([left for left, _right in train_pairs + test_pairs])
    )

    target_embeddings = encode_texts(
        model,
        tokenizer,
        [entity_text(target[target_id], args.text_mode) for target_id in target_ids],
        device,
        args.encode_batch_size,
        args.max_length,
    )
    query_embeddings = encode_texts(
        model,
        tokenizer,
        [
            entity_text(source[source_id], args.text_mode)
            for source_id in all_retrieval_source_ids
        ],
        device,
        args.encode_batch_size,
        args.max_length,
    )
    scores, indices = retrieve_topk(
        query_embeddings, target_embeddings, args.top_k
    )
    rag_rankings = {
        source_id: [target_ids[int(index)] for index in indices[row]]
        for row, source_id in enumerate(all_retrieval_source_ids)
    }
    write_candidates(
        output_dir / "rag_top20.jsonl",
        source,
        target,
        all_retrieval_source_ids,
        indices,
        scores,
        target_ids,
    )

    results = {
        "exact_name": ranking_metrics(
            exact_rankings(source, target, test_source_ids), test_pairs
        ),
        "rag_dense_retrieval": ranking_metrics(rag_rankings, test_pairs),
    }
    train_dataset = PairDataset(
        train_pairs, source, target, rag_rankings, args.text_mode
    )
    train_contrastive(
        model,
        tokenizer,
        train_dataset,
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.temperature,
        args.max_length,
    )
    torch.save(model.state_dict(), output_dir / "contrastive_encoder.pt")
    reranked = rerank_candidates(
        model,
        tokenizer,
        source,
        target,
        test_source_ids,
        rag_rankings,
        device,
        args.encode_batch_size,
        args.max_length,
        args.text_mode,
    )
    results["rag_contrastive_reranking"] = ranking_metrics(reranked, test_pairs)
    with (output_dir / "reranked_top20.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for source_id in test_source_ids:
            handle.write(
                json.dumps(
                    {
                        "source_id": source_id,
                        "source_name": source[source_id].name,
                        "target_ids": reranked[source_id],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if args.llm_model:
        llm_candidates = llm_filter(
            args.llm_model,
            source,
            target,
            {key: rag_rankings[key] for key in test_source_ids},
            args.device,
            args.llm_keep,
            args.local_files_only,
            args.llm_limit,
        )
        llm_reranked = rerank_candidates(
            model,
            tokenizer,
            source,
            target,
            test_source_ids,
            llm_candidates,
            device,
            args.encode_batch_size,
            args.max_length,
            args.text_mode,
        )
        llm_metrics = ranking_metrics(llm_reranked, test_pairs)
        # Recall@20 remains the original RAG candidate recall by definition.
        llm_metrics["recall_at_20"] = results["rag_dense_retrieval"]["recall_at_20"]
        results["rag_llm_contrastive_reranking"] = llm_metrics

    save_results(output_dir, results, vars(args))
    print(json.dumps(results, indent=2))
    print(f"Wrote EL outputs to {output_dir}")


if __name__ == "__main__":
    main()
