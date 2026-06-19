#!/usr/bin/env python3
"""Two-stage BC5CDR span pruning pre-experiment.

Stage 1 learns span/type compatibility from DictMatching train labels.
Stage 2 is parameter-free: generate boundary variants, score them with the
warm-up model, keep the best variant in each group, threshold, and apply NMS.

Gold development labels are used only for checkpoint/threshold selection.
Gold test labels are evaluated once after all choices are fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


TYPE2ID = {"Chemical": 0, "Disease": 1}
ID2TYPE = {value: key for key, value in TYPE2ID.items()}
WORD_RE = re.compile(r"\S+")
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    ent_type: str


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    ent_type: str
    label: int
    group_id: str
    variant: str


@dataclass
class Example:
    example_id: str
    text: str
    words: List[Tuple[int, int]]
    noisy_spans: List[Span]
    gold_spans: List[Span]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bio_to_spans(
    tokens: Sequence[str], labels: Sequence[str]
) -> Tuple[str, List[Tuple[int, int]], List[Span]]:
    pieces: List[str] = []
    word_offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if pieces:
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        word_offsets.append((start, cursor))
    text = " ".join(pieces)

    spans: List[Span] = []
    current_type: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    for index, raw_label in enumerate(list(labels) + ["O"]):
        prefix, ent_type = ("O", None)
        if raw_label != "O" and "-" in raw_label:
            prefix, ent_type = raw_label.split("-", 1)
        continuation = (
            index < len(labels)
            and prefix == "I"
            and ent_type == current_type
            and current_start is not None
        )
        if current_start is not None and not continuation:
            spans.append(Span(current_start, int(current_end), str(current_type)))
            current_type = None
            current_start = None
            current_end = None
        if index >= len(labels) or raw_label == "O":
            continue
        if prefix == "B" or not continuation:
            current_type = ent_type
            current_start = word_offsets[index][0]
        current_end = word_offsets[index][1]
    return text, word_offsets, [span for span in spans if span.ent_type in TYPE2ID]


def read_remote_conll(path: str, limit: Optional[int] = None) -> List[Example]:
    examples: List[Example] = []
    tokens: List[str] = []
    noisy_labels: List[str] = []
    gold_labels: List[str] = []

    def flush() -> None:
        if not tokens:
            return
        text, words, noisy = bio_to_spans(tokens, noisy_labels)
        _, _, gold = bio_to_spans(tokens, gold_labels)
        examples.append(
            Example(
                example_id=f"{Path(path).stem}:{len(examples)}",
                text=text,
                words=words,
                noisy_spans=noisy,
                gold_spans=gold,
            )
        )
        tokens.clear()
        noisy_labels.clear()
        gold_labels.clear()

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                flush()
                if limit is not None and len(examples) >= limit:
                    break
                continue
            columns = line.split("\t")
            if len(columns) < 3:
                raise ValueError(f"{path}:{line_number}: expected token/noisy/gold columns")
            tokens.append(columns[0])
            noisy_labels.append(columns[1])
            gold_labels.append(columns[2])
    if limit is None or len(examples) < limit:
        flush()
    return examples[:limit]


def read_gold_json(path: str, limit: Optional[int] = None) -> List[Example]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    examples: List[Example] = []
    for row in rows[:limit]:
        text = row["text"]
        spans = [
            Span(int(start), int(end), ent_type)
            for _mention, start, end, ent_type, *_rest in row.get("entities", [])
            if ent_type in TYPE2ID and 0 <= int(start) < int(end) <= len(text)
        ]
        examples.append(
            Example(
                example_id=str(row.get("sentence_id", len(examples))),
                text=text,
                words=[match.span() for match in WORD_RE.finditer(text)],
                noisy_spans=[],
                gold_spans=spans,
            )
        )
    return examples


def find_word_span(
    words: Sequence[Tuple[int, int]], start: int, end: int
) -> Optional[Tuple[int, int]]:
    indices = [
        index
        for index, (word_start, word_end) in enumerate(words)
        if word_end > start and word_start < end
    ]
    if not indices:
        return None
    return indices[0], indices[-1]


def boundary_variants(
    example: Example, span: Span, group_id: str, labeled: bool
) -> List[Candidate]:
    word_span = find_word_span(example.words, span.start, span.end)
    raw: List[Tuple[str, int, int]] = [("original", span.start, span.end)]
    if word_span is not None:
        left, right = word_span
        if left > 0:
            raw.append(("expand_left", example.words[left - 1][0], span.end))
        if right + 1 < len(example.words):
            raw.append(("expand_right", span.start, example.words[right + 1][1]))
        if left > 0 and right + 1 < len(example.words):
            raw.append(
                (
                    "expand_both",
                    example.words[left - 1][0],
                    example.words[right + 1][1],
                )
            )
        if left < right:
            raw.append(("shrink_left", example.words[left + 1][0], span.end))
            raw.append(("shrink_right", span.start, example.words[right - 1][1]))

    gold = set(example.gold_spans)
    candidates: List[Candidate] = []
    seen = set()
    for variant, start, end in raw:
        key = (start, end, span.ent_type)
        if start >= end or key in seen:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                start=start,
                end=end,
                ent_type=span.ent_type,
                label=int(Span(start, end, span.ent_type) in gold) if labeled else int(variant == "original"),
                group_id=group_id,
                variant=variant,
            )
        )
    return candidates


def warmup_candidates(example: Example) -> List[Candidate]:
    candidates: List[Candidate] = []
    for index, span in enumerate(example.noisy_spans):
        group_id = f"{example.example_id}:{index}"
        variants = boundary_variants(example, span, group_id, labeled=False)
        candidates.extend(variants)
        other_type = "Disease" if span.ent_type == "Chemical" else "Chemical"
        candidates.append(
            Candidate(
                span.start,
                span.end,
                other_type,
                0,
                group_id,
                "wrong_type",
            )
        )
    return candidates


def exhaustive_candidates(example: Example, max_words: int) -> List[Candidate]:
    candidates: List[Candidate] = []
    for left in range(len(example.words)):
        for right in range(left, min(len(example.words), left + max_words)):
            start, end = example.words[left][0], example.words[right][1]
            for ent_type in TYPE2ID:
                candidates.append(
                    Candidate(
                        start,
                        end,
                        ent_type,
                        0,
                        f"{example.example_id}:{left}:{right}:{ent_type}",
                        "exhaustive",
                    )
                )
    return candidates


class CandidateDataset(Dataset):
    def __init__(self, examples: Sequence[Example], mode: str, max_words: int = 10):
        self.examples = list(examples)
        self.mode = mode
        self.max_words = max_words

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[Example, List[Candidate]]:
        example = self.examples[index]
        if self.mode == "warmup":
            candidates = warmup_candidates(example)
        elif self.mode == "exhaustive":
            candidates = exhaustive_candidates(example, self.max_words)
        elif self.mode == "original":
            candidates = [
                Candidate(
                    span.start,
                    span.end,
                    span.ent_type,
                    int(span in set(example.gold_spans)),
                    f"{example.example_id}:{i}",
                    "original",
                )
                for i, span in enumerate(example.noisy_spans)
            ]
        elif self.mode == "sliding":
            candidates = [
                candidate
                for i, span in enumerate(example.noisy_spans)
                for candidate in boundary_variants(
                    example, span, f"{example.example_id}:{i}", labeled=True
                )
            ]
        else:
            raise ValueError(f"Unknown candidate mode: {self.mode}")
        return example, candidates


class SpanCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        examples = [item[0] for item in batch]
        candidate_lists = [item[1] for item in batch]
        encoding = self.tokenizer(
            [example.text for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoding.pop("offset_mapping").tolist()
        rows, labels, group_ids, variants, char_spans = [], [], [], [], []
        for batch_index, candidates in enumerate(candidate_lists):
            for candidate in candidates:
                token_ids = [
                    token_index
                    for token_index, (start, end) in enumerate(offsets[batch_index])
                    if not (start == end == 0)
                    and end > candidate.start
                    and start < candidate.end
                ]
                if not token_ids:
                    continue
                rows.append(
                    [
                        batch_index,
                        token_ids[0],
                        token_ids[-1],
                        TYPE2ID[candidate.ent_type],
                    ]
                )
                labels.append(candidate.label)
                group_ids.append(candidate.group_id)
                variants.append(candidate.variant)
                char_spans.append(
                    Span(candidate.start, candidate.end, candidate.ent_type)
                )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "candidates": torch.tensor(rows, dtype=torch.long)
            if rows
            else torch.empty((0, 4), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.bool),
            "group_ids": group_ids,
            "variants": variants,
            "char_spans": char_spans,
            "example_ids": [example.example_id for example in examples],
        }


class SpanTypeModel(nn.Module):
    def __init__(self, model_name: str, projection_dim: int, local_files_only: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        hidden = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, projection_dim),
        )
        self.type_embeddings = nn.Embedding(len(TYPE2ID), projection_dim)

    def forward(self, input_ids, attention_mask, candidates):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = output.last_hidden_state
        if candidates.numel() == 0:
            return sequence.new_empty((0, len(TYPE2ID)))
        batch_ids, starts, ends, _type_ids = candidates.T
        prefix = torch.cat(
            [torch.zeros_like(sequence[:, :1]), sequence.cumsum(dim=1)], dim=1
        )
        lengths = (ends - starts + 1).unsqueeze(-1)
        means = (prefix[batch_ids, ends + 1] - prefix[batch_ids, starts]) / lengths
        features = torch.cat(
            [
                sequence[batch_ids, starts],
                sequence[batch_ids, ends],
                means,
                sequence[batch_ids, 0],
            ],
            dim=-1,
        )
        span_z = F.normalize(self.projector(features), dim=-1)
        type_z = F.normalize(self.type_embeddings.weight, dim=-1)
        return span_z @ type_z.T


def warmup_loss(
    similarities: torch.Tensor,
    candidates: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    negative_margin: float,
) -> torch.Tensor:
    if similarities.numel() == 0:
        return similarities.sum()
    type_ids = candidates[:, 3]
    losses = []
    if labels.any():
        losses.append(
            F.cross_entropy(similarities[labels] / temperature, type_ids[labels])
        )
    if (~labels).any():
        assigned = similarities[~labels, type_ids[~labels]]
        losses.append(F.softplus((assigned - negative_margin) / temperature).mean())
    return torch.stack(losses).mean()


def span_iou(left: Span, right: Span) -> float:
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start))
    if intersection == 0:
        return 0.0
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union


def nms(scored: Sequence[Tuple[Span, float]], overlap: float) -> List[Tuple[Span, float]]:
    kept: List[Tuple[Span, float]] = []
    for span, score in sorted(scored, key=lambda item: item[1], reverse=True):
        if all(
            span.ent_type != other.ent_type or span_iou(span, other) <= overlap
            for other, _ in kept
        ):
            kept.append((span, score))
    return kept


def metrics(predictions: Dict[str, set], examples: Sequence[Example]) -> Dict[str, float]:
    tp = fp = fn = 0
    for example in examples:
        predicted = predictions.get(example.example_id, set())
        gold = set(example.gold_spans)
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def collect_scores(model, loader, device) -> Dict[str, List[Tuple[Span, float, str, str]]]:
    model.eval()
    output: Dict[str, List[Tuple[Span, float, str, str]]] = {}
    for batch in tqdm(loader, desc="Scoring", leave=False):
        candidates = batch["candidates"].to(device)
        similarities = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            candidates,
        )
        if similarities.numel() == 0:
            continue
        assigned = similarities[
            torch.arange(len(candidates), device=device), candidates[:, 3]
        ].cpu()
        for span, score, group_id, variant in zip(
            batch["char_spans"], assigned.tolist(), batch["group_ids"], batch["variants"]
        ):
            example_id = group_id.rsplit(":", 1)[0]
            if variant == "exhaustive":
                example_id = group_id.split(":", 1)[0]
            output.setdefault(example_id, []).append(
                (span, float(score), group_id, variant)
            )
    return output


def predictions_from_scores(
    scored: Dict[str, List[Tuple[Span, float, str, str]]],
    threshold: float,
    overlap: float,
    group_best: bool,
) -> Dict[str, set]:
    predictions: Dict[str, set] = {}
    for example_id, rows in scored.items():
        selected = rows
        if group_best:
            best: Dict[str, Tuple[Span, float, str, str]] = {}
            for row in rows:
                if row[2] not in best or row[1] > best[row[2]][1]:
                    best[row[2]] = row
            selected = list(best.values())
        above = [(span, score) for span, score, _group, _variant in selected if score >= threshold]
        predictions[example_id] = {span for span, _score in nms(above, overlap)}
    return predictions


def tune_postprocessing(
    scored,
    examples,
    thresholds: Sequence[float],
    overlaps: Sequence[float],
    group_best: bool,
):
    best = None
    for threshold in thresholds:
        for overlap in overlaps:
            current = metrics(
                predictions_from_scores(scored, threshold, overlap, group_best),
                examples,
            )
            row = {
                **current,
                "threshold": threshold,
                "nms_overlap": overlap,
            }
            if best is None or (row["f1"], row["recall"]) > (
                best["f1"],
                best["recall"],
            ):
                best = row
    return best


def raw_remote_metrics(examples: Sequence[Example]) -> Dict[str, float]:
    return metrics(
        {example.example_id: set(example.noisy_spans) for example in examples},
        examples,
    )


def sliding_diagnostics(
    original_scored,
    sliding_scored,
    examples,
    threshold,
    overlap,
):
    original_sets = {
        example.example_id: set(example.noisy_spans) for example in examples
    }
    expanded_sets: Dict[str, set] = {}
    for example_id, rows in sliding_scored.items():
        expanded_sets[example_id] = {row[0] for row in rows}
    gold_total = sum(len(set(example.gold_spans)) for example in examples)
    original_hits = sum(
        len(original_sets.get(example.example_id, set()) & set(example.gold_spans))
        for example in examples
    )
    expanded_hits = sum(
        len(expanded_sets.get(example.example_id, set()) & set(example.gold_spans))
        for example in examples
    )
    pruned = predictions_from_scores(
        sliding_scored, threshold, overlap, group_best=True
    )
    pruned_metrics = metrics(pruned, examples)
    original_count = sum(len(value) for value in original_sets.values())
    kept_count = sum(len(value) for value in pruned.values())
    return {
        "original_candidate_recall": original_hits / gold_total if gold_total else 0.0,
        "sliding_oracle_recall": expanded_hits / gold_total if gold_total else 0.0,
        "candidate_retention": kept_count / original_count if original_count else 0.0,
        **{f"pruned_{key}": value for key, value in pruned_metrics.items()},
    }


def make_loader(dataset, tokenizer, max_length, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=SpanCollator(tokenizer, max_length),
    )


def write_outputs(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    rows = []
    for method, values in report["results"].items():
        train_values = values.get("train", {})
        rows.append(
            {
                "method": method,
                "candidate_source": values.get("candidate_source", ""),
                "sliding": values.get("sliding", False),
                "pruning": values.get("pruning", False),
                "train_precision": train_values.get("precision", ""),
                "train_recall": train_values.get("recall", ""),
                "train_f1": train_values.get("f1", ""),
                "dev_precision": values.get("dev", {}).get("precision", ""),
                "dev_recall": values.get("dev", {}).get("recall", ""),
                "dev_f1": values.get("dev", {}).get("f1", ""),
                "test_precision": values.get("test", {}).get("precision", ""),
                "test_recall": values.get("test", {}).get("recall", ""),
                "test_f1": values.get("test", {}).get("f1", ""),
                "candidate_retention": values.get("candidate_retention", ""),
            }
        )
    with (output_dir / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dict-train", default=str(PROJECT_ROOT / "sample_data" / "dict_train.txt")
    )
    parser.add_argument(
        "--chatgpt-train",
        default=str(PROJECT_ROOT / "sample_data" / "chatgpt_train.txt"),
    )
    parser.add_argument(
        "--gold-dev",
        default=str(PROJECT_ROOT / "sample_data" / "gold_dev.json"),
    )
    parser.add_argument(
        "--gold-test",
        default=str(PROJECT_ROOT / "sample_data" / "gold_test.json"),
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument("--output-dir", default="runs/ner_span_pruner")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--negative-margin", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-span-words", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.limit = args.limit or 8
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.max_span_words = min(args.max_span_words, 4)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dict_train = read_remote_conll(args.dict_train, args.limit)
    chatgpt_train = read_remote_conll(args.chatgpt_train, args.limit)
    gold_dev = read_gold_json(args.gold_dev, args.limit)
    gold_test = read_gold_json(args.gold_test, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    model = SpanTypeModel(
        args.model_name, args.projection_dim, args.local_files_only
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    train_loader = make_loader(
        CandidateDataset(dict_train, "warmup"),
        tokenizer,
        args.max_length,
        args.batch_size,
        True,
    )
    dev_loader = make_loader(
        CandidateDataset(gold_dev, "exhaustive", args.max_span_words),
        tokenizer,
        args.max_length,
        args.eval_batch_size,
        False,
    )
    thresholds = np.linspace(-0.2, 0.9, 23).round(3).tolist()
    overlaps = [0.0, 0.3, 0.5]
    best_epoch = None
    best_dev = None
    checkpoint = output_dir / "best_model.pt"

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Warm-up {epoch + 1}/{args.epochs}"):
            candidates = batch["candidates"].to(device)
            if candidates.numel() == 0:
                continue
            optimizer.zero_grad()
            similarities = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                candidates,
            )
            loss = warmup_loss(
                similarities,
                candidates,
                batch["labels"].to(device),
                args.temperature,
                args.negative_margin,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        dev_scores = collect_scores(model, dev_loader, device)
        dev_choice = tune_postprocessing(
            dev_scores, gold_dev, thresholds, overlaps, group_best=False
        )
        dev_choice["train_loss"] = float(np.mean(losses)) if losses else math.nan
        print(
            f"epoch={epoch + 1} loss={dev_choice['train_loss']:.4f} "
            f"dev_f1={dev_choice['f1']:.4f} threshold={dev_choice['threshold']} "
            f"nms={dev_choice['nms_overlap']}"
        )
        if best_dev is None or dev_choice["f1"] > best_dev["f1"]:
            best_dev = dev_choice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_loader = make_loader(
        CandidateDataset(gold_test, "exhaustive", args.max_span_words),
        tokenizer,
        args.max_length,
        args.eval_batch_size,
        False,
    )
    test_scores = collect_scores(model, test_loader, device)
    test_predictions = predictions_from_scores(
        test_scores,
        best_dev["threshold"],
        best_dev["nms_overlap"],
        group_best=False,
    )
    test_metrics = metrics(test_predictions, gold_test)

    original_loader = make_loader(
        CandidateDataset(chatgpt_train, "original"),
        tokenizer,
        args.max_length,
        args.eval_batch_size,
        False,
    )
    sliding_loader = make_loader(
        CandidateDataset(chatgpt_train, "sliding"),
        tokenizer,
        args.max_length,
        args.eval_batch_size,
        False,
    )
    original_scores = collect_scores(model, original_loader, device)
    sliding_scores = collect_scores(model, sliding_loader, device)
    diagnostics = sliding_diagnostics(
        original_scores,
        sliding_scores,
        chatgpt_train,
        best_dev["threshold"],
        best_dev["nms_overlap"],
    )

    dict_raw = raw_remote_metrics(dict_train)
    original_predictions = predictions_from_scores(
        original_scores,
        best_dev["threshold"],
        best_dev["nms_overlap"],
        group_best=False,
    )
    original_metrics = metrics(original_predictions, chatgpt_train)
    report = {
        "config": vars(args),
        "best_epoch": best_epoch,
        "selected_threshold": best_dev["threshold"],
        "selected_nms_overlap": best_dev["nms_overlap"],
        "chatgpt_train_diagnostics": diagnostics,
        "results": {
            "dictmatching_raw": {
                "candidate_source": "DictMatching train",
                "sliding": False,
                "pruning": False,
                "train": dict_raw,
            },
            "warmup_without_sliding": {
                "candidate_source": "ChatGPT train",
                "sliding": False,
                "pruning": True,
                "train": original_metrics,
            },
            "warmup_sliding_pruning": {
                "candidate_source": "exhaustive raw-text spans",
                "sliding": True,
                "pruning": True,
                "train": {
                    "precision": diagnostics["pruned_precision"],
                    "recall": diagnostics["pruned_recall"],
                    "f1": diagnostics["pruned_f1"],
                },
                "dev": {key: best_dev[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")},
                "test": test_metrics,
                "candidate_retention": diagnostics["candidate_retention"],
            },
        },
    }
    write_outputs(output_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote NER outputs to {output_dir}")


if __name__ == "__main__":
    main()
