#!/usr/bin/env python3
"""Reproducible BC5CDR dataset description and noisy-label audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


NOISY_DATASETS = ("bc5cdr-ChatGPT", "bc5cdr-DictMatching")
SPLITS = ("train", "dev", "test")
PUBTATOR_FILES = {
    "train": "CDR_TrainingSet.PubTator.txt",
    "dev": "CDR_DevelopmentSet.PubTator.txt",
    "test": "CDR_TestSet.PubTator.txt",
}
ENTITY_TYPES = ("Chemical", "Disease")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    entity_type: str


@dataclass
class PubTatorDocument:
    pmid: str
    title: str
    abstract: str
    entities: list[dict]
    relations: list[dict]

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    resource_root = Path(
        os.environ.get(
            "CLLLM_RESOURCES",
            str(Path.home() / "Documents" / "Clllm_resources"),
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=resource_root / "shared_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=resource_root / "runs" / "data_description",
    )
    return parser.parse_args()


def read_conll(path: Path) -> list[list[list[str]]]:
    sequences: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            if current:
                sequences.append(current)
                current = []
            continue
        fields = raw_line.split("\t")
        if len(fields) not in (2, 3):
            raise ValueError(f"{path}:{line_no}: expected 2 or 3 tab-separated fields")
        current.append(fields)
    if current:
        sequences.append(current)
    return sequences


def parse_pubtator(path: Path) -> list[PubTatorDocument]:
    documents: list[PubTatorDocument] = []
    blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
    for block in blocks:
        title = ""
        abstract = ""
        pmid = ""
        entities: list[dict] = []
        relations: list[dict] = []
        for line in block.splitlines():
            if "|t|" in line or "|a|" in line:
                doc_id, section, text = line.split("|", 2)
                pmid = doc_id
                if section == "t":
                    title = text
                else:
                    abstract = text
                continue
            fields = line.split("\t")
            if len(fields) == 6:
                entities.append(
                    {
                        "pmid": fields[0],
                        "start": int(fields[1]),
                        "end": int(fields[2]),
                        "mention": fields[3],
                        "entity_type": fields[4],
                        "mesh_id": fields[5],
                    }
                )
            elif len(fields) == 4 and fields[1] == "CID":
                relations.append(
                    {
                        "pmid": fields[0],
                        "relation": fields[1],
                        "chemical_id": fields[2],
                        "disease_id": fields[3],
                    }
                )
        documents.append(PubTatorDocument(pmid, title, abstract, entities, relations))
    return documents


def labels_to_spans(labels: Sequence[str]) -> tuple[set[Span], list[dict]]:
    """Convert BIO labels to strict spans; repair an illegal I-X as B-X and report it."""
    spans: set[Span] = set()
    invalid: list[dict] = []
    start: int | None = None
    entity_type: str | None = None

    def close(end: int) -> None:
        nonlocal start, entity_type
        if start is not None and entity_type is not None:
            spans.add(Span(start, end, entity_type))
        start = None
        entity_type = None

    for index, label in enumerate(list(labels) + ["O"]):
        if label == "O":
            close(index)
            continue
        if "-" not in label:
            raise ValueError(f"Invalid BIO label {label!r} at token {index}")
        prefix, current_type = label.split("-", 1)
        if current_type not in ENTITY_TYPES or prefix not in {"B", "I"}:
            raise ValueError(f"Invalid BIO label {label!r} at token {index}")
        if prefix == "B":
            close(index)
            start = index
            entity_type = current_type
        elif start is None or entity_type != current_type:
            invalid.append(
                {
                    "token_index": index,
                    "label": label,
                    "reason": "I label without a matching preceding B/I span",
                }
            )
            close(index)
            start = index
            entity_type = current_type
    return spans, invalid


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
    }


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text)


def span_text(tokens: Sequence[str], span: Span) -> str:
    return " ".join(tokens[span.start : span.end])


def conll_stats(sequences: Sequence[Sequence[Sequence[str]]], label_col: int) -> dict:
    span_counter: Counter[str] = Counter()
    invalid_count = 0
    for sequence in sequences:
        spans, invalid = labels_to_spans([row[label_col] for row in sequence])
        span_counter.update(span.entity_type for span in spans)
        invalid_count += len(invalid)
    return {
        "sequences": len(sequences),
        "tokens": sum(len(sequence) for sequence in sequences),
        "mentions": sum(span_counter.values()),
        "chemical_mentions": span_counter["Chemical"],
        "disease_mentions": span_counter["Disease"],
        "invalid_i_transitions": invalid_count,
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_scale_rows(data_root: Path) -> tuple[list[dict], dict[str, list[PubTatorDocument]]]:
    gold_by_split: dict[str, list[PubTatorDocument]] = {}
    rows: list[dict] = []
    for split in SPLITS:
        documents = parse_pubtator(data_root / "BC5CDR" / PUBTATOR_FILES[split])
        gold_by_split[split] = documents
        entity_counts = Counter(
            entity["entity_type"] for document in documents for entity in document.entities
        )
        rows.append(
            {
                "dataset": "BC5CDR-PubTator-gold",
                "split": split,
                "annotation": "gold span + entity type + MeSH ID",
                "documents": len(documents),
                "conll_sequences": "",
                "tokens": "",
                "mentions": sum(entity_counts.values()),
                "chemical_mentions": entity_counts["Chemical"],
                "disease_mentions": entity_counts["Disease"],
                "cid_relations": sum(len(document.relations) for document in documents),
                "invalid_i_transitions": "",
            }
        )

    for dataset in NOISY_DATASETS:
        for split in SPLITS:
            sequences = read_conll(data_root / dataset / f"{split}.txt")
            stats = conll_stats(sequences, 1)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "annotation": "noisy BIO" if split == "train" else "gold BIO",
                    "documents": 500,
                    "conll_sequences": stats["sequences"],
                    "tokens": stats["tokens"],
                    "mentions": stats["mentions"],
                    "chemical_mentions": stats["chemical_mentions"],
                    "disease_mentions": stats["disease_mentions"],
                    "cid_relations": "",
                    "invalid_i_transitions": stats["invalid_i_transitions"],
                }
            )
    return rows, gold_by_split


def build_consistency_rows(
    data_root: Path, gold_by_split: dict[str, list[PubTatorDocument]]
) -> list[dict]:
    rows: list[dict] = []
    for split in SPLITS:
        chatgpt = read_conll(data_root / "bc5cdr-ChatGPT" / f"{split}.txt")
        dictionary = read_conll(data_root / "bc5cdr-DictMatching" / f"{split}.txt")
        same_shape = len(chatgpt) == len(dictionary) and all(
            len(left) == len(right) for left, right in zip(chatgpt, dictionary)
        )
        same_tokens = same_shape and all(
            left_row[0] == right_row[0]
            for left_sequence, right_sequence in zip(chatgpt, dictionary)
            for left_row, right_row in zip(left_sequence, right_sequence)
        )
        noisy_label_differences = (
            sum(
                left_row[1] != right_row[1]
                for left_sequence, right_sequence in zip(chatgpt, dictionary)
                for left_row, right_row in zip(left_sequence, right_sequence)
            )
            if same_shape
            else ""
        )
        train_gold_same = (
            split != "train"
            or same_shape
            and all(
                left_row[2] == right_row[2]
                for left_sequence, right_sequence in zip(chatgpt, dictionary)
                for left_row, right_row in zip(left_sequence, right_sequence)
            )
        )

        gold_documents = [normalized(document.text) for document in gold_by_split[split]]
        conll_sequences = [
            "".join(row[0] for row in sequence) for sequence in chatgpt
        ]
        matched_sequences = sum(
            any(sequence in document for document in gold_documents)
            for sequence in conll_sequences
        )
        conll_all_text = "".join(conll_sequences)
        gold_all_text = "".join(gold_documents)
        rows.append(
            {
                "split": split,
                "chatgpt_vs_dict_same_sequence_boundaries": same_shape,
                "chatgpt_vs_dict_same_tokens": same_tokens,
                "chatgpt_vs_dict_noisy_label_token_differences": noisy_label_differences,
                "train_gold_column_same": train_gold_same,
                "conll_sequences_found_in_pubtator": matched_sequences,
                "total_conll_sequences": len(conll_sequences),
                "all_sequences_found_in_pubtator": matched_sequences
                == len(conll_sequences),
                "normalized_character_count_same": len(conll_all_text)
                == len(gold_all_text),
                "normalized_character_multiset_same": Counter(conll_all_text)
                == Counter(gold_all_text),
                "same_document_order": conll_all_text == gold_all_text,
                "interpretation": (
                    "same official split content; CoNLL sequences are reordered and omit PMID"
                ),
            }
        )
    return rows


def evaluate_noisy_dataset(data_root: Path, dataset: str) -> tuple[list[dict], list[dict], dict]:
    sequences = read_conll(data_root / dataset / "train.txt")
    totals = {
        "strict-overall": [0, 0, 0],
        "strict-Chemical": [0, 0, 0],
        "strict-Disease": [0, 0, 0],
        "token-overall": [0, 0, 0],
    }
    examples: list[dict] = []
    invalid_examples: list[dict] = []

    for sequence_index, sequence in enumerate(sequences):
        tokens = [row[0] for row in sequence]
        noisy_labels = [row[1] for row in sequence]
        gold_labels = [row[2] for row in sequence]
        predicted, predicted_invalid = labels_to_spans(noisy_labels)
        gold, gold_invalid = labels_to_spans(gold_labels)

        for key, predicted_set, gold_set in (
            ("strict-overall", predicted, gold),
            (
                "strict-Chemical",
                {span for span in predicted if span.entity_type == "Chemical"},
                {span for span in gold if span.entity_type == "Chemical"},
            ),
            (
                "strict-Disease",
                {span for span in predicted if span.entity_type == "Disease"},
                {span for span in gold if span.entity_type == "Disease"},
            ),
        ):
            totals[key][0] += len(predicted_set & gold_set)
            totals[key][1] += len(predicted_set - gold_set)
            totals[key][2] += len(gold_set - predicted_set)

        for noisy_label, gold_label in zip(noisy_labels, gold_labels):
            if noisy_label == gold_label and gold_label != "O":
                totals["token-overall"][0] += 1
            elif noisy_label != "O" and gold_label == "O":
                totals["token-overall"][1] += 1
            elif noisy_label == "O" and gold_label != "O":
                totals["token-overall"][2] += 1
            elif noisy_label != gold_label:
                totals["token-overall"][1] += 1
                totals["token-overall"][2] += 1

        if len(examples) < 18:
            for error_type, error_spans in (
                ("FP", sorted(predicted - gold, key=lambda span: (span.start, span.end))),
                ("FN", sorted(gold - predicted, key=lambda span: (span.start, span.end))),
            ):
                for span in error_spans[:2]:
                    examples.append(
                        {
                            "dataset": dataset,
                            "sequence_index": sequence_index,
                            "error_type": error_type,
                            "entity_type": span.entity_type,
                            "mention": span_text(tokens, span),
                            "token_start": span.start,
                            "token_end": span.end,
                            "sequence_text": " ".join(tokens),
                        }
                    )
        for source, invalid_rows in (
            ("noisy", predicted_invalid),
            ("gold", gold_invalid),
        ):
            for invalid in invalid_rows[:3]:
                if len(invalid_examples) < 20:
                    invalid_examples.append(
                        {
                            "dataset": dataset,
                            "sequence_index": sequence_index,
                            "source": source,
                            **invalid,
                            "token": tokens[invalid["token_index"]],
                        }
                    )

    metric_rows: list[dict] = []
    for metric_name, (tp, fp, fn) in totals.items():
        level, entity_type = metric_name.split("-", 1)
        metric_rows.append(
            {
                "dataset": dataset,
                "level": level,
                "entity_type": entity_type,
                **metrics(tp, fp, fn),
            }
        )
    return metric_rows, examples, {"invalid_bio_examples": invalid_examples}


def literature_rows() -> list[dict]:
    return [
        {
            "study": "BINDER",
            "dataset": "BC5CDR",
            "setting": "distantly supervised NER",
            "train_data": "500 training documents; dictionary exact matching labels",
            "dev_data": "500 development documents; dictionary exact matching labels",
            "test_data": "500 test documents; gold labels",
            "precision": 87.6,
            "recall": 76.3,
            "f1": 81.6,
            "note": "paper result; not a score computed from this repository",
            "paper": "https://arxiv.org/abs/2208.14565",
            "code": "https://github.com/microsoft/binder",
        }
    ]


def related_dataset_rows(data_root: Path) -> list[dict]:
    return [
        {
            "dataset": "BC5CDR",
            "domain": "biomedical abstracts",
            "entity_types": "Chemical; Disease",
            "local_data": (data_root / "BC5CDR").exists(),
            "candidate_dictionary": "MeSH chemical/disease descriptors and synonyms",
            "dict_matching": "highly suitable",
            "main_risk": "abbreviations, ambiguous synonyms, boundary mismatch",
            "recommended_next_step": "current audit target",
        },
        {
            "dataset": "NCBI Disease",
            "domain": "biomedical abstracts",
            "entity_types": "Disease",
            "local_data": (data_root / "NCBI").exists(),
            "candidate_dictionary": "MeSH/CTD disease names and synonyms",
            "dict_matching": "highly suitable",
            "main_risk": "composite mentions and abbreviation resolution",
            "recommended_next_step": "easiest local replication after BC5CDR",
        },
        {
            "dataset": "JNLPBA",
            "domain": "biomedical abstracts",
            "entity_types": "protein; DNA; RNA; cell line; cell type",
            "local_data": False,
            "candidate_dictionary": "UniProt, NCBI Gene, Cellosaurus, ontology terms",
            "dict_matching": "partly suitable",
            "main_risk": "high ambiguity and broad surface-form variation",
            "recommended_next_step": "use type-specific dictionaries and disambiguation",
        },
    ]


def validate_expected_results(scale_rows: Sequence[dict], consistency_rows: Sequence[dict], metric_rows: Sequence[dict]) -> None:
    lookup = {
        (row["dataset"], row["split"]): row
        for row in scale_rows
        if row["dataset"] in NOISY_DATASETS
    }
    for dataset in NOISY_DATASETS:
        train = lookup[(dataset, "train")]
        assert train["conll_sequences"] == 4560
        assert train["tokens"] == 118170
    train_consistency = next(row for row in consistency_rows if row["split"] == "train")
    assert train_consistency["chatgpt_vs_dict_same_tokens"] is True
    assert train_consistency["chatgpt_vs_dict_noisy_label_token_differences"] == 30333
    for split in ("dev", "test"):
        row = next(item for item in consistency_rows if item["split"] == split)
        assert row["chatgpt_vs_dict_noisy_label_token_differences"] == 0
    expected = {
        "bc5cdr-ChatGPT": (33.08, 61.97, 43.14),
        "bc5cdr-DictMatching": (89.45, 66.95, 76.58),
    }
    for dataset, values in expected.items():
        row = next(
            item
            for item in metric_rows
            if item["dataset"] == dataset
            and item["level"] == "strict"
            and item["entity_type"] == "overall"
        )
        assert (row["precision"], row["recall"], row["f1"]) == values


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_rows, gold_by_split = build_scale_rows(data_root)
    consistency_rows = build_consistency_rows(data_root, gold_by_split)

    metric_rows: list[dict] = []
    error_examples: list[dict] = []
    diagnostics: dict[str, dict] = {}
    for dataset in NOISY_DATASETS:
        dataset_metrics, dataset_examples, dataset_diagnostics = evaluate_noisy_dataset(
            data_root, dataset
        )
        metric_rows.extend(dataset_metrics)
        error_examples.extend(dataset_examples)
        diagnostics[dataset] = dataset_diagnostics

    binder_rows = literature_rows()
    related_rows = related_dataset_rows(data_root)
    validate_expected_results(scale_rows, consistency_rows, metric_rows)

    write_csv(output_dir / "bc5cdr_scale.csv", scale_rows)
    write_csv(output_dir / "bc5cdr_consistency.csv", consistency_rows)
    write_csv(output_dir / "bc5cdr_noisy_metrics.csv", metric_rows)
    write_csv(output_dir / "bc5cdr_error_examples.csv", error_examples)
    write_csv(output_dir / "binder_bc5cdr.csv", binder_rows)
    write_csv(output_dir / "related_biomedical_datasets.csv", related_rows)

    gold_example = gold_by_split["train"][0]
    report = {
        "notes": {
            "conll_sequence_definition": (
                "A sequence is a block separated by a blank line; it is not necessarily "
                "a linguistically segmented sentence."
            ),
            "gold_authority": (
                "PubTator is authoritative for document counts, original mention spans, "
                "entity types, MeSH IDs, and CID relations."
            ),
            "evaluation_gold": (
                "The third train.txt column is the token-aligned gold BIO label used for "
                "noisy-vs-gold evaluation."
            ),
            "remote_pred_warning": (
                "data/data_pre/remote_pred is not used: its current files contain gold "
                "mentions rather than the second-column noisy labels."
            ),
            "merged_json_warning": (
                "merged_bc5cdr JSON is not used as the main gold count because its chunking "
                "pipeline drops mentions crossing or failing reconstructed chunk boundaries."
            ),
        },
        "pubtator_format": {
            "text_line": "PMID|t|title or PMID|a|abstract",
            "entity_line": (
                "PMID<TAB>start<TAB>end<TAB>mention<TAB>entity_type<TAB>MeSH_ID"
            ),
            "relation_line": (
                "PMID<TAB>CID<TAB>chemical_MeSH_ID<TAB>disease_MeSH_ID"
            ),
            "offset_convention": "0-based character offsets; end is exclusive",
            "example_document": asdict(gold_example),
        },
        "diagnostics": diagnostics,
    }
    (output_dir / "bc5cdr_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote BC5CDR audit outputs to {output_dir}")
    for row in metric_rows:
        if row["level"] == "strict" and row["entity_type"] == "overall":
            print(
                f"{row['dataset']}: P={row['precision']:.2f} "
                f"R={row['recall']:.2f} F1={row['f1']:.2f}"
            )


if __name__ == "__main__":
    main()
