import json
from pathlib import Path


REMOTE_DATASETS = ("bc5cdr-ChatGPT", "bc5cdr-DictMatching")
SPLITS = {
    "train": "train",
    "dev": "development",
    "test": "test",
}


def convert_item(item):
    text = item["text"]
    entities = []
    for ent_type, start, end in zip(
        item.get("entity_types", []),
        item.get("entity_start_chars", []),
        item.get("entity_end_chars", []),
    ):
        mention = text[start:end]
        entities.append([mention, start, end, ent_type.capitalize(), "REMOTE"])

    return {
        "text": text,
        "sentence_id": item.get("id", ""),
        "source": "remote_annotation",
        "entities": entities,
    }


def convert_file(input_path, output_path):
    output_data = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            converted = convert_item(item)
            for mention, start, end, ent_type, _ in converted["entities"]:
                if ent_type not in {"Chemical", "Disease"}:
                    raise ValueError(f"{input_path}:{line_no} unsupported entity type: {ent_type}")
                if converted["text"][start:end] != mention:
                    raise ValueError(f"{input_path}:{line_no} invalid span: {mention!r}")
            output_data.append(converted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    return len(output_data), sum(len(item["entities"]) for item in output_data)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    input_root = repo_root / "data"
    output_root = repo_root / "data" / "data_pre" / "remote_pred" / "BC5CDR"

    for dataset in REMOTE_DATASETS:
        for in_split, out_split in SPLITS.items():
            input_path = input_root / dataset / f"{in_split}.json"
            output_path = output_root / dataset / f"merged_bc5cdr_{out_split}_data.json"
            rows, entities = convert_file(input_path, output_path)
            print(f"{dataset}/{in_split} -> {output_path.relative_to(repo_root)} ({rows} rows, {entities} entities)")


if __name__ == "__main__":
    main()
