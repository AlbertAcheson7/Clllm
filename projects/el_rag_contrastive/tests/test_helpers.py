import unittest

from projects.el_rag_contrastive.run import (
    PROJECT_ROOT,
    attach_paths,
    discover_dataset_files,
    entity_text,
    exact_rankings,
    normalize_name,
    ranking_metrics,
    read_entities,
    read_pairs,
    read_paths,
)


class ELHelpersTest(unittest.TestCase):
    def test_committed_sample_data_loads(self):
        paths = discover_dataset_files(PROJECT_ROOT / "sample_data")
        source = attach_paths(
            read_entities(paths["source_entities"]), read_paths(paths["source_paths"])
        )
        target = attach_paths(
            read_entities(paths["target_entities"]), read_paths(paths["target_paths"])
        )
        train = read_pairs(paths["train_pairs"], source, target)
        test = read_pairs(paths["test_pairs"], source, target)
        self.assertGreaterEqual(len(source), 10)
        self.assertGreaterEqual(len(target), 20)
        self.assertGreaterEqual(len(train), 5)
        self.assertGreaterEqual(len(test), 5)

    def test_exact_name_metrics(self):
        paths = discover_dataset_files(PROJECT_ROOT / "sample_data")
        source = read_entities(paths["source_entities"])
        target = read_entities(paths["target_entities"])
        pairs = read_pairs(paths["test_pairs"], source, target)
        source_ids = list(dict.fromkeys(left for left, _ in pairs))
        result = ranking_metrics(exact_rankings(source, target, source_ids), pairs)
        self.assertIn("hits_at_1", result)
        self.assertEqual(normalize_name("Diabetes-mellitus"), "diabetes mellitus")

    def test_hierarchy_is_available(self):
        paths = discover_dataset_files(PROJECT_ROOT / "sample_data")
        source = attach_paths(
            read_entities(paths["source_entities"]), read_paths(paths["source_paths"])
        )
        first = next(iter(source.values()))
        self.assertIn("Hierarchy:", entity_text(first, "name_path"))


if __name__ == "__main__":
    unittest.main()
