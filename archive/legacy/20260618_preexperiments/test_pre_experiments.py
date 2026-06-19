import tempfile
import unittest
from pathlib import Path

from experiments.el.run_rag_contrastive import (
    Entity,
    attach_paths,
    entity_text,
    exact_rankings,
    normalize_name,
    ranking_metrics,
    read_entities,
    read_paths,
    read_pairs,
)
from experiments.ner.run_span_pruner import (
    Example,
    Span,
    bio_to_spans,
    boundary_variants,
    metrics,
    predictions_from_scores,
)


class NERHelpersTest(unittest.TestCase):
    def test_bio_conversion_repairs_invalid_i(self):
        text, words, spans = bio_to_spans(
            ["aspirin", "toxicity"],
            ["I-Chemical", "B-Disease"],
        )
        self.assertEqual(text, "aspirin toxicity")
        self.assertEqual(words, [(0, 7), (8, 16)])
        self.assertEqual(
            spans,
            [Span(0, 7, "Chemical"), Span(8, 16, "Disease")],
        )

    def test_sliding_adds_expansion_and_shrinkage(self):
        example = Example(
            "x",
            "severe kidney failure today",
            [(0, 6), (7, 13), (14, 21), (22, 27)],
            [],
            [Span(7, 21, "Disease")],
        )
        variants = boundary_variants(
            example, Span(7, 21, "Disease"), "x:0", labeled=True
        )
        names = {candidate.variant for candidate in variants}
        self.assertIn("expand_left", names)
        self.assertIn("expand_right", names)
        self.assertIn("shrink_left", names)
        self.assertIn("shrink_right", names)

    def test_group_best_threshold_and_nms(self):
        scored = {
            "x": [
                (Span(0, 7, "Chemical"), 0.8, "x:0", "original"),
                (Span(0, 16, "Chemical"), 0.9, "x:0", "expand_right"),
                (Span(8, 16, "Chemical"), 0.7, "x:1", "original"),
            ]
        }
        predicted = predictions_from_scores(scored, 0.5, 0.3, group_best=True)
        self.assertEqual(predicted["x"], {Span(0, 16, "Chemical")})
        evaluated = metrics(
            predicted,
            [Example("x", "aspirin toxicity", [], [], [Span(0, 16, "Chemical")])],
        )
        self.assertEqual(evaluated["f1"], 1.0)


class ELHelpersTest(unittest.TestCase):
    def test_entity_pair_parsing_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entities_path = root / "entities.tsv"
            entities_path.write_text("t1\tDiabetes mellitus\nt2\tAsthma\n")
            source_path = root / "source.tsv"
            source_path.write_text("s1\tdiabetes-mellitus\n")
            pairs_path = root / "pairs"
            pairs_path.write_text("s1\tt1\n")
            target = read_entities(str(entities_path))
            source = read_entities(str(source_path))
            pairs = read_pairs(str(pairs_path), source, target)
            rankings = exact_rankings(source, target, ["s1"])
            result = ranking_metrics(rankings, pairs)
            self.assertEqual(normalize_name("Diabetes-mellitus"), "diabetes mellitus")
            self.assertEqual(result["hits_at_1"], 1.0)
            self.assertEqual(result["recall_at_20"], 1.0)

    def test_integration_paths_are_attached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entities_path = root / "ent_ids_1.txt"
            paths_path = root / "paths_1.txt"
            entities_path.write_text("0\tCholera\n")
            paths_path.write_text(
                "0\tCertain infectious diseases, Intestinal infectious diseases\t\n"
            )
            entities = attach_paths(
                read_entities(str(entities_path)), read_paths(str(paths_path))
            )
            self.assertIn(
                "Hierarchy: Certain infectious diseases",
                entity_text(entities["0"], "name_path"),
            )


if __name__ == "__main__":
    unittest.main()
