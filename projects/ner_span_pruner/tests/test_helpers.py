import unittest

from projects.ner_span_pruner.run import (
    Example,
    Span,
    bio_to_spans,
    boundary_variants,
    metrics,
    predictions_from_scores,
    read_gold_json,
    read_remote_conll,
)


class NERHelpersTest(unittest.TestCase):
    def test_bio_conversion_repairs_invalid_i(self):
        text, words, spans = bio_to_spans(
            ["aspirin", "toxicity"], ["I-Chemical", "B-Disease"]
        )
        self.assertEqual(text, "aspirin toxicity")
        self.assertEqual(words, [(0, 7), (8, 16)])
        self.assertEqual(
            spans, [Span(0, 7, "Chemical"), Span(8, 16, "Disease")]
        )

    def test_sliding_adds_expansion_and_shrinkage(self):
        example = Example(
            "x",
            "severe kidney failure today",
            [(0, 6), (7, 13), (14, 21), (22, 27)],
            [],
            [Span(7, 21, "Disease")],
        )
        names = {
            candidate.variant
            for candidate in boundary_variants(
                example, Span(7, 21, "Disease"), "x:0", labeled=True
            )
        }
        self.assertTrue(
            {"expand_left", "expand_right", "shrink_left", "shrink_right"}
            <= names
        )

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
        result = metrics(
            predicted,
            [Example("x", "aspirin toxicity", [], [], [Span(0, 16, "Chemical")])],
        )
        self.assertEqual(result["f1"], 1.0)

    def test_committed_sample_data_loads(self):
        from projects.ner_span_pruner.run import PROJECT_ROOT

        sample = PROJECT_ROOT / "sample_data"
        self.assertGreaterEqual(len(read_remote_conll(sample / "dict_train.txt")), 8)
        self.assertGreaterEqual(len(read_remote_conll(sample / "chatgpt_train.txt")), 8)
        self.assertGreaterEqual(len(read_gold_json(sample / "gold_dev.json")), 8)
        self.assertGreaterEqual(len(read_gold_json(sample / "gold_test.json")), 8)


if __name__ == "__main__":
    unittest.main()
