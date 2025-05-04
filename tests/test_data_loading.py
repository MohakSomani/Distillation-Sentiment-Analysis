import unittest
from datasets import load_from_disk

class TestDataLoading(unittest.TestCase):
    def test_sst2(self):
        dataset = load_from_disk("data/raw/sst2")
        self.assertGreater(len(dataset["train"]), 1000)

    def test_imdb(self):
        dataset = load_from_disk("data/raw/imdb")
        self.assertEqual(len(dataset["train"].column_names), 2)

if __name__ == "__main__":
    unittest.main()