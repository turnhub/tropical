import unittest
import wget
import os
import pandas as pd
import json

from tropical.models import semantic_cluster_analysis
from tropical.models.tests import BaseTestCase


class TestSemanticClusterAnalysis(BaseTestCase):

    def test(self):
        print("Testing TestSemanticClusterAnalysis.test ...", flush=True)

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20." \
                   "csv?x-goog-signature=57da4f3fe51e7345ecbfd3a332aee3e91fa425a9bb3eb358d56283bb7500b9611e3b5bf758e8c5b95a44fe70a57f4" \
                   "df4ae17ce96cf7c9f5401f1d2b89e6dfe1bc2aba8f39ebb2d3c190006f6b75372fc4d74a6bc15093fa746a7faaf4a7e13202b63ba786e6629b" \
                   "7ced41d149deb54e686e4df7cac4094c31017ed15e6415fde7b1a0935069017ac6d18d42689d8d16b410ecb0f1b5e81e3ca4743fb6a6ad1f37" \
                   "abbed51cb4ce045c2a68f50dcf6efa15b05747060a978a8ad6661e578577204e794a86502cf6604e5f38be943d9e9d2b8c62eefc505705e6e9" \
                   "42c8a714a015d6d9aa1054ec64e09aad15319f6a34e7f5959094defa2c594d9e39306176e6875&x-goog-algorithm=GOOG4-RSA-SHA256&x-" \
                   "goog-credential=gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200617%2Fmulti%2Fstorage%2Fgoog4_request&" \
                   "x-goog-date=20200617T030749Z&x-goog-expires=604800&x-goog-signedheaders=host"

        # ===
        filename = wget.download(file_url)

        print("filename =", filename)
        df = pd.read_csv(filename, na_filter=False)
        os.remove(filename)

        print(df.columns)
        print(df.describe())
        print(df.sample(n=5))
        print(df.sample(n=5))
        # ===

        analyser = semantic_cluster_analysis.SemanticClusterAnalysis()
        self.assertTrue(analyser is not None)
        result = analyser.analyse_dataframe(df)

        # ToDo: Add a reference static test file and add some specific result tests here.

        print(json.dumps(result, indent=4, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
