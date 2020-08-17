import unittest
import json
import os

import wget
import pandas as pd

from tropical.models import ngram_analysis_gensim_bpe
from tropical.models.tests import BaseTestCase


class TestNGramAnalysis(BaseTestCase):

    def test(self):
        print("Testing TestNGramAnalysis.test ...", flush=True)

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20.csv?x-" \
                   "goog-signature=a4cdb1d4f809174ffbb1954f210cbfa2586c6d52783a328d46e95dbacd1fdf6c2f33660fdc603df9a8599357dd341c990eb10" \
                   "f918561e702962f4f79d673436d07748f000e9bfbfbc0887e207995c7b8b5a8fb5b49dc01a3d5de5418fbe729e92ce76de4d35d7e736c924b411" \
                   "7641b0a6a554e8bbb2c715eef4b86a9c669bdff7866c374197aa8d579e976116ee637012f06e27df16b9dc26d40ae3249593abf661d7224d4397" \
                   "d1d0133b00f635cb434c38c0b3dd199344c434bcb5efc8a7c7f9175e855fd0c16d0121c4cbf6b46889f86528460e05fae232848a3d2e598d19b7" \
                   "0bb9f3a4bae05a2995618806c9e128c1cab4fb7e52cb016673814074584a16e&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=" \
                   "gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200523%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=202005" \
                   "23T124251Z&x-goog-expires=604800&x-goog-signedheaders=host"

        # ===
        filename = wget.download(file_url)
        print("filename =", filename)
        df = pd.read_csv(filename, na_filter=False)
        os.remove(filename)

        print(df.columns)
        print(df.describe())
        print(df.sample(n=5))
        # ===

        analyser = ngram_analysis_gensim_bpe.NGramAnalysisGensimBPE()
        self.assertTrue(analyser is not None)
        result = analyser.analyse_dataframe(df, delimiter=b'|')

        # ToDo: Add a reference static test file and add some specific result tests here.

        print(json.dumps(result, indent=4, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
