import unittest
import wget
import os
import pandas as pd
import json

from tropical.models import ngram_analysis_gensim
from tropical.models.tests import BaseTestCase


class TestNGramAnalysis(BaseTestCase):

    def test(self):
        print("Testing TestNGramAnalysis.test ...", flush=True)

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20.csv?x-" \
                   "goog-signature=7ec8f4ad2861be33c91dd09c07f6b73b9236a432ad2d8dd0d781f66a000798c8a0a82640640cb08cb23342c7bedd6c21e3355" \
                   "36eefdad8557ad70ae849ae4fb082b905c583bdbc6121029c696bc1d4c0986bdeb82862e1312bde3b3c1f7bfd25c2f8c3a3d4bcd586e1876128e" \
                   "f5b6f84bae7794f5a54e1be4b189972a3bd4dd17e23453b8cb88c34df4de2752936f4b1c9d349f40106849b0c29ad33e0bce6e3b3cce3679097b" \
                   "a016cb24e0fd2d879c5cd0bcc85097ef5f3721c4c393344d2b33e7a9e785008e0b457a1cce9f3bcf81cb2fae01f638e277eacac8e0131c7620df" \
                   "b4ff902dc59da4ea7f28981b11d5644e0d6c44ba03ae926f20569c9a66d1a59&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=" \
                   "gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200427%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=202004" \
                   "27T170613Z&x-goog-expires=604800&x-goog-signedheaders=host"

        # ===
        filename = wget.download(file_url)
        print("filename =", filename)
        df = pd.read_csv(filename)
        os.remove(filename)

        print(df.columns)
        print(df.describe())
        print(df.sample(n=5))
        # ===

        analyser = ngram_analysis_gensim.NGramAnalysisGensim()
        self.assertTrue(analyser is not None)
        result = analyser.analyse_dataframe(df, delimiter=b' ')

        # ToDo: Add a reference static test file and add some specific result tests here.

        print(json.dumps(result, indent=4, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
