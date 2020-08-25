import unittest
import wget
import os
import pandas as pd
import json

from tropical.models import topic_modelling_gensim_LDA
from tropical.models.tests import BaseTestCase


class TestTopicModellingGensimLDA(BaseTestCase):

    def test(self):
        print("Testing TestTopicModellingGensimLDA.test ...", flush=True)

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that"\
            "_triggered_catchall_2020_04_20.csv?x-goog-signature=41036ee6b5e4c59e009009cdfbaa5c9d4"\
            "dc220ebf3da5b1357a34f0279779583053865aa8f576bd6c85c49f874f1a82a74c4f24c7509a27f996b97"\
            "5bdf9b3755f8faba2c06fa6ff6067d96284b9e7adf6407e4c3dccc35b255408adcdfc741a93703a8eee11"\
            "d1718ecd43b5a1cddd7ca29fd78e795c5bffc43255226686e797197b10ce7619c8048d09b5b5b0bbc7857"\
            "d0c96cbf9bae18ef724a186a1ce3ff2993ec0463cac806130d8bd524af0ba2498cdee7b55251a2925cebf"\
            "559caa3eb2d4f13e0deaa97228054fd3c9ed12973038f18f7d84c5a93d4789790c29d92fa174a3b19dd07"\
            "90c82cbe811506bd014750e555734689e13ab6f21a3a25a792ac89&x-goog-algorithm=GOOG4-RSA-SH"\
            "A256&x-goog-credential=gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F202008"\
            "25%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=20200825T082713Z&x-goog-expires=604"\
            "800&x-goog-signedheaders=host"

        # ===
        filename = wget.download(file_url)
        df = pd.read_csv(filename, na_filter=False, nrows=200)
        os.remove(filename)

        analyser = topic_modelling_gensim_LDA.TopicModellingGensimLDA(max_topics=10, min_topics=5, step=5)
        self.assertTrue(analyser is not None)
        result = analyser.analyse_dataframe(df, delimiter=b'_')
        # # ToDo: Add a reference static test file and add some specific result tests here.
        #
        print(json.dumps(result, indent=4, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
