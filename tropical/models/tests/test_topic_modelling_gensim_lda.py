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

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20.csv?"\
"x-goog-signature=7a038d055fa6b6fd06384a84690deae8beda042450efbfd4935e23be843c624db6d34a573faa084a0539029d0f85dddedb6"\
"88c843ad0671fe6578da0b041675ed9f68b315dafd97da197dd44503096cce45cdd364592fc61df16f6be41f8c8268a537b40bb57a6ea2c61fa4"\
"381f3973222a57a5dbc38d82b1b0d91cbefb35024a173cbd726363dcdc8ba5b412de448b6309d5121eee470163c46750e92f93fc8b5e40750365"\
"184a337c927b72c8cf9cedef830ecc475fec72a594fdea869a5f683e131144ae37f6413d84c79605f485aeb8a999f989f718fe63d68f0c6b2984"\
"6f73ece8dac59a30cb68b9397947855f3caeb25aa41c1cba52d36f711c96fd031&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credentia"\
"l=gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200818%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=20200"\
"818T082034Z&x-goog-expires=604800&x-goog-signedheaders=host"

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
