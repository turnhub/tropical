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

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall"\
                "_2020_04_20.csv?x-goog-signature=6a5cda9bcbbfad95e1c35887c3f72b3b3ed19d772bd5cecb45c498ff85495ca3fb"\
                "30b889dcc8884351dcb05006a3aa4cd1d6e54d25ce6081a0be83e039a684ed420adc963ab6a2b89eb25b3629b74b565252a9"\
                "8b24fe9b16bbaefd68050fc879c4a20cbf5ca2d7d9015b42c7fb9de75756cf55fa72b086f5bb3e5c8f4ecbd86375f48f66461"\
                "857e3e57835f8e9ded72dbe5b9b2480791c62b240ad84cfad82b83fd2164bd4842a3bfcb1f8e9e4ed03a0330d72561572809"\
                "aa066bc88e438c579b4dd61000701b2af2659c29b5277e758ea9813894b552880c8c99236fc657dced20bcac77e7e4e529fc"\
                "a1ce3d8d9bcf6182cd39b48723a890b3a845b98144b26&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=gc"\
                "p-storage%40feersum-221018.iam.gserviceaccount.com%2F20200727%2Fmulti%2Fstorage%2Fgoog4_request&x-go"\
                "og-date=20200727T140145Z&x-goog-expires=604800&x-goog-signedheaders=host"

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
