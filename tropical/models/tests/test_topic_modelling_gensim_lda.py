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

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_" \
                   "catchall_2020_04_20.csv?x-goog-signature=966cbb21e9eac23ea7811b94a7b6c050c97901f204acdfc1c99b49567810" \
                   "ce8df1242e4953f886877da375d6968e8f5e7ce7d5ffc74bacc8cb057998e3346cf87f34d5109e6b26884cd4f268fb51d053609" \
                   "1fd0eeb65e3979081ebb306c3f369d7bf90989e4f75bc29849c61658b5a61144720364550d9b63e4686bf61885603cdc0c25af0ef" \
                   "afb50634e8c97aba77e66b9cd6403564907fd1e76b403d683c97561b6ad81323f55b0fe212b147381d2d5210b6df0c8cdc74373e71b" \
                   "c6f27d68dd788c71b4a370452ec5cc32a6bb6c27403a6958adc4310ac6c696fa5badf70d6d54f62e7e82190524bba61ecea2167421bf" \
                   "fa3b6a914d64f19b877d7fd654099&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=gcp-storage%40feersum-2210" \
                   "18.iam.gserviceaccount.com%2F20200618%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=20200618T081124Z&x-goog" \
                   "-expires=604800&x-goog-signedheaders=host"

        # ===
        filename = wget.download(file_url)
        print("filename =", filename)
        df = pd.read_csv(filename, na_filter=False)
        os.remove(filename)

        print(df.columns)
        print(df.describe())
        print(df.sample(n=5))
        # ===

        analyser = topic_modelling_gensim_LDA.TopicModellingGensimLDA()
        self.assertTrue(analyser is not None)
        result = analyser.analyse_dataframe(df, delimiter=b'|')

        # ToDo: Add a reference static test file and add some specific result tests here.

        print(json.dumps(result, indent=4, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()