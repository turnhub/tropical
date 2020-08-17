import unittest  # noqa

import responses
import time
import json

from tackle.rest_api.flask_server.tests import BaseTestCase, send_request, check_response, send_request_check_response
from tropical.rest_api import get_path


# @unittest.skip("skipping during dev")
class TestRestTopicsLDAGensimEndpoint(BaseTestCase):
    def __init__(self, *args, **kwargs):
        BaseTestCase.__init__(self,
                              *args,
                              specification_dir=get_path() + '',
                              requested_logging_path="~/.tropical/logs",
                              **kwargs)

        self.__test_topics_endpoint_async_num_requests = 3

    def callback_sink(self, request):
        # Note: This will be called in multiple forked processes!
        payload = json.loads(request.body)
        print("CALLBACK SINKED!!!", payload)
        return 200, {}, ""

    # @unittest.skip("skipping during dev")
    @responses.activate
    def test_topics_lda_gensim_endpoint_async(self):
        print("Rest HTTP test_topics_lda_gensim_endpoint_async:")
        start_time = time.time()

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_"\
        "catchall_2020_04_20.csv?x-goog-signature=966cbb21e9eac23ea7811b94a7b6c050c97901f204acdfc1c99b49567810"\
        "ce8df1242e4953f886877da375d6968e8f5e7ce7d5ffc74bacc8cb057998e3346cf87f34d5109e6b26884cd4f268fb51d053609"\
        "1fd0eeb65e3979081ebb306c3f369d7bf90989e4f75bc29849c61658b5a61144720364550d9b63e4686bf61885603cdc0c25af0ef"\
        "afb50634e8c97aba77e66b9cd6403564907fd1e76b403d683c97561b6ad81323f55b0fe212b147381d2d5210b6df0c8cdc74373e71b"\
        "c6f27d68dd788c71b4a370452ec5cc32a6bb6c27403a6958adc4310ac6c696fa5badf70d6d54f62e7e82190524bba61ecea2167421bf"\
        "fa3b6a914d64f19b877d7fd654099&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=gcp-storage%40feersum-2210"\
        "18.iam.gserviceaccount.com%2F20200618%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=20200618T081124Z&x-goog"\
        "-expires=604800&x-goog-signedheaders=host"

        # Register the async callback sink callback.
        responses.add_callback(responses.POST,
                               'http://api.sink.io/callback',
                               callback=self.callback_sink,
                               content_type='application/json')

        for idx in range(self.__test_topics_endpoint_async_num_requests):
            response_check = send_request_check_response(self.client, "/topic_model_gensim_LDA_url", "post",
                                                         {"file_url": file_url,
                                                          "file_format_version": "1.0",
                                                          "callback": "http://api.sink.io/callback"},
                                                         202,
                                                         {'file_format_version': '1.0',
                                                          'callback': 'http://api.sink.io/callback',
                                                          'response_frames': None},
                                                         treat_list_as_set=True)

            self.assertTrue(response_check)

        wait_iterations = 10
        while wait_iterations > 0:
            print(f"test_topics_gensim_lda_endpoint_async: Blindly waiting for callbacks to be received.... {wait_iterations}", flush=True)
            time.sleep(3)
            wait_iterations -= 1

        print("test_topics_gensim_lda_endpoint_async: DONE waiting for callbacks to be received.", flush=True)

        print('time = ' + str(time.time() - start_time))

    # @unittest.skip("skipping during dev")
    def test_topics_gensim_lda_endpoint_sync(self):
        print("Rest HTTP test_topics_gensim_lda_endpoint_sync:")
        start_time = time.time()

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_" \
                   "2020_04_20.csv?x-goog-signature=966cbb21e9eac23ea7811b94a7b6c050c97901f204acdfc1c99b49567810ce8df1242e4953f8868" \
                   "77da375d6968e8f5e7ce7d5ffc74bacc8cb057998e3346cf87f34d5109e6b26884cd4f268fb51d0536091fd0eeb65e3979081ebb306c3f3" \
                   "69d7bf90989e4f75bc29849c61658b5a61144720364550d9b63e4686bf61885603cdc0c25af0efafb50634e8c97aba77e66b9cd640356490" \
                   "7fd1e76b403d683c97561b6ad81323f55b0fe212b147381d2d5210b6df0c8cdc74373e71bc6f27d68dd788c71b4a370452ec5cc32a6bb6c" \
                   "27403a6958adc4310ac6c696fa5badf70d6d54f62e7e82190524bba61ecea2167421bffa3b6a914d64f19b877d7fd654099&x-goog-algor" \
                   "ithm=GOOG4-RSA-SHA256&x-goog-credential=gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200618%2Fmult" \
                   "i%2Fstorage%2Fgoog4_request&x-goog-date=20200618T081124Z&x-goog-expires=604800&x-goog-signedheaders=host"

        response = send_request(self.client, "/topic_model_gensim_LDA_url", "post",
                                {"file_url": file_url,
                                 "file_format_version": "1.0"})
        result = response.json.get('response_frames')
        print(json.dumps(result, indent=4, sort_keys=True, ensure_ascii=False))

        # ToDo: Add a reference static test file and add some specific result tests here.

        response_check = check_response(response,
                                        200,
                                        {'file_format_version': '1.0',
                                         "callback": None},
                                        treat_list_as_set=True)

        self.assertTrue(response_check)

        print('time = ' + str(time.time() - start_time))
