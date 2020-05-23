import unittest  # noqa

import responses
import time
import json

from tackle.rest_api.flask_server.tests import BaseTestCase, send_request, check_response, send_request_check_response
from tropical.rest_api import get_path


# @unittest.skip("skipping during dev")
class TestRestNGramsEndpoint(BaseTestCase):
    def __init__(self, *args, **kwargs):
        BaseTestCase.__init__(self,
                              *args,
                              specification_dir=get_path() + '',
                              requested_logging_path="~/.tropical/logs",
                              **kwargs)

        self.__test_ngrams_endpoint_async_num_requests = 3

    def callback_sink(self, request):
        # Note: This will be called in multiple forked processes!
        payload = json.loads(request.body)
        print("CALLBACK SINKED!!!", payload)
        return 200, {}, ""

    # @unittest.skip("skipping during dev")
    @responses.activate
    def test_ngrams_endpoint_async(self):
        print("Rest HTTP test_ngrams_endpoint_async:")
        start_time = time.time()

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20.csv?x-" \
                   "goog-signature=a4cdb1d4f809174ffbb1954f210cbfa2586c6d52783a328d46e95dbacd1fdf6c2f33660fdc603df9a8599357dd341c990eb10" \
                   "f918561e702962f4f79d673436d07748f000e9bfbfbc0887e207995c7b8b5a8fb5b49dc01a3d5de5418fbe729e92ce76de4d35d7e736c924b411" \
                   "7641b0a6a554e8bbb2c715eef4b86a9c669bdff7866c374197aa8d579e976116ee637012f06e27df16b9dc26d40ae3249593abf661d7224d4397" \
                   "d1d0133b00f635cb434c38c0b3dd199344c434bcb5efc8a7c7f9175e855fd0c16d0121c4cbf6b46889f86528460e05fae232848a3d2e598d19b7" \
                   "0bb9f3a4bae05a2995618806c9e128c1cab4fb7e52cb016673814074584a16e&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=" \
                   "gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200523%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=202005" \
                   "23T124251Z&x-goog-expires=604800&x-goog-signedheaders=host"

        # Register the async callback sink callback.
        responses.add_callback(responses.POST,
                               'http://api.sink.io/callback',
                               callback=self.callback_sink,
                               content_type='application/json')

        for idx in range(self.__test_ngrams_endpoint_async_num_requests):
            response_check = send_request_check_response(self.client, "/ngrams_url", "post",
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
            print(f"test_ngrams_endpoint_async: Blindly waiting for callbacks to be received.... {wait_iterations}", flush=True)
            time.sleep(3)
            wait_iterations -= 1

        print("test_ngrams_endpoint_async: DONE waiting for callbacks to be received.", flush=True)

        print('time = ' + str(time.time() - start_time))

    def test_ngrams_endpoint_sync(self):
        print("Rest HTTP test_ngrams_endpoint_sync:")
        start_time = time.time()

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20.csv?x-" \
                   "goog-signature=a4cdb1d4f809174ffbb1954f210cbfa2586c6d52783a328d46e95dbacd1fdf6c2f33660fdc603df9a8599357dd341c990eb10" \
                   "f918561e702962f4f79d673436d07748f000e9bfbfbc0887e207995c7b8b5a8fb5b49dc01a3d5de5418fbe729e92ce76de4d35d7e736c924b411" \
                   "7641b0a6a554e8bbb2c715eef4b86a9c669bdff7866c374197aa8d579e976116ee637012f06e27df16b9dc26d40ae3249593abf661d7224d4397" \
                   "d1d0133b00f635cb434c38c0b3dd199344c434bcb5efc8a7c7f9175e855fd0c16d0121c4cbf6b46889f86528460e05fae232848a3d2e598d19b7" \
                   "0bb9f3a4bae05a2995618806c9e128c1cab4fb7e52cb016673814074584a16e&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=" \
                   "gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200523%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=202005" \
                   "23T124251Z&x-goog-expires=604800&x-goog-signedheaders=host"

        response = send_request(self.client, "/ngrams_url", "post",
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
