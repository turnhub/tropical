import unittest  # noqa

import responses
import time
import json

from tackle.rest_api.flask_server.tests import BaseTestCase, send_request, check_response, send_request_check_response
from tropical.rest_api import get_path


# @unittest.skip("skipping during dev")
class TestRestSemanticClustersEndpoint(BaseTestCase):
    def __init__(self, *args, **kwargs):
        BaseTestCase.__init__(self,
                              *args,
                              specification_dir=get_path() + '',
                              requested_logging_path="~/.tropical/logs",
                              **kwargs)

        self.__test_endpoint_async_num_requests = 3

    def callback_sink(self, request):
        # Note: This will be called in multiple forked processes!
        payload = json.loads(request.body)
        print("CALLBACK SINKED!!!", payload)
        return 200, {}, ""

    # @unittest.skip("skipping during dev")
    @responses.activate
    def test_endpoint_async(self):
        print("Rest HTTP test_endpoint_async:")
        start_time = time.time()

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20." \
                   "csv?x-goog-signature=57da4f3fe51e7345ecbfd3a332aee3e91fa425a9bb3eb358d56283bb7500b9611e3b5bf758e8c5b95a44fe70a57f4" \
                   "df4ae17ce96cf7c9f5401f1d2b89e6dfe1bc2aba8f39ebb2d3c190006f6b75372fc4d74a6bc15093fa746a7faaf4a7e13202b63ba786e6629b" \
                   "7ced41d149deb54e686e4df7cac4094c31017ed15e6415fde7b1a0935069017ac6d18d42689d8d16b410ecb0f1b5e81e3ca4743fb6a6ad1f37" \
                   "abbed51cb4ce045c2a68f50dcf6efa15b05747060a978a8ad6661e578577204e794a86502cf6604e5f38be943d9e9d2b8c62eefc505705e6e9" \
                   "42c8a714a015d6d9aa1054ec64e09aad15319f6a34e7f5959094defa2c594d9e39306176e6875&x-goog-algorithm=GOOG4-RSA-SHA256&x-" \
                   "goog-credential=gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200617%2Fmulti%2Fstorage%2Fgoog4_request&" \
                   "x-goog-date=20200617T030749Z&x-goog-expires=604800&x-goog-signedheaders=host"

        # Register the async callback sink callback.
        responses.add_callback(responses.POST,
                               'http://api.sink.io/callback',
                               callback=self.callback_sink,
                               content_type='application/json')

        for idx in range(self.__test_endpoint_async_num_requests):
            response_check = send_request_check_response(self.client, "/semantic_clusters_url", "post",
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
            print(f"test_endpoint_async: Blindly waiting for callbacks to be received.... {wait_iterations}", flush=True)
            time.sleep(3)
            wait_iterations -= 1

        print("test_endpoint_async: DONE waiting for callbacks to be received.", flush=True)

        print('time = ' + str(time.time() - start_time))

    def test_endpoint_sync(self):
        print("Rest HTTP test_endpoint_sync:")
        start_time = time.time()

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20." \
                   "csv?x-goog-signature=57da4f3fe51e7345ecbfd3a332aee3e91fa425a9bb3eb358d56283bb7500b9611e3b5bf758e8c5b95a44fe70a57f4" \
                   "df4ae17ce96cf7c9f5401f1d2b89e6dfe1bc2aba8f39ebb2d3c190006f6b75372fc4d74a6bc15093fa746a7faaf4a7e13202b63ba786e6629b" \
                   "7ced41d149deb54e686e4df7cac4094c31017ed15e6415fde7b1a0935069017ac6d18d42689d8d16b410ecb0f1b5e81e3ca4743fb6a6ad1f37" \
                   "abbed51cb4ce045c2a68f50dcf6efa15b05747060a978a8ad6661e578577204e794a86502cf6604e5f38be943d9e9d2b8c62eefc505705e6e9" \
                   "42c8a714a015d6d9aa1054ec64e09aad15319f6a34e7f5959094defa2c594d9e39306176e6875&x-goog-algorithm=GOOG4-RSA-SHA256&x-" \
                   "goog-credential=gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200617%2Fmulti%2Fstorage%2Fgoog4_request&" \
                   "x-goog-date=20200617T030749Z&x-goog-expires=604800&x-goog-signedheaders=host"

        response = send_request(self.client, "/semantic_clusters_url", "post",
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
