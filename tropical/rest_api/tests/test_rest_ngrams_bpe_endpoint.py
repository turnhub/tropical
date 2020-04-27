import unittest  # noqa

import responses
import time
import json

from tackle.rest_api.flask_server.tests import BaseTestCase, send_request, check_response, send_request_check_response
from tropical.rest_api import get_path


# @unittest.skip("skipping during dev")
class TestRestNGramsBPEEndpoint(BaseTestCase):
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
    def test_ngrams_bpe_endpoint_async(self):
        print("Rest HTTP test_ngrams_bpe_endpoint_async:")
        start_time = time.time()

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20.csv?x-" \
                   "goog-signature=7ec8f4ad2861be33c91dd09c07f6b73b9236a432ad2d8dd0d781f66a000798c8a0a82640640cb08cb23342c7bedd6c21e3355" \
                   "36eefdad8557ad70ae849ae4fb082b905c583bdbc6121029c696bc1d4c0986bdeb82862e1312bde3b3c1f7bfd25c2f8c3a3d4bcd586e1876128e" \
                   "f5b6f84bae7794f5a54e1be4b189972a3bd4dd17e23453b8cb88c34df4de2752936f4b1c9d349f40106849b0c29ad33e0bce6e3b3cce3679097b" \
                   "a016cb24e0fd2d879c5cd0bcc85097ef5f3721c4c393344d2b33e7a9e785008e0b457a1cce9f3bcf81cb2fae01f638e277eacac8e0131c7620df" \
                   "b4ff902dc59da4ea7f28981b11d5644e0d6c44ba03ae926f20569c9a66d1a59&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=" \
                   "gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200427%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=202004" \
                   "27T170613Z&x-goog-expires=604800&x-goog-signedheaders=host"

        # Register the async callback sink callback.
        responses.add_callback(responses.POST,
                               'http://api.sink.io/callback',
                               callback=self.callback_sink,
                               content_type='application/json')

        for idx in range(self.__test_ngrams_endpoint_async_num_requests):
            response_check = send_request_check_response(self.client, "/ngrams_bpe_url", "post",
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

    def test_ngrams_bpe_endpoint_sync(self):
        print("Rest HTTP test_ngrams_bpe_endpoint_sync:")
        start_time = time.time()

        file_url = "https://storage.googleapis.com/io-feersum-vectors-nlu-prod/Extract_inbound_that_triggered_catchall_2020_04_20.csv?x-" \
                   "goog-signature=7ec8f4ad2861be33c91dd09c07f6b73b9236a432ad2d8dd0d781f66a000798c8a0a82640640cb08cb23342c7bedd6c21e3355" \
                   "36eefdad8557ad70ae849ae4fb082b905c583bdbc6121029c696bc1d4c0986bdeb82862e1312bde3b3c1f7bfd25c2f8c3a3d4bcd586e1876128e" \
                   "f5b6f84bae7794f5a54e1be4b189972a3bd4dd17e23453b8cb88c34df4de2752936f4b1c9d349f40106849b0c29ad33e0bce6e3b3cce3679097b" \
                   "a016cb24e0fd2d879c5cd0bcc85097ef5f3721c4c393344d2b33e7a9e785008e0b457a1cce9f3bcf81cb2fae01f638e277eacac8e0131c7620df" \
                   "b4ff902dc59da4ea7f28981b11d5644e0d6c44ba03ae926f20569c9a66d1a59&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=" \
                   "gcp-storage%40feersum-221018.iam.gserviceaccount.com%2F20200427%2Fmulti%2Fstorage%2Fgoog4_request&x-goog-date=202004" \
                   "27T170613Z&x-goog-expires=604800&x-goog-signedheaders=host"

        response = send_request(self.client, "/ngrams_bpe_url", "post",
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
