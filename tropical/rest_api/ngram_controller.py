"""
Flask controller for the ngram model.
"""

from tackle.rest_api.flask_server.controllers import controller_util
from tropical.rest_api import ngram_wrapper


@controller_util.controller_decorator
def start_process_url(user, token_info,
                      request_detail):
    auth_token = controller_util.get_auth_token()
    caller_name = controller_util.get_caller_name()

    file_url = request_detail.get('file_url')
    file_format_version = request_detail.get("file_format_version")
    callback = request_detail.get('callback')

    response_code, response_json = ngram_wrapper.start_process_url(auth_token=auth_token,
                                                                   caller_name=caller_name,
                                                                   file_url=file_url,
                                                                   file_format_version=file_format_version,
                                                                   callback=callback)
    return response_json, response_code


@controller_util.controller_decorator
def start_process_form(user, token_info,
                       **kwargs):
    auth_token = controller_util.get_auth_token()
    caller_name = controller_util.get_caller_name()

    upfile = kwargs.get("upfile")
    file_format_version = kwargs.get("file_format_version")
    callback = kwargs.get("callback")

    response_code, response_json = ngram_wrapper.start_process_form(auth_token=auth_token,
                                                                    caller_name=caller_name,
                                                                    upfile=upfile,
                                                                    file_format_version=file_format_version,
                                                                    callback=callback)
    return response_json, response_code
