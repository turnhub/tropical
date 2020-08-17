"""
Service wrapper for the topic model - Gensim implementation of LDA.
"""

from typing import Tuple, Optional  # noqa # pylint: disable=unused-import
import logging
import os
import uuid
import requests
import json

import pandas as pd
from multiprocessing import Process

from werkzeug.datastructures import FileStorage
import wget

from tackle.rest_api import wrapper_util
from tropical.models import topic_modelling_gensim_LDA


def __topic_gensim_lda_task(df: pd.DataFrame, file_format_version: str):
    analyser = topic_modelling_gensim_LDA.TopicModellingGensimLDA()
    response_frames = analyser.analyse_dataframe(df, delimiter=b' ')
    return response_frames


def __topic_gensim_lda_task_and_callback(df: pd.DataFrame, file_format_version: str, task_uuid: str, callback: str):
    print(f"Process FORKED!!! Now in process with ID={os.getpid()}")
    response_frames = __topic_gensim_lda_task(df, file_format_version)
    payload = {
        "uuid": task_uuid,
        "file_format_version": file_format_version,
        "callback": callback,
        "response_frames": response_frames
    }
    try:
        response = requests.post(
            url=callback,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            timeout=3.0,
            headers={'content-type': 'application/json'}
        )
        response_status_code, response_text = response.status_code, response.text
    except requests.Timeout as e:
        logging.error(f"topic_gensim_lda_wrapper.__topic_gensim_lda_task_and_callback: requests.Timeout {e}!")
        response_status_code, response_text = 400, json.dumps({"error": f"Callback request timeout ({e})."})
    except requests.RequestException as e:
        logging.error(f"topic_gensim_lda_wrapper.__topic_gensim_lda_task_and_callback: requests.RequestException {e}!")
        response_status_code, response_text = 500, json.dumps({"error": f"Callback request exception ({e})."})

    logging.info(f"topic_gensim_lda_wrapper.__topic_gensim_lda_task_and_callback: Completed {task_uuid} with callback={callback}.")
    return response_status_code, response_text


def __start_process(task_filename: str,
                    file_format_version: str,
                    callback: Optional[str]) -> Tuple[int, wrapper_util.JSONType]:
    task_uuid = str(uuid.uuid4())
    logging.info(f"topic_gensim_lda_wrapper.start_process: Start {task_uuid} with callback={callback}.")

    # ToDo: catch read_csv and other exceptions.
    df = pd.read_csv(task_filename, na_filter=False)

    if callback is None:
        # Just do it synchronously.
        response_frames = __topic_gensim_lda_task(df, file_format_version)
        logging.info(f"topic_gensim_lda_wrapper.__start_process: Completed {task_uuid} with callback={callback}.")
        response_status = 200
    else:
        # Start an async task ...
        print(f"Process WILL BE FORKED! Now in process with ID={os.getpid()}")
        p = Process(target=__topic_gensim_lda_task_and_callback, args=(df, file_format_version, task_uuid, callback))
        p.start()
        # p.join()  # Fire and forget for now. Could in future keep track of the processes via the API.
        print(f"Process JOINED!!! Now in process with ID={os.getpid()}")

        response_frames = None  # Response will be sent later to callback URL.
        response_status = 202

    return response_status, {
        "uuid": task_uuid,
        "file_format_version": file_format_version,
        "callback": callback,
        "response_frames": response_frames
    }


@wrapper_util.lock_decorator
@wrapper_util.auth_decorator
def start_process_url(auth_token: str, caller_name: Optional[str],
                      file_url: str,
                      file_format_version: str,
                      callback: Optional[str]) -> Tuple[int, wrapper_util.JSONType]:
    filename = wget.detect_filename(file_url)

    if filename.endswith(".csv"):
        task_filename = str(uuid.uuid4()) + "_" + filename
        task_filename = wget.download(file_url, out=task_filename)
        response = __start_process(task_filename, file_format_version, callback)
        os.remove(task_filename)
        return response
    else:
        return 400, {"error": "Please provide a .csv file."}


@wrapper_util.lock_decorator
@wrapper_util.auth_decorator
def start_process_form(auth_token: str, caller_name: Optional[str],
                       upfile: FileStorage,
                       file_format_version: str,
                       callback: Optional[str]) -> Tuple[int, wrapper_util.JSONType]:
    filename = str(upfile.filename)

    if filename.endswith(".csv"):
        task_filename = str(uuid.uuid4()) + "_" + filename
        upfile.save(task_filename)
        upfile.close()
        response = __start_process(task_filename, file_format_version, callback)
        os.remove(task_filename)
        return response
    else:
        return 400, {"error": "Please upload a .csv file."}
