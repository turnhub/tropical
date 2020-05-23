from typing import Dict, List, Set, Any  # noqa # pylint: disable=unused-import

import pandas as pd
import numpy as np
import random

from gensim.models import Phrases
from gensim.models.phrases import Phraser

from tropical.models.ngram_analysis_base import NGramAnalysisBase

from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from nltk.corpus import stopwords as nltk_stopwords


class NGramAnalysisGensim(NGramAnalysisBase):
    """
    Tropical Ngram Analysis: ....
    """

    def __init__(self) -> None:
        super().__init__()
        self.uuid = ""  # Duck typed unique version identifier.
        self.__basic_stopwords = set(nltk_stopwords.words('english')) & spacy_stopwords
        self.__basic_stop_phrases = self.__basic_stopwords

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def __tokenize_and_lower(utterances):
        return [utterance.lower().split() for utterance in utterances]

    def __build_ngrams(self,
                       tokenised_utterances,
                       stopwords,
                       min_count=2,
                       threshold=10,
                       delimiter: bytes = b'_'):

        bigram = Phrases(
            tokenised_utterances,
            min_count=min_count,
            threshold=threshold,
            # max_vocab_size=40000000,
            delimiter=delimiter,
            common_terms=stopwords
        )

        ngram = Phrases(
            bigram[tokenised_utterances],
            min_count=min_count,
            threshold=50,  # ToDo: experiment with this threshold.
            # max_vocab_size=40000000,
            delimiter=delimiter,
            common_terms=stopwords
        )

        # Export the trained model = use less RAM, faster processing, but model updates no longer possible.
        bigram_phraser = Phraser(bigram)
        ngram_phraser = Phraser(ngram)

        # bigrammed_utterances = [bigram_phraser[utterance] for utterance in tokenised_utterances]
        ngrammed_utterances = [ngram_phraser[bigram_phraser[utterance]] for utterance in tokenised_utterances]

        return ngrammed_utterances

    def __extract_top_ngrams(self,
                             uuids,
                             ngrammed_utterances,
                             max_count=10,
                             delimiter: bytes = b'_'):
        """ Extract top n ngrams. """
        # ===
        phrase_count_dict: Dict[str, int] = dict()  # {phrase: count}
        phrase_uuid_dict: Dict[str, Set[str]] = dict()  # phrase: [uuids]

        for idx, ngrammed_utterance in enumerate(ngrammed_utterances):
            utterance_uuid = uuids[idx]

            # Iterate over each phrase in the ngrammed/processed utterance and keep count of each unique phrase.
            for phrase in ngrammed_utterance:
                if phrase not in self.__basic_stop_phrases:
                    count = phrase_count_dict.get(phrase, 0)
                    uuid_set = phrase_uuid_dict.get(phrase, set())

                    count += 1
                    uuid_set.add(utterance_uuid)

                    phrase_count_dict[phrase] = count
                    phrase_uuid_dict[phrase] = uuid_set

        # === Sort the phrases by count.
        sorted_phrase_count_dict = {phrase: count for phrase, count in
                                    sorted(phrase_count_dict.items(), key=lambda item: item[1], reverse=True)}

        # === Iterate over the sorted phrases/n-grams and keep at most max_count of each size of n-gram.
        ngram_count_dict: Dict[int, int] = dict()
        top_phrase_count_dict: Dict[str, int] = dict()  # {phrase: count}

        for phrase, count in sorted_phrase_count_dict.items():
            n = phrase.count(delimiter.decode(encoding="utf-8")) + 1
            ngram_count = ngram_count_dict.get(n, 0)
            if ngram_count < max_count:
                # Can still add more of this length of ngram so add this phrase to the the list of top phrases.
                top_phrase_count_dict[phrase] = count
            ngram_count_dict[n] = ngram_count + 1  # Update the count.

        return top_phrase_count_dict, phrase_uuid_dict  # Dict[phrase, count], Dict[phrase, List[uuid_str]]

    def analyse_dataframe(self, big_dataframe_raw: pd.DataFrame,
                          delimiter: bytes = b'_') -> List[Dict[str, Any]]:
        """
        Analyse the messages in the incoming dataframe for common ngrams.

        :param big_dataframe_raw: The input dataframe {'time_frame':, 'uuid':, 'content':}
        :param delimiter:
        :return: A dict [{'time_frame':, 'num_utterances':, 'top_phrases': ['phrase':, 'importance':, 'utterances':[]]}]
        """

        # Remove any rows with NaNs and missing values.
        big_dataframe_raw = big_dataframe_raw.dropna()

        if big_dataframe_raw.columns[0] == 'day':
            new_columns = list(big_dataframe_raw.columns)
            new_columns[0] = 'time_frame'
            big_dataframe_raw.columns = new_columns

        # big_dataframe_raw['time_frame'] = 'one_time_frame_to_rule_them_all'

        big_dataframe_raw.insert(len(big_dataframe_raw.columns), 'utterance_length', big_dataframe_raw['content'].str.len())

        utterance_length_threshold = np.percentile(big_dataframe_raw['utterance_length'], 98.0)
        print(f"utterance_length_threshold = {utterance_length_threshold}")
        big_dataframe = big_dataframe_raw[big_dataframe_raw['utterance_length'] <= utterance_length_threshold]

        # print(dataframe.columns)
        # print(dataframe.describe())
        # print(dataframe.sample(n=5))

        response_frame_list: List[Dict[str, Any]] = list()

        for time_frame, data_frame in big_dataframe.groupby('time_frame'):
            dataframe_utterances = list(data_frame['content'])
            num_dataframe_utterances = len(dataframe_utterances)

            uuids = list(data_frame['uuid'])

            tokenized_dataframe_utterances = self.__tokenize_and_lower(dataframe_utterances)
            ngrammed_dataframe_utterances = self.__build_ngrams(tokenized_dataframe_utterances,
                                                                threshold=10,  # ToDo: experiment with this threshold.
                                                                stopwords=self.__basic_stopwords,
                                                                delimiter=delimiter)

            # Dict[phrase, count], Dict[phrase, List[uuid_str]]
            phrase_count_dict, phrase_uuid_dict = self.__extract_top_ngrams(uuids,
                                                                            ngrammed_dataframe_utterances,
                                                                            max_count=20,  # max_count of each of uni, bi, tri, etc.
                                                                            delimiter=delimiter)

            # ===

            count_total = sum(phrase_count_dict.values())  # Used to normalise the importance.

            # The phrases is likely already sorted, but below would then be fairly quick to execute.
            sorted_phrase_count_dict = {phrase: count for phrase, count in
                                        sorted(phrase_count_dict.items(), key=lambda item: item[1], reverse=True)}

            print(f"time_frame = {time_frame}, num_phrases = {count_total}: ")

            response_frame: Dict[str, Any] = dict()
            response_frame['time_frame'] = time_frame
            response_frame['num_utterances'] = num_dataframe_utterances
            response_frame['top_phrases'] = [{"phrase": phrase,
                                              "num_tokens": phrase.count(delimiter.decode(encoding="utf-8")) + 1,
                                              "importance": (count / count_total),
                                              "num_utterances": len(phrase_uuid_dict.get(phrase, set())),  # num_phrase_utterances
                                              "num_utterances_percentage":
                                                  len(phrase_uuid_dict.get(phrase, set())) * 100.0 / num_dataframe_utterances,
                                              "utterances": random.sample(phrase_uuid_dict.get(phrase, set()),
                                                                          min(len(phrase_uuid_dict.get(phrase, set())), 20))}  # ToDo: 20.
                                             for phrase, count in sorted_phrase_count_dict.items()]

            response_frame_list.append(response_frame)

        return response_frame_list
