from typing import Dict, List, Set, Any, Tuple  # noqa # pylint: disable=unused-import

import pandas as pd
import numpy as np
import random

from gensim.models import Phrases
from gensim.models.phrases import Phraser

import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel as LDA

from tropical.models.topic_modelling_base import TopicModellingBase

from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from nltk.corpus import stopwords as nltk_stopwords


class TopicModellingGensimLDA(TopicModellingBase):
    """
    Tropical Topic Modelling Gensim - Latent Dirichlet Allocation: ....
    """

    def __init__(self, n_topics) -> None:
        super().__init__()
        self.uuid = ""  # Duck typed unique version identifier.
        self.__basic_stopwords = set(nltk_stopwords.words('english')) & spacy_stopwords
        self.__basic_stop_phrases = self.__basic_stopwords
        self.n_topics = n_topics

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # Pre-processing

    @staticmethod
    def __tokenize_and_lower(utterances):
        return [utterance.lower().split() for utterance in utterances]

    @staticmethod
    def __remove_urls(utterances):
        """Explicitly remove URLS from content"""   # Usually Spam
        pass

    @staticmethod
    def __lemmatize(utterances):
        """Can be used to allow only certain POS to be used (POS tagger for that language required)"""
        pass

    def __remove_urls(self):
        pass

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

    def __build_lda_model(self, ngrammed_utterances):
        """ gensim implementation of Latent Dirichlet Allocation
        """
        # Create Dictionary
        id2word = corpora.Dictionary(ngrammed_utterances)

        # Create Corpus
        texts = ngrammed_utterances

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # Build LDA model
        lda_model = LDA(corpus=corpus,
                        id2word=id2word,
                        num_topics=self.n_topics,
                        random_state=42,
                        update_every=8,
                        chunksize=2048,
                        passes=10,
                        alpha='auto',
                        per_word_topics=True)

        return lda_model

    def __extract_topics(self, model):
        """

        """
        return model.show_topics(formatted=False, num_topics=self.n_topics, num_words=10)

    def analyse_dataframe(self, big_dataframe_raw: pd.DataFrame,
                          delimiter: bytes = b'_') -> List[int, Tuple[str, float]]:
        """
        Analyse the messages in the incoming dataframe and extract topics.

        :param big_dataframe_raw: The input dataframe {'time_frame':, 'uuid':, 'content':}
        :param delimiter:
        :return: A dict [{'time_frame':, 'num_utterances':, 'topic_index': ['topic_terms':[]]}]
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

            lda_model = self.__build_lda_model(ngrammed_dataframe_utterances )
            topics = self.__extract_topics(lda_model)
            topic_num = [topic[0] for topic in topics]
            topic_terms = [topic[1] for topic in topics]

            # Dict[phrase, count], Dict[phrase, List[uuid_str]]
            # phrase_count_dict, phrase_uuid_dict = self.__extract_top_ngrams(uuids,
            #                                                                 ngrammed_dataframe_utterances,
            #                                                                 max_count=20,  # max_count of each of uni, bi, tri, etc.
            #                                                                 delimiter=delimiter)

            # ===

            # count_total = sum(phrase_count_dict.values())  # Used to normalise the importance.

            # The phrases is likely already sorted, but below would then be fairly quick to execute.
            # sorted_phrase_count_dict = {phrase: count for phrase, count in
            #                             sorted(phrase_count_dict.items(), key=lambda item: item[1], reverse=True)}

            # print(f"time_frame = {time_frame}, num_phrases = {count_total}: ")

            response_frame: Dict[str, Any] = dict()
            response_frame['time_frame'] = time_frame
            response_frame['num_utterances'] = num_dataframe_utterances
            response_frame['topics'] = [{'topic_number': topic_num,
                                         'topic_terms': topic_terms}]

            # response_frame['top_phrases'] = [{"phrase": phrase,
            #                                   "num_tokens": phrase.count(delimiter.decode(encoding="utf-8")) + 1,
            #                                   "num_utterances": len(phrase_uuid_dict.get(phrase, set())),  # num_phrase_utterances
            #                                   "num_utterances_percentage":
            #                                       len(phrase_uuid_dict.get(phrase, set())) * 100.0 / num_dataframe_utterances,
            #                                   "utterances": random.sample(phrase_uuid_dict.get(phrase, set()),
            #                                                               min(len(phrase_uuid_dict.get(phrase, set())), 20))}  # ToDo: 20.
            #                                  for phrase, count in sorted_phrase_count_dict.items()]

            response_frame_list.append(response_frame)

        return response_frame_list
