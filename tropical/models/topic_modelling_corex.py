from typing import Dict, List, Set, Any, Tuple  # noqa # pylint: disable=unused-import

import pandas as pd
import numpy as np

from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from nltk.corpus import stopwords as nltk_stopwords

from corextopic import corextopic as ct

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tropical.models.topic_modelling_base import TopicModellingBase

class TopicModellingCorex(TopicModellingBase):
    """
    Tropical Topic Modelling - Corex. 
    paper found here : https://transacl.org/ojs/index.php/tacl/article/view/1244

    Parameters:
    ----------
            max_topics (int): The maximum number of topics to try
            min_topics (int): The minimum number of topics to try
            step (int): step size for number of topics

    """

    def __init__(self, n_topics: int = None) -> None:

        super().__init__()
        self.uuid = ""  # Duck typed unique version identifier.
        self.__basic_stopwords = set(nltk_stopwords.words('english')) & spacy_stopwords
        self.__basic_stop_phrases = self.__basic_stopwords

        if n_topics is None:
            self.n_topics = 10
        else:
            self.n_topics = n_topics

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # Pre-processing

    @staticmethod
    def __tokenize_and_lower(utterances):
        """ Whitespace tokenize and lowercase"""
        return [utterance.lower().split() for utterance in utterances]

    def __remove_stopwords(self, utterances, custom_stopwords=None):
        """ add custom stop words to default set, then remove all exact matches
        """

        if custom_stopwords:
            custom_stop_words = set(custom_stopwords)
        else:
            custom_stop_words = set()

        basic_stopwords = set(nltk_stopwords.words('english')) & spacy_stopwords
        stops = set(basic_stopwords | custom_stop_words)

        sans_stopwords = [' '.join([word for word in utterance if word not in stops]) for utterance in utterances]
        
        # TODO : add certain stopwords back, i.e. remove stopwords that may be relevant

        return sans_stopwords

    @staticmethod
    def __remove_urls(utterances):
        """Explicitly remove URLS from content"""   # Usually Spam
        # TODO : create the regex for this function
        
    # ToDo : populate functionality, using spacy (for english)
    @staticmethod
    def __lemmatize(utterances):
        """Can be used to allow only certain POS (POS tagger for that language required)"""

    @staticmethod
    def __get_top_utterances():
        """ return uuids and probabilities for top n utterances per topic"""

    @staticmethod
    def _vectorize_text(data, method='tfidf'):
        """Vectorise Text For Topic Model"""

        if method.lower().replace('-', '') == 'tfidf':
            print('Vectorizing using TF-IDF')
            
            vectorizer = TfidfVectorizer(
                min_df=5,
                max_features=None,
                ngram_range=(1, 2),
                norm=None,
                binary=True,
                use_idf=False,
                sublinear_tf=False)
            
            vectorizer = vectorizer.fit(data['processsed_content'])
            vocab = vectorizer.get_feature_names()
            vectors = vectorizer.transform(data['processsed_content'])
        else:
            print('Vectorizing using Counts')
            
            vectorizer = CountVectorizer(
                min_df=5,
                max_features=None,
                ngram_range=(1, 2),
                binary=True)
            
            vectorizer = vectorizer.fit(data['processsed_content'])
            vocab = vectorizer.get_feature_names()
            vectors = vectorizer.transform(data['processsed_content'])
            
        print(f"vocab size  : {len(vocab)}")
        print(f"vocab head : {vocab[:5]}")
        print(f"vocab tail : {vocab[-5:]}")

        return vectors, vocab

    def train_topic_model(self, data, anchors=None, anchor_strength=3, vectorizer_method='tf-idf'):
 
        vectors, vocab = self._vectorize_text(data, method=vectorizer_method)
        
        if anchors:
            anchors = [[anchor for anchor in topic if anchor in vocab] for topic in anchors]
        
        model = ct.Corex(n_hidden=self.n_topics, max_iter=150, seed=42)
        model = model.fit(vectors, anchors=anchors, anchor_strength=anchor_strength, words=vocab)
        
        return model
    
    def create_doc_topic_matrix(self, model, data, method='default'):
        
        # Todo : Extract vectors from model object instead of recalculating
        vectors, _ = self._vectorize_text(data, 'tf-idf')
        n_topics = self.n_topics
    
        if method == 'default':
            topic_df = pd.DataFrame(
                model.transform(vectors, details=False),
                columns=["topic_{}".format(i+1) for i in range(n_topics)]).astype(int)

            topic_df.index = data.index
            document_topic_matrix = pd.concat([data, topic_df], axis=1)

        return document_topic_matrix

    def _extract_topics(self, model):
        """get top terms per topic """

        return model.get_topics(n_words=10)

        def print_topics(self, model):
            """print topics """

            for i, topic_ngrams in enumerate(model.get_topics(n_words=10)):
                mutual_info = [ngrams[1] for ngrams in topic_ngrams if ngrams[1] > 0]
                topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
                print(f"Topic #{i+1}: {', '.join(t+'('+str(np.round(mi,2))+') ' for t,mi in zip(topic_ngrams, mutual_info))}")
                print()
            return

    def analyse_dataframe(self, big_dataframe_raw: pd.DataFrame,
                          delimiter: bytes = b'_') -> List[Dict[str, Any]]:
        """
        Analyse the messages in the incoming dataframe and extract topics.

        :param big_dataframe_raw: The input dataframe {'time_frame':, 'uuid':, 'content':}
        :param delimiter:
        :return: A dict [{'time_frame':, 'num_utterances':, 'doc_topic_matrix :{{}}','topic_index': ['topic_terms':[]]}]
        """
        big_dataframe_raw = big_dataframe_raw.dropna()
        if big_dataframe_raw.columns[0] == 'day':
            new_columns = list(big_dataframe_raw.columns)
            new_columns[0] = 'time_frame'
            big_dataframe_raw.columns = new_columns

        big_dataframe_raw.insert(len(big_dataframe_raw.columns), 'utterance_length', big_dataframe_raw['content'].str.len())

        utterance_length_threshold = np.percentile(big_dataframe_raw['utterance_length'], 98.0)
        print(f"utterance_length_threshold = {utterance_length_threshold}")
        big_dataframe = big_dataframe_raw[big_dataframe_raw['utterance_length'] <= utterance_length_threshold]

        response_frame_list: List[Dict[str, Any]] = list()
        for time_frame, data_frame in big_dataframe.groupby('time_frame'):
            print("Working on Frame", time_frame)

            dataframe_utterances = list(data_frame['content'])
            num_dataframe_utterances = len(dataframe_utterances)

            uuids = list(data_frame['uuid'])

            tokenized_dataframe_utterances = self.__tokenize_and_lower(dataframe_utterances)
            processsed_content = self.__remove_stopwords(tokenized_dataframe_utterances, custom_stopwords=None)

            data_frame['processsed_content'] = processsed_content
            # print(data_frame.head())

            model = self.train_topic_model(data_frame)
            doc_topic_matrix = self.create_doc_topic_matrix(model, data_frame)
            print(doc_topic_matrix)
            topic_terms = self._extract_topics(model)
            # print(topic_terms)


#             topic_terms = self.__extract_topics()

#             topic_number = []
#             token_ids = []
#             token_weights = []
#             tokens = []

#             min_prob = 5e-4

#             for row in topic_terms:
#                 for tok_id, tok_weight in row[1]:
#                     if tok_weight > min_prob:
#                         topic_number.append(row[0])
#                         tokens.append(tok_id)
#                         token_weights.append(tok_weight)
#                     else:
#                         print(
#                             f" Prob of {np.round(tok_weight, 5)} for token : '{tok_id}' is below threshold of {min_prob}")
#             topic_terms_df = pd.DataFrame({'topic_num': topic_number,
# 'token': tokens, 
# 'weights': token_weights})

#             topic_terms_float32 = [topic[1] for topic in topic_terms]
#             topic_terms_float64 = []

#             for i, topics in enumerate(topic_terms_float32):
#                 topics_float64 = []
#                 for word, score in topics:
#                     topics_float64.append((word, float(score)))
#                 topic_terms_float64.append({'Topic '+str(i): topics_float64})

#             # Convert doc_topic df to dict
            doc_topic_matrix.set_index('uuid', inplace=True)
            doc_topic_matrix_dict = doc_topic_matrix.to_dict(orient='index')

            response_frame: Dict[str, Any] = dict()
            response_frame['time_frame'] = time_frame
            response_frame['num_utterances'] = num_dataframe_utterances
            response_frame['document_topic_matrix'] = doc_topic_matrix_dict

#             response_frame['topics'] = topic_terms_float64
#             # response_frame['top_utterances_per_topic'] = top_utterances

            response_frame_list.append(response_frame)

        return response_frame_list