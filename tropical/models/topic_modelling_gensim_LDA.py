from typing import Dict, List, Set, Any, Tuple  # noqa # pylint: disable=unused-import

import pandas as pd
import numpy as np

from gensim.models import Phrases
from gensim.models.phrases import Phraser

import gensim.corpora as corpora

from gensim.models.ldamodel import LdaModel as LDA
from gensim.models import CoherenceModel

from tropical.models.topic_modelling_base import TopicModellingBase

from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from nltk.corpus import stopwords as nltk_stopwords


class TopicModellingGensimLDA(TopicModellingBase):
    """
    Tropical Topic Modelling Gensim - Latent Dirichlet Allocation.

    Parameters:
    ----------
            max_topics (int): The maximum number of topics to try
            min_topics (int): The minimum number of topics to try
            step (int): step size for number of topics

    """

    def __init__(self, max_topics: int = None, min_topics: int = None, step: int = None) -> None:

        super().__init__()
        self.uuid = ""  # Duck typed unique version identifier.
        self.__basic_stopwords = set(nltk_stopwords.words('english')) & spacy_stopwords
        self.__basic_stop_phrases = self.__basic_stopwords

        if max_topics is None:
            self.max_topics = 10
        else:
            self.max_topics = max_topics

        if min_topics is None:
            self.min_topics = 2
        else:
            self.min_topics = min_topics

        if step is None:
            self.step = 2
        else:
            self.step = step

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

    def __remove_stopwords(self, utterances, custom_stopwords=[]):
        """ add custom stop words to default set, then remove all exact matches
        """

        basic_stopwords = set(nltk_stopwords.words('english')) & spacy_stopwords
        custom_stop_words = set(custom_stopwords)
        stops = set(basic_stopwords | custom_stop_words)

        sans_stopwords = [[' '.join([word for word in utterance if word not in stops])] for utterance in utterances]

        return sans_stopwords

    # ToDo : create the regex for this function
    @staticmethod
    def __remove_urls(utterances):
        """Explicitly remove URLS from content"""   # Usually Spam
        pass

    # ToDo : populate functionality, using spacy (for english)
    @staticmethod
    def __lemmatize(utterances):
        """Can be used to allow only certain POS to be used (POS tagger for that language required)"""
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

    def __build_lda_model(self, ngrammed_utterances, num_topics):
        """ gensim implementation of Latent Dirichlet Allocation
        """
        # Create Corpus
        texts = ngrammed_utterances

        # Create Dictionary
        id2word = corpora.Dictionary(ngrammed_utterances)

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # Build LDA model
        lda_model = LDA(corpus=corpus,
                        id2word=id2word,
                        num_topics=num_topics,
                        random_state=42,
                        update_every=8,
                        chunksize=500,
                        passes=100,
                        alpha='auto',
                        per_word_topics=True)

        return lda_model

    def __compute_coherence_values(self, ngrammed_utterances):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """

        coherence_values = []
        model_list = []

        start = self.min_topics
        limit = self.max_topics
        step = self.step
        print(start, limit, step)
        variations = (limit-start)//step
        if variations > 10:
            print(f"Running {variations} variations of the topic model, this might take a few minutes")

        for num_topics in range(start, limit+step, step):
            print(f"Running Model for {num_topics} topics")
            model = self.__build_lda_model(ngrammed_utterances, num_topics)

            dictionary = corpora.Dictionary(ngrammed_utterances)
            texts = ngrammed_utterances

            model_list.append(model)
            coherence_model = CoherenceModel(model=model,
                                             texts=texts,
                                             dictionary=dictionary,
                                             coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())

        return model_list, coherence_values

    def __get_best_model(self, ngrammed_utterances):
        """ Choose model with best coherence from a list of models

        """
        model_list, coherence_values = self.__compute_coherence_values(ngrammed_utterances)

        best_model = model_list[np.argmax(coherence_values)]
        best_model_coherence = max(coherence_values)

        return best_model, best_model_coherence

    def __extract_topics(self, best_model):
        """
        """
        return best_model.show_topics(formatted=False, num_topics=self.max_topics, num_words=10)

    @staticmethod
    def __get_top_utterances(best_model, ngrammed_utterances, uuids, top_n):
        """ return uuids and probabilities for top n utterances per topic

        """
        # Create Dictionary
        id2word = corpora.Dictionary(ngrammed_utterances)

        # Term Document Frequency
        corpus = [id2word.doc2bow(utterance) for utterance in ngrammed_utterances]

        # Init output
        utterance_topics_df = pd.DataFrame()

        # Get main topic in each utterance
        for i, row in enumerate(best_model[corpus]):
            row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            #  Get the Dominant topic, Percent Contribution and Keywords for each utterance
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = best_model.show_topic(topic_num)    # word probability per topic
                    topic_keywords = ", ".join([word for word, prop in wp])    # Don't need this for now
                    utterance_topics_df = utterance_topics_df.append(pd.Series([int(topic_num),
                                                                                round(prop_topic, 4),
                                                                                topic_keywords]),
                                                                     ignore_index=True)
                else:
                    break
        utterance_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add uuid to dataframe
        utterance_topics_df['uuids'] = uuids
        utterance_topics_df.sort_values(['Dominant_Topic', 'Perc_Contribution'], ascending=[True, False], inplace=True)

        # return top_n utterances per topic, with probability
        top_utterances = []
        by_topic = utterance_topics_df.groupby('Dominant_Topic')

        for topic_num, topic_df in by_topic:
            topic_df = topic_df.head(top_n)
            uuids = topic_df['uuids'].tolist()
            scores = topic_df['Perc_Contribution'].tolist()
            uuid_score = zip(uuids, scores)
            top_utterances.append((topic_num, tuple(uuid_score)))

        return top_utterances

    def create_doc_topic_matrix(self, best_model, ngrammed_utterances, uuids):
        """

        """

        # Create Dictionary
        id2word = corpora.Dictionary(ngrammed_utterances)

        # Term Document Frequency
        corpus = [id2word.doc2bow(utterance) for utterance in ngrammed_utterances]

        topic_distribution = best_model.get_document_topics(corpus, minimum_probability=0.0)

        # create each row
        topic_dict = {}
        df_list = []
        original_text = []

        for row in range(0, len(corpus)):
            for topic, weight in topic_distribution[row]:
                topic_dict[topic] = weight
            row_to_add = pd.DataFrame.from_dict(topic_dict, orient='index')
            original_text.append(' '.join([t for t in ngrammed_utterances[row]]))
            df_list.append(row_to_add.T)

        big_df = pd.concat([df for df in df_list])
        big_df.insert(0, 'uuid', uuids)

        new_columns = ['uuid']
        new_columns.extend('Topic ' + str(x) for x in range(0, best_model.num_topics))
        big_df.columns = new_columns

        return big_df

    @staticmethod
    def __format_topic_utterances(best_model, ngrammed_utterances, uuids):
        """ return a dataframe with each utterance and it's topic distribution
        """
        # Create Dictionary
        id2word = corpora.Dictionary(ngrammed_utterances)

        # Term Document Frequency
        corpus = [id2word.doc2bow(utterance) for utterance in ngrammed_utterances]

        # Init output
        utterance_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(best_model[corpus]):
            row = row_list[0] if best_model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = best_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    utterance_topics_df = utterance_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        utterance_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text and UUID to the end of the output
        utterance_topics_df['uuid'] = uuids
        utterance_topics_df['Text'] = ngrammed_utterances

        return utterance_topics_df[['Text', 'uuid', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']]

    def analyse_dataframe(self, big_dataframe_raw: pd.DataFrame,
                          delimiter: bytes = b'_',
                          tag='') -> List[Dict[str, Any]]:
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
            ngrammed_dataframe_utterances = self.__build_ngrams(tokenized_dataframe_utterances,
                                                                threshold=10,  # ToDo: experiment with this threshold.
                                                                stopwords=self.__basic_stopwords,
                                                                delimiter=delimiter)

            # Choosing to always remove stopwords
            ngrammed_dataframe_utterances = self.__remove_stopwords(ngrammed_dataframe_utterances,
                                                                     custom_stopwords=['u', 'm', 'l'])

            best_model, best_model_coherence = self.__get_best_model(ngrammed_dataframe_utterances)


            doc_topic_matrix = self.create_doc_topic_matrix(best_model=best_model,
                                                            ngrammed_utterances=ngrammed_dataframe_utterances,
                                                            uuids=uuids)

            topic_terms = self.__extract_topics(best_model)
            # top_utterances = self.__get_top_utterances(best_model,
            #                                            ngrammed_dataframe_utterances,
            #                                            uuids, 10)

            topic_number = []
            token_ids = []
            token_weights = []
            tokens = []

            min_prob = 5e-4

            for row in topic_terms:
                for tok_id, tok_weight in row[1]:
                    if tok_weight > min_prob:
                        topic_number.append(row[0])
                        tokens.append(tok_id)
                        token_weights.append(tok_weight)
                    else:
                        print(
                            f" Prob of {np.round(tok_weight, 5)} for token : '{tok_id}' is below threshold of {min_prob}")
            topic_terms_df = pd.DataFrame({'topic_num': topic_number, 'token': tokens, 'weights': token_weights})

            topic_terms_float32 = [topic[1] for topic in topic_terms]
            topic_terms_float64 = []

            for i, topics in enumerate(topic_terms_float32):
                topics_float64 = []
                for word, score in topics:
                    topics_float64.append((word, float(score)))
                topic_terms_float64.append({'Topic '+str(i): topics_float64})

            # Convert doc_topic df to dict
            doc_topic_matrix.set_index('uuid', inplace=True)
            doc_topic_matrix_dict = doc_topic_matrix.to_dict(orient='index')

            response_frame: Dict[str, Any] = dict()
            response_frame['time_frame'] = time_frame
            response_frame['num_utterances'] = num_dataframe_utterances
            response_frame['document_topic_matrix'] = doc_topic_matrix_dict

            response_frame['topics'] = topic_terms_float64
            # response_frame['top_utterances_per_topic'] = top_utterances
            response_frame['coherence'] = best_model_coherence

            response_frame_list.append(response_frame)

        return response_frame_list
