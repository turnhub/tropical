from typing import Dict, List, Set, Any, Tuple  # noqa # pylint: disable=unused-import

import pandas as pd
import numpy as np

from tropical.models.semantic_cluster_analysis_base import SemanticClusterAnalysisBase

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


class SemanticClusterAnalysis(SemanticClusterAnalysisBase):
    """
    Tropical Semantic Cluster Analysis: ....
    """

    def __init__(self) -> None:
        super().__init__()
        self.uuid = ""  # Duck typed unique version identifier.

        self._encoder = SentenceTransformer('average_word_embeddings_glove.6B.300d')

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __get_vect(self, text: str) -> np.array:
        sent_vect_list = self._encoder.encode([text],
                                              show_progress_bar=False,
                                              convert_to_numpy=True)
        return sent_vect_list[0]

    def __cluster(self,
                  uuids: List[str],
                  utterances: List[str]) -> Tuple[int, List[Tuple[str, str, str, np.array]]]:
        """ Extract clusters. Returns [(text, uuid, cluster, vector)]. """
        # ===
        semantic_clustering = []  # type: List[Tuple[str, str, str, np.array]]
        vectors = []  # type; List[np.array]

        for utterance in utterances:
            vectors.append(self.__get_vect(utterance))

        clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=30.0).fit(vectors)

        # print(clustering_model.labels_)
        num_clusters = clustering.n_clusters_
        # print(f"n_clusters={n_clusters}")

        for i in range(len(utterances)):
            semantic_clustering.append((utterances[i], uuids[i], str(clustering.labels_[i]), vectors[i]))

        return int(num_clusters), semantic_clustering

    def analyse_dataframe(self, big_dataframe_raw: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyse the messages in the incoming dataframe for semantic clusters.

        :param big_dataframe_raw: The input dataframe {'time_frame':, 'uuid':, 'content':}
        :return: A dict [{'timeframe':,  'utterances': ['text':, 'uuid':, 'cluster':, 'vector':]}]
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
            utterances = list(data_frame['content'])
            uuids = list(data_frame['uuid'])

            semantic_clustering: List[Tuple[str, str, str, np.array]]
            num_clusters, semantic_clustering = self.__cluster(uuids=uuids,
                                                               utterances=utterances)

            print(f"time_frame = {time_frame}, utterances = {len(utterances)}: ")

            response_frame: Dict[str, Any] = dict()
            response_frame['time_frame'] = time_frame
            response_frame['num_clusters'] = num_clusters
            response_frame['utterances'] = [{"text": text,
                                             "uuid": uuid,
                                             "cluster": cluster,
                                             "vector": vector.tolist()}
                                            for text, uuid, cluster, vector in semantic_clustering]

            response_frame_list.append(response_frame)

        return response_frame_list
