from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Set, Union  # noqa # pylint: disable=unused-import

import pandas as pd


class SemanticClusterAnalysisBase(ABC):
    """
    Tropical Semantic Cluster Analysis: Base class.
    """

    def __init__(self) -> None:
        self.uuid = ""  # Duck typed unique version identifier.

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @abstractmethod
    def analyse_dataframe(self, dataframe: pd.DataFrame) -> List[Dict[str, List]]:
        """
        Analyse the messages in the incoming dataframe for semantic clusters.

        :param dataframe: The input dataframe {'timeframe':, 'uuid':, 'content':}
        :return: A dict [{'timeframe':,  'utterances': ['text':, 'uuid':, 'cluster':, 'vector':]}]
        """
        pass
