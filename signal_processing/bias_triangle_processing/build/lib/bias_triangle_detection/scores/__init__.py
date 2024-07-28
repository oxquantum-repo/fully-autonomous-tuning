from enum import Enum
from functools import partial

from .avg_number_of_edges import score_number_of_components, score_number_of_edges
from .base import ScoreFunction, ScoreResult
from .edsr_peak import EntropyScore, prominence_score
from .separation import score_separation


class ScoreType(Enum):
    N_COMPONENTS = partial(score_number_of_components)
    N_EDGES = partial(score_number_of_edges)
    ENTROPY = partial(EntropyScore)
    PROMINENCE = partial(prominence_score)
    SEPARATION = partial(score_separation)
