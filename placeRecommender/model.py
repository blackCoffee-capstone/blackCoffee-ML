import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

class PlaceRecommender():
    def __init__(
        self,
        initial_taste_weight = 0.8,
        saved_spot_weight = 0.4,
        dict_theme = {},
        item_theme_map = {}
    ):
        self.initial_taste_weight = initial_taste_weight
        self.saved_spot_weight = saved_spot_weight
        self.dict_theme = {}

    def _generate_matrix_from_dataset():
        return

    def _matrix_factorization():
        return
    
    def _find_top_N_item(
        self,
        N,
        discarding_item_ids
    ):
        return
    
