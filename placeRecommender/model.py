import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

class PlaceRecommender():
    def __init__(
        self,
        num_user,
        num_spot,
        user_theme_matrix,
        theme_item_matrix,
        location_item_matrix,
        user_item_saved_matrix,
        weighted_user_item_matrix = np.zeros([1,1], dtype=float),
        initial_taste_weight = 0.8,
        saved_spot_weight = 0.4,
        dict_theme = {},
        dict_loc = {},
        item_theme_map = {}
    ):  
        self.num_user = num_user
        self.num_spot = num_spot
        self.user_theme_matrix = user_theme_matrix
        self.theme_item_matrix = theme_item_matrix
        self.location_item_matrix = location_item_matrix
        self.user_item_saved_matrix = user_item_saved_matrix
        self.weighted_user_item_matrix = weighted_user_item_matrix
        self.initial_taste_weight = initial_taste_weight
        self.saved_spot_weight = saved_spot_weight
        self.dict_theme = dict_theme
        self.dict_loc = dict_loc


    def _gen_taste_weight(self):
        return self.initial_taste_weight * self.user_theme_matrix * self.theme_item_matrix
    
    def _gen_saved_weight(self):  
        return self.saved_spot_weight * self.user_item_saved_matrix


    def gen_weight(
        self,
        user_theme_matrix,
        theme_item_matrix
    ):
        ## TODO:
        ## currently only content based if
        ## later we add _gen_saved_weight if we have saved_spot data
        self.weighted_user_item_matrix = self._gen_taste_weight(user_theme_matrix, theme_item_matrix)


    def _generate_empty_matrix_from_dataset(self):
        self.weighted_user_item_matrix = np.zeros([self.num_user, ], dtype=float)
        return None

    def _matrix_factorization(self):
        return

    def _find_top_N_item(
        self,
        user_id,
        N,
        candidate_spot_mask
    ):
        item_weights = np.multiply(self.weighted_user_item_matrix[user_id], candidate_spot_mask)
        recommending_ranks = item_weights.argsort().argsort()
        return np.where(recommending_ranks > self.num_spot - N)

    def _get_location_related_spot_mask(
        self,
        loc_id
    ):
        #filter with masking
        return self.location_item_matrix[loc_id]

    def recommend_by_loc(
        self,
        user_id,
        per_loc = 1
    ):
        per_loc_recommendations = {}
        for loc in self.dict_loc:
            candidate_spot_mask = self._get_location_related_spot_mask(loc)
            per_loc_recommendations[loc] = self._find_top_N_item(user_id, per_loc, candidate_spot_mask)

        return per_loc_recommendations


    def _get_theme_related_spots_mask(
        self,
        theme
    ):
        #filter
        return self.theme_item_matrix[theme]

    def recommend_by_theme(
        self,
        user_id,
        per_theme = 10
    ):  
        per_theme_recommendations = {}
        for theme in self.dict_theme:
            candidate_spot_mask = self._get_theme_related_spots_mask(theme)
            per_theme_recommendations[theme] = self._find_top_N_item(user_id, per_theme, candidate_spot_mask)

        return per_theme_recommendations