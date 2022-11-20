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
        dict_theme = {'산'     : 0,
                       '럭셔리' : 1,
                       '역사'   : 2,
                       '웰빙'   : 3,
                       '바다'   : 4,
                       '카페'   : 5,
                       '공원'   : 6,
                       '전시장' : 7,
                       '건축'   : 8,
                       '사찰'   : 9,
                       '가족'   : 10,
                       '학교'   : 11,
                       '놀이공원':12,
                       '겨울'   : 13,
                       '엑티비티':14,
                       '캠핑'   :15,
                       '섬'     :16,
                       '커플'   :17,
                       '저수지' :18,
                       '폭포'   :19
                       }, 
        dict_loc = {'서울특별시':0,
                    '부산광역시':1,
                    '대구광역시':2,
                    '인천광역시':3,
                    '광주광역시':4,
                    '대전광역시':5,
                    '울산광역시':6,
                    '세종특별자치시':7,
                    '경기도':8,
                    '강원도':9,
                    '충청북도':10,
                    '전라북도':11,
                    '전라남도':12,
                    '경상북도':13,
                    '경상남도':14,
                    '제주특별자치도':15,
                    'Uknown':16
        },
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
        return np.matmul(self.initial_taste_weight * self.user_theme_matrix, self.theme_item_matrix)
    
    def _gen_saved_weight(self):  
        return self.saved_spot_weight * self.user_item_saved_matrix


    def gen_weight(self):
        ## TODO:
        ## currently only content based if
        ## later we add _gen_saved_weight if we have saved_spot data
        self.weighted_user_item_matrix = self._gen_taste_weight()


    def _generate_empty_matrix_from_dataset(self):
        self.weighted_user_item_matrix = np.zeros([self.num_spot, self.num_user], dtype=float)
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
        for _, loc in self.dict_loc.items():
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
        for _, theme in self.dict_theme.items():
            candidate_spot_mask = self._get_theme_related_spots_mask(theme)
            per_theme_recommendations[theme] = self._find_top_N_item(user_id, per_theme, candidate_spot_mask)

        return per_theme_recommendations