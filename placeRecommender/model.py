import numpy as np
import torch
import json
import sys
import os
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
                       '폭포'   :19,
                       'Uknown' :20
                       }, 
        dict_loc = {'서울':0,
                    '부산':1,
                    '대구':2,
                    '인천':3,
                    '광주':4,
                    '대전':5,
                    '울산':6,
                    '세종':7,
                    '경기도':8,
                    '강원도':9,
                    '충청도':10,
                    '전라도':11,
                    '경상도':12,
                    '제주':13,
                    'Uknown':14
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
        if candidate_spot_mask is None:
            item_weights = self.weighted_user_item_matrix[user_id]
        else :
            item_weights = np.multiply(self.weighted_user_item_matrix[user_id], candidate_spot_mask)
        
        recommending_ranks = item_weights.argsort().argsort()
        print(recommending_ranks)
        print(np.where(recommending_ranks > self.num_spot - N - 1))
        ##TODO
        ## WE are only giving back the top N elements not ordered top N elements

        return np.where(recommending_ranks > self.num_spot - N - 1)

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
            #print(candidate_spot_mask)
            per_loc_recommendations[loc] = self._find_top_N_item(user_id, per_loc, candidate_spot_mask)
            #print(per_loc_recommendations[loc])

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
            #print(candidate_spot_mask)
            per_theme_recommendations[theme] = self._find_top_N_item(user_id, per_theme, candidate_spot_mask)
            #print(per_theme_recommendations[theme])

        return per_theme_recommendations

    def recommend(
        self,
        user_id,
        topN = 10
    ) :
        return self._find_top_N_item(user_id, topN, None)

class HybridRecSystem(torch.nn.Module):
    def __init__(
        self,
        number_of_user = 7500,
        number_of_item = 7500,
        user_feature_size = 21,
        item_feature_size = 39,
        embedding_size = 50,
        n_hidden = 20
    ):  
        super(HybridRecSystem, self).__init__()
        self.number_of_user    = number_of_user
        self.number_of_item    = number_of_item
        self.user_feature_size = user_feature_size
        self.item_feature_size = item_feature_size
        self.embedding_size    = embedding_size
        self.n_hidden          = n_hidden

        self.user_embedding    = torch.nn.Embedding(number_of_user + 1, embedding_size)
        self.item_embedding    = torch.nn.Embedding(number_of_item + 1, embedding_size)
        
        self.user_feature_layer= torch.nn.Linear(in_features = embedding_size + user_feature_size, out_features= embedding_size)
        self.item_feature_layer= torch.nn.Linear(in_features = embedding_size + item_feature_size, out_features= embedding_size)

        self.relu = torch.nn.ReLU()

        self.layer1 = torch.nn.Linear(embedding_size*2, n_hidden)
        self.drop1  = torch.nn.Dropout(0,1)

        self.layer2 = torch.nn.Linear(n_hidden, 1)
        

    def forward(
        self,
        user_id,
        item_id,
        user_features,
        item_features
    ) :

        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        
        user_hidden_feature = self.relu(torch.cat([user_embedding, user_features], dim = 1))
        user_hidden_feature = self.user_feature_layer(user_hidden_feature)
        user_hidden_feature = user_hidden_feature + user_embedding
 
        item_hidden_feature = self.relu(torch.cat([item_embedding, item_features], dim = 1))
        item_hidden_feature = self.item_feature_layer(item_hidden_feature)
        item_hidden_feature = item_hidden_feature + item_embedding

        hidden_feature = self.relu(torch.cat([user_hidden_feature, item_hidden_feature], dim = 1))
        hidden_feature = self.layer1(hidden_feature)
        hidden_feature = self.drop1(hidden_feature)
        
        hidden_feature = self.relu(hidden_feature)
        hidden_feature = self.layer2(hidden_feature)

        ## Final hidden_feature is the predicted ratings that user will make on item
        ## in Actual Recommendation, use forward with candidate items
        ## Select items has the higest predicted ratings
        return hidden_feature

    def _get_model_config(self):
        config = {
            'number_of_user' : self.number_of_user,
            'number_of_item' : self.number_of_item,
            'user_feature_size' : self.user_feature_size,
            'item_feature_size' : self.item_feature_size,
            'embedding_size'    : self.embedding_size,
            'n_hidden'   : self.n_hidden
        }
        
        return config

    def _make_model_from_config(
        self,
        config
    ):  
        number_of_user    = config["number_of_user"]
        number_of_item    = config["number_of_item"]
        user_feature_size = config["user_feature_size"]
        item_feature_size = config["item_feature_size"]
        embedding_size    = config["embedding_size"]
        n_hidden          = config["n_hidden"]

        self.number_of_user    = number_of_user
        self.number_of_item    = number_of_item
        self.user_feature_size = user_feature_size
        self.item_feature_size = item_feature_size
        self.embedding_size    = embedding_size
        self.n_hidden          = n_hidden

        self.user_embedding    = torch.nn.Embedding(number_of_user + 1, embedding_size)
        self.item_embedding    = torch.nn.Embedding(number_of_item + 1, embedding_size)
        
        self.user_feature_layer= torch.nn.Linear(in_features = embedding_size + user_feature_size, out_features= embedding_size)
        self.item_feature_layer= torch.nn.Linear(in_features = embedding_size + item_feature_size, out_features= embedding_size)

        self.relu = torch.nn.ReLU()

        self.layer1 = torch.nn.Linear(embedding_size*2, n_hidden)
        self.drop1  = torch.nn.Dropout(0,1)

        self.layer2 = torch.nn.Linear(n_hidden, 1)

    def save_trained(
        self,
        save_path
    ):
        config = self._get_model_config()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path+'/config.json', 'w') as fp:
            json.dump(config, fp)
        torch.save(self.state_dict(), save_path+"/weight.pt")


    def load_trained(
        self,
        load_path
    ):
        config_file = open(load_path+'/config.json')
        config = json.load(config_file)
        self._make_model_from_config(config)
        self.load_state_dict(torch.load(load_path+'/weight.pt'))