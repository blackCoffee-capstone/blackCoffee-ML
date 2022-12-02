import numpy as np
import pandas as pd
import json
import torch
import pickle
import sys
from torch.utils.data import Dataset

class Item():
    def __init__(
        self,
        item_id = np.array([], dtype = 'i'),
        themes = np.array([], dtype = 'i')
    ):
        self.item_id = item_id
        self.themes = themes
        
        return

class RecommendationDataset():
    def __init__(
        self,
        user_taste_dict = None,
        spot_dict = None,
        user_map = None,
        spot_map = None,
        spotNum = None,
        userNum = None,
        theme_dict = {'산'     : 0,
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
        loc_dict = {'서울':0,
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
        }
    ):
        self.user_taste = user_taste_dict
        self.spots = spot_dict
        self.userMap = user_map
        self.spotMap = spot_map
        self.spotNum = spotNum
        self.userNum = userNum
        self.theme_dict = theme_dict
        self.loc_dict = loc_dict
        return

    def _load_tables_from_json_file(
        self,
        path
    ):
        ## Load Data from Json
        with open(path, 'r', encoding="UTF-8") as file:
            data = json.load(file)
    
        self.user_taste = data['usersTastes']
        self.spots = data['spots']

        self.userNum = len(self.user_taste)
        self.spotNum = len(self.spots)
        return

    def _make_spot_map(self):
        spotMap = {}
        n = 0
        for spot in self.spots:
            spotMap[n]=spot['RecommendationsSpotResponseDto']
            n = n + 1 
        self.spotMap = spotMap
        self.spotNum = len(spotMap)
    
    def _make_user_map(self):
        userMap = {}
        n = 0
        for user in self.user_taste:
            userMap[n]=user['UsersTasteThemesResponseDto']
            n = n + 1 
        self.userMap = userMap
        self.userNum = len(userMap)

    def get_user_id(
        self,
        input_user_id
    ):
        for user_id, original_user in self.userMap.items():
            if original_user['user_id'] == input_user_id:
                return user_id
        
        return 0

    def load_data_from_json(
        self,
        path,
    ):  
        self._load_tables_from_json_file(path)
        self._make_spot_map()
        self._make_user_map()

        return

    def _process_data(
        self,
        path
    ):
        return

    def get_spots(self):
        self.spots
        self.spotMap

    def get_spot_theme_matrix(self):
        spot_theme_matrix = []
        for id, spot in self.spotMap.items():
            theme_list = [0] * len(self.theme_dict)
            for theme in spot['themes']:
                theme_id = self.theme_dict[theme]
                theme_list[theme_id] = 1
            spot_theme_matrix.append(theme_list)

        return spot_theme_matrix

    def get_spot_loc_matrix(self):
        spot_loc_matrix = []
        for id, spot in self.spotMap.items():
            loc_list = [0] * len(self.loc_dict)
            if spot['metroName'] in self.loc_dict:
                loc_id = self.loc_dict[spot['metroName']]
            else:
                loc_id = self.loc_dict['Uknown']

            loc_list[loc_id] = 1
            spot_loc_matrix.append(loc_list)
            
        return spot_loc_matrix

    def get_user_theme_matrix(self):
        user_theme_matrix = []
        for id, user in self.userMap.items():
            theme_list = [0] * len(self.theme_dict)
            for theme in user['themes']:
                theme_id = self.theme_dict[theme]
                theme_list[theme_id] = 1
            user_theme_matrix.append(theme_list)

        return user_theme_matrix

        
    def from_npArray_get_spot(
        self,
        ndArray
    ):
        new_list = []
        item_ids = ndArray.tolist()
        
        for item_id in item_ids:
            new_list.append(self.spotMap[item_id])
        return new_list


    def from_npArray_get_spot_id(
        self,
        ndArray
    ):
        new_list = []
        item_ids = ndArray.tolist()
        
        for item_id in item_ids:
            new_list.append(self.spotMap[item_id]['id'])
        return new_list

class newRecommendationDataset():
    def __init__(
        self,
        user_taste_dict = None,
        spot_dict = None,
        user_map = None,
        spot_map = None,
        spotNum = None,
        userNum = None,
        theme_dict = {'산'     : 0,
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
        loc_dict = {'서울':0,
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
        }
    ):
        self.user_taste = user_taste_dict
        self.spots = spot_dict
        self.userMap = user_map
        self.spotMap = spot_map
        self.spotNum = spotNum
        self.userNum = userNum
        self.theme_dict = theme_dict
        self.loc_dict = loc_dict
        return

    def _load_tables_from_json_file(
        self,
        path
    ):
        ## Load Data from Json
        with open(path, 'r', encoding="UTF-8") as file:
            data = json.load(file)
    
        self.user_taste = data['allTasteThemes']
        self.spots = data['spots']

        self.userNum = len(self.user_taste)
        self.spotNum = len(self.spots)
        return

    def _make_spot_map(self):
        spotMap = {}
        n = 0
        for spot in self.spots:
            spotMap[n]=spot
            n = n + 1 
        self.spotMap = spotMap
        self.spotNum = len(spotMap)
    
    def _make_user_map(self):
        userMap = {}
        n = 0
        for user in self.user_taste:
            userMap[n]=user
            n = n + 1 
        self.userMap = userMap
        self.userNum = len(userMap)

    def get_user_id(
        self,
        input_user_id
    ):
        for user_id, original_user in self.userMap.items():
            if original_user['id'] == input_user_id:
                return user_id
        
        return 0

    def load_data_from_json(
        self,
        path,
    ):  
        self._load_tables_from_json_file(path)
        self._make_spot_map()
        self._make_user_map()

        return

    def _process_data(
        self,
        path
    ):
        return

    def get_spots(self):
        self.spots
        self.spotMap

    def get_spot_theme_matrix(self):
        spot_theme_matrix = []
        for id, spot in self.spotMap.items():
            theme_list = [0] * len(self.theme_dict)
            for theme in spot['themes']:
                if theme in self.theme_dict:
                    theme_id = self.theme_dict[theme]
                else :
                    theme_id = self.theme_dict['Uknown']
                theme_list[theme_id] = 1
            spot_theme_matrix.append(theme_list)

        return spot_theme_matrix

    def get_spot_loc_matrix(self):
        spot_loc_matrix = []
        for id, spot in self.spotMap.items():
            loc_list = [0] * len(self.loc_dict)
            if spot['metroName'] in self.loc_dict:
                loc_id = self.loc_dict[spot['metroName']]
            else:
                loc_id = self.loc_dict['Uknown']

            loc_list[loc_id] = 1
            spot_loc_matrix.append(loc_list)
            
        return spot_loc_matrix

    def get_user_theme_matrix(self):
        user_theme_matrix = []
        for id, user in self.userMap.items():
            theme_list = [0] * len(self.theme_dict)
            for theme in user['themes']:
                if theme in self.theme_dict:
                    theme_id = self.theme_dict[theme]
                else :
                    theme_id = self.theme_dict['Uknown']
                theme_list[theme_id] = 1
            user_theme_matrix.append(theme_list)

        return user_theme_matrix

        
    def from_npArray_get_spot(
        self,
        ndArray
    ):
        new_list = []
        item_ids = ndArray.tolist()
        
        for item_id in item_ids:
            new_list.append(self.spotMap[item_id])
        return new_list


    def from_npArray_get_spot_id(
        self,
        ndArray
    ):
        new_list = []
        item_ids = ndArray.tolist()
        
        for item_id in item_ids:
            new_list.append(self.spotMap[item_id]['id'])
        return new_list


class SpotMap():
    def __init__(
        self,
        spot_map = None,
        spot_feature_map = None,
        number_of_spots = None,
    ):
        self.spot_map = spot_map
        self.spot_feature_map = spot_feature_map
        self.number_of_spots = number_of_spots

    def get_features_according_to_map(
        self,
        spot_id : int,
        feature_map : dict,
        unknown_feature_id : int
    ) -> list:
        feature_ids = []

        for feature in self.spot_feature_map[spot_id]:
            if feature in feature_map:
                feature_ids.append(feature_map[feature])
            else :
                feature_ids.append(unknown_feature_id)

        return feature_ids

    def filter_by_spot_features(
        self,
        valid_feature_list : list
    ) -> list:
        valid_spot_ids = []

        for spot_id, spot_feature in self.spot_feature_map.items(): 
            
            is_valid  = False
            for target_feature in valid_feature_list:
                if target_feature in spot_feature:
                    is_valid = True
                    break
            
            if is_valid:
                valid_spot_ids.append(spot_id)

        return valid_spot_ids

    def from_dfspots_make_map(
        self,
        df_spots
    ):
        spot_record = df_spots.to_dict("records")
        spot_map = {}
        spot_feature_map = {}

        new_spot_id = 1
        for spot in spot_record:
            spot_map[spot['id']] = new_spot_id
            
            spot_features = self._from_spot_record_get_feature_list(spot)
            spot_feature_map[new_spot_id] = spot_features
            
            new_spot_id = new_spot_id + 1

        self.spot_map = spot_map
        self.spot_feature_map = spot_feature_map
        self.number_of_spots = len(spot_map)


    def _from_spot_record_get_feature_list(
        self,
        spot
    ):  

        feature_list = spot['themes'] 
        feature_list.append(spot['metroName'])

        return feature_list


    def load_from_pickle(
        self,
        path
    ) :
        spot_map = {}
        spot_feature_map = {}
        
        path_spot = path + "/spot_map.pickle"
        path_spot_feature = path + "/spot_feature_map.pickle"
        
        with open(path_spot, 'rb') as f:
            spot_map = pickle.load(f)
        
        with open(path_spot_feature, 'rb') as f:
            spot_feature_map = pickle.load(f)
        
        self.spot_map = spot_map
        self.spot_feature_map = spot_feature_map
        self.number_of_spots = len(spot_map)

    def export_to_pickle(
        self,
        path
    ) :
        path_spot = path + "/spot_map.pickle"
        path_spot_feature = path + "/spot_feature_map.pickle"

        spot_map = self.spot_map
        with open(path_spot, 'wb') as f:
            pickle.dump(spot_map, f)

        spot_feature_map = self.spot_feature_map
        with open(path_spot_feature, 'wb') as f:
            pickle.dump(spot_feature_map, f)
    
class UserMap():
    def __init__(
        self,
        user_map = None,
        user_feature_map = None,
        number_of_users = None,
    ):
        self.user_map = user_map
        self.user_feature_map = user_feature_map
        self.number_of_users = number_of_users

    def get_features_according_to_map(
        self,
        user_id : int,
        feature_map : dict,
        unknown_feature_id : int
    ) -> list:
        feature_ids = []

        for feature in self.user_feature_map[user_id]:
            if feature in feature_map:
                feature_ids.append(feature_map[feature])
            else :
                feature_ids.append(unknown_feature_id)

        return feature_ids

    def _from_user_record_get_feature_list(
        self,
        spot
    ):  
        feature_list = spot['themes']

        return feature_list

    def from_dfuserTaste_make_map(
        self,
        df_userTaste
    ):
        taste_record = df_userTaste.to_dict("records")
        user_map = {}
        user_feature_map = {}
        new_user_id = 1
        for user_taste in taste_record:
            user_map[user_taste['id']] = new_user_id

            user_features = self._from_user_record_get_feature_list(user_taste)
            user_feature_map[new_user_id] = user_features

            new_user_id = new_user_id + 1

        self.user_map = user_map
        self.user_feature_map = user_feature_map
        self.number_of_users = len(user_map)

    def load_from_pickle(
        self,
        path
    ) :
        user_map = {}
        user_feature_map = {}
        
        path_user = path + "/user_map.pickle"
        path_user_feature = path + "/user_feature_map.pickle"
        
        with open(path_user, 'rb') as f:
            user_map = pickle.load(f)

        with open(path_user_feature, 'rb') as f:
            user_feature_map = pickle.load(f)

        self.user_map = user_map
        self.number_of_users = len(user_map)
        self.user_feature_map = user_feature_map

    def export_to_pickle(
        self,
        path
    ) :
        path_user = path + "/user_map.pickle"
        path_user_feature = path + "/user_feature_map.pickle"

        user_map = self.user_map
        user_feature = self.user_feature_map
        with open(path_user, 'wb') as f:
            pickle.dump(user_map, f)

        with open(path_user_feature, 'wb') as f:
            pickle.dump(user_feature, f)

class HybridRecDataset(Dataset):
    
    def __init__(
        self,
        df_userTaste : pd.DataFrame = None,
        df_spotFeature : pd.DataFrame = None,
        df_userlikedspot : pd.DataFrame = None,
        df_uservisitedspot : pd.DataFrame = None,
        dict_user_feature_map : dict = {
            '산'           : 0,
            '럭셔리'       : 1,
            '역사'         : 2,
            '웰빙'         : 3,
            '바다'         : 4,
            '카페'         : 5,
            '공원'         : 6,
            '전시장'       : 7,
            '건축'         : 8,
            '사찰'         : 9,
            '가족'         : 10,
            '학교'         : 11,
            '놀이공원'     : 12,
            '겨울'         : 13,
            '엑티비티'     : 14,
            '캠핑'         : 15,
            '섬'           : 16,
            '커플'         : 17,
            '저수지'       : 18,
            '폭포'         : 19,
            'unknown_theme': 20
        },
        dict_item_feature_map : dict = {
            '산'           : 0,
            '럭셔리'       : 1,
            '역사'         : 2,
            '웰빙'         : 3,
            '바다'         : 4,
            '카페'         : 5,
            '공원'         : 6,
            '전시장'       : 7,
            '건축'         : 8,
            '사찰'         : 9,
            '가족'         : 10,
            '학교'         : 11,
            '놀이공원'     : 12,
            '겨울'         : 13,
            '엑티비티'     : 14,
            '캠핑'         : 15,
            '섬'           : 16,
            '커플'         : 17,
            '저수지'       : 18,
            '폭포'         : 19,
            'unknown_theme': 20,
            '서울'         : 21,
            '부산'         : 22,
            '대구'         : 23,
            '인천'         : 24,
            '광주'         : 25,
            '대전'         : 26,
            '울산'         : 27,
            '세종'         : 28,
            '경기'         : 29,
            '강원'         : 30,
            '충북'         : 31,
            '충남'         : 32,
            '전북'         : 33,
            '전남'         : 34,
            '경북'         : 35,
            '경남'         : 36,
            '제주'         : 37,
            'unknown_loc'  : 38
        }
    ) -> None:  
        self.dict_user_feature_map = dict_user_feature_map
        self.dict_item_feature_map = dict_item_feature_map
        
        if df_userTaste is None or df_spotFeature is None or df_userlikedspot is None or df_uservisitedspot is None:
            return

        df_user_item_interaction = self._generate_user_item_interaction(df_userlikedspot, df_uservisitedspot)
        df_user_feature_ids = self._generate_user_feature_ids(df_userTaste, dict_user_feature_map)
        df_spot_feature_ids = self._generate_spot_feature_ids(df_spotFeature, dict_item_feature_map)

        df_data = pd.merge(left=df_user_item_interaction, right=df_user_feature_ids, how="left", on="user_id", sort=True)
        df_data = pd.merge(left=df_data, right=df_spot_feature_ids, how="left", on="item_id", sort=True)
        print(df_data)

        self.data = df_data.to_dict("index")
        self._min_max_normalize_rating(5.0, 0.0)
        """
        data = {
            "0" : {
                "user_id" : 23,
                "user_feature" : [0,0,1,0,1,0,1,1,1],
                "item_id" : 543,
                "item_feature" : [0,0,1,0,1,1,1,0,1],
                "rating"  : 0.7
            },
            "1" : ...
        }
        """
    def _min_max_normalize_rating(
        self,
        max_rating : float,
        min_rating : float
    ) -> None:  

        data = self.data

        for index, record in data.items():
            rating = record["rating"]
            normalized_rating = (rating - min_rating) / (max_rating - min_rating)
            data[index]["min_max_normalized_rating"] = normalized_rating

        self.data = data



    def _calculate_ratings(
        self,
        row : pd.Series
    ) -> float:
        rating = 0.0 

        if not np.isnan(row["liked_item_id"]):
            rating = rating + 3.6
        if not np.isnan(row["visited_item_id"]):
            rating = rating + 0.1 * min(row['visit_count'] ,14)

        return rating
    
    def _get_item_id(
        self,
        row : pd.Series
    ) -> int:
        item_id = 0
        if not np.isnan(row["liked_item_id"]):
            item_id = int(row["liked_item_id"])
        if not np.isnan(row["visited_item_id"]):
            item_id = int(row["visited_item_id"])

        return item_id
    
    def _generate_user_item_interaction(
        self,
        df_userlikedspot : pd.DataFrame,
        df_uservisitedspot : pd.DataFrame
    ) -> pd.DataFrame:

        df_userlikedspot = df_userlikedspot.rename(columns={"spot_id":"liked_item_id"})
        df_visitcount = df_uservisitedspot.groupby(["user_id", "spot_id"]).size().reset_index(name="visit_count")
        df_visitcount= df_visitcount.rename(columns={"spot_id":"visited_item_id"})

        df_user_item_interaction = pd.merge(left = df_userlikedspot , right = df_visitcount, left_on=["user_id","liked_item_id"], right_on=["user_id","visited_item_id"], how = "outer", sort=True)
        
        df_user_item_interaction["rating"]  = df_user_item_interaction.apply(lambda x : self._calculate_ratings(x), axis = 1)
        df_user_item_interaction["item_id"] = df_user_item_interaction.apply(lambda x : self._get_item_id(x), axis = 1)
        df_user_item_interaction = df_user_item_interaction[["user_id","item_id","rating"]]
        
        return df_user_item_interaction

    def _get_user_feature_ids(
        self,
        row : pd.Series,
        dict_user_feature_map : dict
    ) -> pd.Series:

        user_id = row['id']
        user_feature_ids = [0] * len(dict_user_feature_map)
        for theme_name in row['themes']:
            user_feature_ids[dict_user_feature_map[theme_name]] = 1

        return pd.Series([user_id, user_feature_ids])

    def _generate_user_feature_ids(
        self,
        df_userTaste : pd.DataFrame,
        dict_user_feature_map : dict
    ) -> pd.DataFrame:

        df_userTaste[["user_id","user_feature"]] = df_userTaste.apply(lambda x : self._get_user_feature_ids(x, dict_user_feature_map),axis = 1)
        df_user_feature_ids = df_userTaste[["user_id","user_feature"]]

        return df_user_feature_ids
    
    def _get_item_feature_ids(
        self,
        row : pd.Series,
        dict_item_feature_map : dict
    ) -> pd.Series:

        item_id = row['id']
        item_feature_ids = [0] * len(dict_item_feature_map)
        for theme_name in row['themes']:
            item_feature_ids[dict_item_feature_map[theme_name]] = 1
        metroName = row['metroName']
        item_feature_ids[dict_item_feature_map[metroName]] = 1

        return pd.Series([item_id, item_feature_ids])

    def _generate_spot_feature_ids(
        self,
        df_spotFeature : pd.DataFrame,
        dict_item_feature_map : dict
    ) -> pd.DataFrame:
        
        df_spotFeature[["item_id","item_feature"]] = df_spotFeature.apply(lambda x : self._get_item_feature_ids(x, dict_item_feature_map),axis = 1)
        df_spot_feature_ids = df_spotFeature[["item_id","item_feature"]]
        
        return df_spot_feature_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, 
        index
    ):
        single_data = self.data[index]

        user_id      = torch.tensor(single_data["user_id"])
        user_feature = torch.tensor(single_data["user_feature"])
        item_id      = torch.tensor(single_data["item_id"])
        item_feature = torch.tensor(single_data["item_feature"])
        rating       = torch.tensor(single_data["rating"])
        min_max_normalized_rating = torch.tensor(single_data["min_max_normalized_rating"])
        return {
            'user_id'      : user_id.to(dtype = torch.long),
            'user_feature' : user_feature.to(dtype = torch.long),
            'item_id'      : item_id.to(dtype = torch.long),
            'item_feature' : item_feature.to(dtype = torch.long),
            'rating'       : rating.to(dtype = torch.float32),
            'min_max_normalized_rating' : min_max_normalized_rating.to(dtype = torch.float32)
        }

class RecCandidateDataset(HybridRecDataset):
    def generate_rec_candidate(
        self,
        user_id : int,
        user_feature : list,
        spot_map : SpotMap
    ) -> None:

        data = {}
        
        user_feature_ids = self._get_feature_ids_from_list(user_feature, self.dict_user_feature_map)   
        idx = 0
        for spot_id, spot_features in spot_map.spot_feature_map.items():
            item_feature_ids = self._get_feature_ids_from_list(spot_features, self.dict_item_feature_map)    
            data[idx] = {
                "user_id" : user_id,
                "user_feature" : user_feature_ids,
                "item_id" : spot_id,
                "item_feature" : item_feature_ids,
                "rating" : 0.0,
                "min_max_normalized_rating" : 0.0
            }
            idx = idx + 1
        
        self.data = data
    
    def _get_feature_ids_from_list(
        self,
        list : list,
        feature_map : dict
    ) -> list:

        feature_ids = [0] * len(feature_map)
        for feature_name in list:
            feature_ids[feature_map[feature_name]] = 1

        return feature_ids


class TargetUser():
    def __init__(
        self
    ):
        None

    def get_target_user_id_and_feature(
        self,
        file_path : str,
        user_map : UserMap,
        max_similarity_score :int,
    ) -> None:
        user_id = 0
        user_feature = []

        ## Load Data from Json
        with open(file_path, 'r', encoding="UTF-8") as f:
            data = json.load(f)

        feature_records = data['usersTastes']
        for record in feature_records:
            user_feature.append(record['name'])

        if data['userId'] in user_map.user_map :
            ## Seen User
            user_id = user_map.user_map[data['userId']]
        else:
            ## Unseen User get the most similar user
            user_id = self._generate_replacement_id_for_unseen_user(user_map, user_feature, max_similarity_score)
        
        self.user_id = user_id
        self.user_feature = user_feature

    def _jaccard_index(
        self,
        first_list : list,
        second_list : list,
    ) -> float:  
        
        co_exist = []
        for element in first_list:
            if element in second_list:
                co_exist.append(element)
        
        jaccard_index = len(co_exist) / (len(first_list) + len(second_list) + sys.float_info.epsilon)

        return jaccard_index

    def _score_user_features_similarity(
        self,
        first_features : list,
        second_features : list
    ) -> float:

        return self._jaccard_index(first_features, second_features)

    def _generate_replacement_id_for_unseen_user(
        self,
        user_map : UserMap,
        user_features : list,
        max_similarity_score : float
    ) -> int:
        user_id = 0

        top_similar_user = 0
        top_similarity_score = -1.0
        for candidate_user_id, candidate_user_feature in user_map.user_feature_map.items():
            new_score = self._score_user_features_similarity(user_features, candidate_user_feature)
            if new_score > top_similarity_score:
                top_similar_user = candidate_user_id
            
            if new_score > max_similarity_score:
                break
        
        user_id = top_similar_user

        return user_id