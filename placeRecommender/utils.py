import numpy as np
import pandas as pd
import json
import torch
import pickle
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
        number_of_spots = None,
    ):
        self.spot_map = spot_map
        self.number_of_spots = number_of_spots

    def from_dfspots_make_map(
        self,
        df_spots
    ):
        spot_record = df_spots.to_dict("records")
        spot_map = {}
        new_spot_id = 1
        for spot in spot_record:
            spot_map[spot['id']] = new_spot_id
            new_spot_id = new_spot_id + 1

        self.spot_map = spot_map
        self.number_of_spots = len(spot_map)

    def load_from_pickle(
        self,
        path
    ) :
        spot_map = {}
        with open(path, 'rb') as f:
            user_map = pickle.load(f)
        self.spot_map = spot_map
        self.number_of_spots = len(spot_map)

    def export_to_pickle(
        self,
        path
    ) :
        spot_map = self.spot_map
        with open(path, 'wb') as f:
            pickle.dump(spot_map, f)
    
class UserMap():
    def __init__(
        self,
        user_map = None,
        number_of_users = None,
    ):
        self.user_map = user_map
        self.number_of_users = number_of_users

    def from_dfuserTaste_make_map(
        self,
        df_userTaste
    ):
        taste_record = df_userTaste.to_dict("records")
        user_map = {}
        new_user_id = 1
        for user_taste in taste_record:
            user_map[user_taste['id']] = new_user_id
            new_user_id = new_user_id + 1

        self.user_map = user_map
        self.number_of_users = len(user_map)

    def load_from_pickle(
        self,
        path
    ) :
        user_map = {}
        with open(path, 'rb') as f:
            user_map = pickle.load(f)
        self.user_map = user_map
        self.number_of_users = len(user_map)

    def export_to_pickle(
        self,
        path
    ) :
        user_map = self.user_map
        with open(path, 'wb') as f:
            pickle.dump(user_map, f)


class HybridRecDataset(Dataset):
    
    def __init__(
        self,
        df_userTaste,
        df_spotFeature,
        df_userlikedspot,
        df_uservisitedspot,
        dict_user_feature_map = {
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
        dict_item_feature_map = {
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
            '경기도'       : 29,
            '강원도'       : 30,
            '충청도'       : 31,
            '전라도'       : 32,
            '경상도'       : 33,
            '제주'         : 34,
            'unknown_loc'  : 35
        }
    ):
        df_userlikedspot = df_userlikedspot.rename(columns={"spot_id":"liked_item_id"})
        df_visitcount = df_uservisitedspot.groupby(["user_id", "spot_id"]).size().reset_index(name="visit_count")
        df_visitcount= df_visitcount.rename(columns={"spot_id":"visited_item_id"})
        
        #print(df_userlikedspot, df_visitcount)
        df_user_item_interaction = pd.merge(left = df_userlikedspot , right = df_visitcount, left_on=["user_id","liked_item_id"], right_on=["user_id","visited_item_id"], how = "outer", sort=True)
        def calculate_ratings(row):
            ratings = 0.0

            #print(row["liked_item_id"], row["visited_item_id"])
            if not np.isnan(row["liked_item_id"]):
                ratings = ratings + 3.6
            if not np.isnan(row["visited_item_id"]):
                ratings = ratings + 0.1 * min(row['visit_count'] ,14)

            return ratings

        def get_item_id(row) :
            item_id = 0
            if not np.isnan(row["liked_item_id"]):
                item_id = int(row["liked_item_id"])
            if not np.isnan(row["visited_item_id"]):
                item_id = int(row["visited_item_id"])

            return item_id

        df_user_item_interaction["rating"]  = df_user_item_interaction.apply(lambda x : calculate_ratings(x), axis = 1)
        df_user_item_interaction["item_id"] = df_user_item_interaction.apply(lambda x : get_item_id(x), axis = 1)
        df_user_item_interaction = df_user_item_interaction[["user_id","item_id","rating"]]

        def get_user_feature_ids(row):
            user_id = row['id']
            user_feature_ids = [0] * len(dict_user_feature_map)
            for theme_name in row['themes']:
                user_feature_ids[dict_user_feature_map[theme_name]] = 1

            return pd.Series([user_id, user_feature_ids])

        df_userTaste[["user_id","user_feature"]] = df_userTaste.apply(lambda x : get_user_feature_ids(x),axis = 1)
        df_user_feature_ids = df_userTaste[["user_id","user_feature"]]

        def get_item_feature_ids(row):
            item_id = row['id']
            item_feature_ids = [0] * len(dict_item_feature_map)
            for theme_name in row['themes']:
                item_feature_ids[dict_item_feature_map[theme_name]] = 1
            metroName = row['metroName']
            item_feature_ids[dict_item_feature_map[metroName]] = 1

            return pd.Series([item_id, item_feature_ids])

        df_spotFeature[["item_id","item_feature"]] = df_spotFeature.apply(lambda x : get_item_feature_ids(x),axis = 1)
        df_spot_feature_ids = df_spotFeature[["item_id","item_feature"]]

        df_data = pd.merge(left=df_user_item_interaction, right=df_user_feature_ids, how="left", on="user_id", sort=True)
        df_data = pd.merge(left=df_data, right=df_spot_feature_ids, how="left", on="item_id", sort=True)
        print(df_data)

        self.data = df_data.to_dict("index")
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
        
        return {
            'user_id'      : user_id.to(dtype = torch.long),
            'user_feature' : user_feature.to(dtype = torch.long),
            'item_id'      : item_id.to(dtype = torch.long),
            'item_feature' : item_feature.to(dtype = torch.long),
            'rating'       : rating.to(dtype = torch.float32)
        }