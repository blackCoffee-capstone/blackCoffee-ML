import numpy as np
import pandas as pd
import json

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


class RecommendExporter():
    def __init__(
        self,

    ):
        return
