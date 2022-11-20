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
        loc_dict = {'서울특별시':0,
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
        }
    ):
        self.user_taste = user_taste_dict
        self.spots = spot_dict
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
        
        for spot in self.spots:
            spotMap[spot['RecommendationsSpotResponseDto']['id']]=spot['RecommendationsSpotResponseDto']
        self.spotMap = spotMap
        self.spotNum = len(spotMap)

    def load_data_from_json(
        self,
        path,
    ):  
        self._load_tables_from_json_file(path)
        self._make_spot_map()

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
            try:
                loc_id = self.loc_dict(spot['metroName'])
            except:
                loc_id = 16
            loc_list[loc_id] = 1
            spot_loc_matrix.append(loc_list)
            
        return spot_loc_matrix

    def get_user_theme_matrix(self):
        user_theme_matrix = []
        for user in self.user_taste:
            theme_list = [0] * len(self.theme_dict)
            for theme in user['UsersTasteThemesResponseDto']['themes']:
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
