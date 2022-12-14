#-*- coding: utf-8 -*-
import re
import requests
import json
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from kobert_tokenizer import KoBERTTokenizer
import itertools 
from preprocessor import PreProcessor

class OneClassClassificationDataset(Dataset):

    def __init__(
        self, 
        dataframe, 
        tokenizer, 
        input_max_len,
        labels_dict = {'true' : 0, '__label2__' : 1}, 
    ):
        self.tokenizer  = tokenizer 
        self.data       = dataframe
        self.input_text = self.data['text']
        self.labels     = self.data['is_trip']
        self.labels_dict = labels_dict
        self.input_max_len = input_max_len
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, 
        index
    ):
        input_text = str(self.input_text[index])
        myPreProcessor = PreProcessor()
        input_text = myPreProcessor(input_text)
        input_text = ' '.join(input_text.split())
        input_text = self.tokenizer([input_text],
                                    padding = 'max_length',
                                    max_length=self.input_max_len,
                                    truncation = True,
                                    return_tensors="pt")

        input_text_ids = input_text['input_ids'].squeeze()
        input_mask     = input_text['attention_mask']
     
        if self.labels[index] == True :
            labels_y = 0
        else :
            labels_y = 1
       
        labels_y       = torch.tensor([labels_y])
        
        return {
            'input_text_ids' : input_text_ids.to(dtype=torch.long),
            'input_mask'     : input_mask.to(dtype=torch.long),
            'labels_y'       : labels_y.to(dtype=torch.long)
        }

class OneClassClassificationDataset_no_label(OneClassClassificationDataset):
    def __init__(
        self, 
        dataframe, 
        tokenizer, 
        input_max_len,
        labels_dict = {'true' : 0, '__label2__' : 1}, 
    ):
        self.tokenizer  = tokenizer 
        self.data       = dataframe
        self.input_text = self.data['text']
        self.labels_dict = labels_dict
        self.input_max_len = input_max_len
    
    def __getitem__(
        self, 
        index
    ):
        input_text = str(self.input_text[index])
        myPreProcessor = PreProcessor()
        input_text = myPreProcessor(input_text)
        input_text = ' '.join(input_text.split())
        input_text = self.tokenizer([input_text],
                                    padding = 'max_length',
                                    max_length=self.input_max_len,
                                    truncation = True,
                                    return_tensors="pt")

        input_text_ids = input_text['input_ids'].squeeze()
        input_mask     = input_text['attention_mask']
     
        return {
            'input_text_ids' : input_text_ids.to(dtype=torch.long),
            'input_mask'     : input_mask.to(dtype=torch.long)
        }

class PostClassificationDataset(Dataset):

    def __init__(
        self, 
        dataframe, 
        tokenizer, 
        input_max_len,
        labels_dict = {'???'     : 0,
                       '?????????' : 1,
                       '??????'   : 2,
                       '??????'   : 3,
                       '??????'   : 4,
                       '??????'   : 5,
                       '??????'   : 6,
                       '?????????' : 7,
                       '??????'   : 8,
                       '??????'   : 9,
                       '??????'   : 10,
                       '??????'   : 11,
                       '????????????':12,
                       '??????'   : 13,
                       '????????????':14,
                       '??????'   :15,
                       '???'     :16,
                       '??????'   :17,
                       '?????????' :18,
                       '??????'   :19
                       }, 
    ):
        self.tokenizer  = tokenizer 
        self.data       = dataframe
        self.input_text = self.data['text']
        self.labels     = self.data['topic']
        self.labels_dict = labels_dict
        self.input_max_len = input_max_len
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.data)
    
    def _get_theme_id(
        self,
        input_tags,
    ):  
        input_tags = str(input_tags)
        tags = input_tags.split(", ")
        try :
            theme_id = self.labels_dict[tags[0]]
        except :
            try : 
                theme_id = self.labels_dict[tags[1]]
            except :
                theme_id = 20
        return theme_id

    def __getitem__(
        self, 
        index
    ):
        input_text = str(self.input_text[index])
        myPreProcessor = PreProcessor()
        input_text = myPreProcessor(input_text)
        input_text = ' '.join(input_text.split())
        input_text = self.tokenizer([input_text],
                                    padding = 'max_length',
                                    max_length=self.input_max_len,
                                    truncation = True,
                                    return_tensors="pt")

        input_text_ids = input_text['input_ids'].squeeze()
        input_mask     = input_text['attention_mask']
        
        labels_y       = self._get_theme_id(self.labels[index])
        labels_y       = torch.tensor([labels_y])
        
        return {
            'input_text_ids' : input_text_ids.to(dtype=torch.long),
            'input_mask'     : input_mask.to(dtype=torch.long),
            'labels_y'       : labels_y.to(dtype=torch.long)
        }

class PostClassificationDataset_no_label(PostClassificationDataset):
    def __init__(
        self, 
        dataframe, 
        tokenizer, 
        input_max_len,
        labels_dict = {'???'     : 0,
                       '?????????' : 1,
                       '??????'   : 2,
                       '??????'   : 3,
                       '??????'   : 4,
                       '??????'   : 5,
                       '??????'   : 6,
                       '?????????' : 7,
                       '??????'   : 8,
                       '??????'   : 9,
                       '??????'   : 10,
                       '??????'   : 11,
                       '????????????':12,
                       '??????'   : 13,
                       '????????????':14,
                       '??????'   :15,
                       '???'     :16,
                       '??????'   :17,
                       '?????????' :18,
                       '??????'   :19
                       }, 
    ):
        self.tokenizer  = tokenizer 
        self.data       = dataframe
        self.input_text = self.data['text']
        self.labels_dict = labels_dict
        self.input_max_len = input_max_len
    
    def __getitem__(
        self, 
        index
    ):
        input_text = str(self.input_text[index])
        myPreProcessor = PreProcessor()
        input_text = myPreProcessor(input_text)
        input_text = ' '.join(input_text.split())
        input_text = self.tokenizer([input_text],
                                    padding = 'max_length',
                                    max_length=self.input_max_len,
                                    truncation = True,
                                    return_tensors="pt")

        input_text_ids = input_text['input_ids'].squeeze()
        input_mask     = input_text['attention_mask']

        return {
            'input_text_ids' : input_text_ids.to(dtype=torch.long),
            'input_mask'     : input_mask.to(dtype=torch.long),
        }


class DataExporter():
    """
        Export Datas 
    """
    
    def __init__(
        self,
        dataframe,
        rest_api_key
    ):
        self.data = dataframe
        self.rest_api_key = rest_api_key

    def _get_address(
        self,
        row
    ):  
        """
        "address":{"0":{"meta":{"total_count":2},
        "documents":
            [{"region_type":"B",
            "code":"4713012000",
            "address_name":"???????????? ????????? ?????????"
            ,"region_1depth_name":"????????????",
            "region_2depth_name":"?????????",
            "region_3depth_name":"?????????",
            "region_4depth_name":"",
            "x":129.2406321303,"y":35.7909298509},{"region_type":"H","code":"4713060500","address_name":"???????????? ????????? ?????????","region_1depth_name":"????????????","region_2depth_name":"?????????","region_3depth_name":"?????????","region_4depth_name":"","x":129.2196808723,"y":35.8365079286}]}
        """
        
        lat = str(row['latitude'])
        lng = str(row['longitude'])
        #print(lat,lng)
        #url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
        params = {"x": f"{lng}",
                  "y": f"{lat}"}
        url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x="+lng+"&y="+lat
        #res = requests.get(self.URL_02, headers=self.headers, params=params)
        headers = {"Authorization": "KakaoAK "+ self.rest_api_key}
        #api_json = requests.get(url, headers=headers, params=params)
        api_json = requests.get(url, headers=headers)
        full_address = json.loads(api_json.text)
        
        #print(full_address, "latitude should be 37~38 but", lat)
        return full_address['documents'][0]['region_1depth_name'], full_address['documents'][0]['region_2depth_name'], full_address['documents'][0]["address_name"]

    def _search_kakao_coordinate(
        self,
        search_address
    ):
        url = 'https://dapi.kakao.com/v2/local/search/address.json'
        # rest_api_key = secrets['Kakao']['rest_key']

        header = {'Authorization': 'KakaoAK ' + self.rest_api_key}
        params = dict(query=search_address, analyze_type='exact')
        result = requests.get(url, headers=header, params=params).json()

        if len(result['documents']) > 0:
            #print(result['documents'][0])
            longitude = result['documents'][0]['x']
            latitude  = result['documents'][0]['y']
            address   = result['documents'][0]['address_name']
            metroName = result['documents'][0]['address']['region_1depth_name']
            localName = result['documents'][0]['address']['region_2depth_name']
        else:
            longitude = None
            latitude  = None
            address   = None
            metroName = None
            localName = None

        return pd.Series([longitude, latitude, metroName, localName, address])
    
    def add_log_lat_metro_local_address(self):
        data = self.data 
        data[["longitude","latitude","metroName","localName","address"]]= data.apply(lambda x : self._search_kakao_coordinate(x["place"]),axis=1)
        self.data = data

    def _get_pandas_series_address(
        self,
        row
    ):
        try :
            metroName, localName, address = self._get_address(row)
            #print([metroName, localName, address])
        except :
            metroName, localName, address = None, None, None
        
        
        return pd.Series([metroName, localName, address])


    def add_metro_local_address(self):
        data = self.data 
        data[["metroName","localName","address"]]= data.apply(lambda x : self._get_pandas_series_address(x),axis=1)
        self.data = data


    def _replace_theme_id_to_name(
        self,
        row
    ):  
        labelMap = ['???', '?????????', '??????', '??????', '??????','??????','??????','?????????','??????' ,'??????','??????','??????','????????????','??????','????????????','??????','???','??????','?????????','??????','ERR']

        return labelMap[row["theme_id"]]

    def add_theme_name_and_replace(
        self
    ):  
        #labelMap = ['???', '?????????', '??????', '??????', '??????','??????','??????','?????????','??????' ,'??????','??????','??????','????????????','??????','????????????','??????','???','??????','?????????','??????','ERR']
        
        self.data['themeName'] = self.data.apply(lambda x : self._replace_theme_id_to_name(x), axis = 1)

        return


    def _clean_datetime(
        self,
        row
    ):  
        if type(row["datetime"]) != str:
            newdatetime = row["datetime"].strftime('%Y-%m-%d')
        else:
            newdatetime = row["datetime"][:10]
        return newdatetime

    def clean_datetime_and_replace(self):
        self.data['datetime'] = self.data.apply(lambda x : self._clean_datetime(x), axis = 1)

    def _aget_weekly_hot_top_N(
            self,
            N = 10,
        ):
        ## from weekly collected sns post data
        ## return top N hot places
        
        place_like_pairs = {}
        for place in self.data["place"].unique():
            place_like_pairs[place] = self.data.loc[self.data['place'] == place, 'like'].sum()
        
        place_like_pairs =  dict(sorted(place_like_pairs.items(), key=lambda item: item[1], reverse=True))
        top_place_list = list(place_like_pairs.keys())
        if len(top_place_list) >= N :
            return top_place_list[:N]#dict(itertools.islice(place_like_pairs.items(), N))
        else : 
            return top_place_list
    
    def _clean_like(self):
        ## clean like value
        ## if is type string fill 0
        def removeComma(text):
            text = re.sub( ",", "",text)
            return text

        self.data['like'] = self.data.apply(lambda x : str(x['like']), axis = 1)
        self.data['like'] = self.data.apply(lambda x : removeComma(x['like']) if type(x['like']) is type("string") else x['like'], axis = 1)
        self.data['like'] = self.data.apply(lambda x : eval(x['like']) if x['like'].isdigit() else x['like'], axis = 1)
        self.data['like'] = self.data.apply(lambda x : 0 if type(x['like']) is type("string") else x['like'], axis = 1)


    def add_rank_and_replace(self):
        self._clean_like()
        top_20_places = self._aget_weekly_hot_top_N(len(self.data))
        self.data['rank'] = self.data.apply(lambda x : top_20_places.index(x["place"])+1 if x["place"] in top_20_places else None, axis = 1)

        return


    def convert_metro_and_local_name(self):
        metroMap = {"???????????????" : "??????",
                    "???????????????" : "??????",
                    "???????????????" : "??????",
                    "???????????????" : "??????",
                    "???????????????" : "??????",
                    "???????????????" : "??????",
                    "???????????????" : "??????",
                    "?????????????????????" : "??????", 
                    "?????????????????????" : "??????",
                    "?????????" : "??????",
                    "?????????" : "??????",
                    "????????????" : "??????",
                    "????????????" : "??????",
                    "????????????" : "??????",
                    "????????????" : "??????",
                    "????????????" : "??????",
                    "????????????" : "??????"
                    }

        
        localNullMetro = {
                    "?????????????????????" : "??????", 
                    }
        
        def suwon_exception(
            text
        ):  
            if text == None:
                return None
                
            else:
                return text.split(' ')[0]

        def is_in_metro(input_metro_name):
            if input_metro_name in metroMap:
                return metroMap[input_metro_name]
            else: return None
        
        self.data['localName'] = self.data.apply(lambda x : None if x["metroName"] in localNullMetro else x["localName"], axis = 1)
        self.data['localName'] = self.data.apply(lambda x : suwon_exception(x["localName"]), axis = 1)
        self.data['metroName'] = self.data.apply(lambda x : is_in_metro(x['metroName']), axis = 1)


    def _markSpotByName(
        self,
        row : pd.Series
    ) -> int:  
        spot_name = row["place"]

        filtering_names = [
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "Seoul",
            "Seoul Korea",
            "Seoul. Korea",
            "South Korea",
            "Daegu"
        ]
    
        if spot_name in filtering_names:
            ## INVALID SPOT NAME
            return 0    
        else :
            ## VALID SPOT NAME
            return 1

    def sortOutByNames(self):
        data = self.data 
        data["invalid_name"]= data.apply(lambda x : self._markSpotByName(x),axis=1)
        data = data[data.invalid_name != 0]
        
        data = data.drop(columns = ['invalid_name'])
        self.data = data

def testOneClassClassificationDataset():
    dfdataset  = pd.read_excel('testingData/instagram_post.xlsx')
    dftrainset = dfdataset.sample(frac=0.8,random_state=420)
    dftestset  = dfdataset.drop(dftrainset.index)
    dftrainset.reset_index(drop=True, inplace=True)
    dftestset.reset_index(drop=True, inplace=True)
    
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    training_set = OneClassClassificationDataset(dftrainset, tokenizer, 512)
    test_set     = OneClassClassificationDataset(dftestset , tokenizer, 512)
    
    train_params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
        }

    train_loader = DataLoader(training_set, **train_params)
    test_loader  = DataLoader(test_set, **val_params)
    
    for _, data in enumerate(train_loader, 0):
        
        print(data['labels_y'])

    assert False


def testClassificationDataset():
    dfdataset  = pd.read_excel('testingData/instagram_post.xlsx')
    dftrainset = dfdataset.sample(frac=0.8,random_state=420)
    dftestset  = dfdataset.drop(dftrainset.index)
    dftrainset.reset_index(drop=True, inplace=True)
    dftestset.reset_index(drop=True, inplace=True)
    
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    training_set = PostClassificationDataset(dftrainset, tokenizer, 512)
    test_set     = PostClassificationDataset(dftestset , tokenizer, 512)
    
    train_params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
        }

    train_loader = DataLoader(training_set, **train_params)
    test_loader  = DataLoader(test_set, **val_params)
    
    for _, data in enumerate(train_loader, 0):
        
        print(data['labels_y'])

    assert False

def testExporter():
    assert False