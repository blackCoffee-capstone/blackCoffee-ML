#-*- coding: utf-8 -*-

import requests
import json
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from kobert_tokenizer import KoBERTTokenizer

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
        return len(self.labels)

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
        labels_dict = {'산'     : 0,
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
        return len(self.labels)
    
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
        labels_dict = {'산'     : 0,
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
        tags,
        rest_api_key
    ):
        self.data = dataframe
        self.latitude = self.data['latitude']
        self.longitude= self.data['longitude']
        self.tags = tags
        self.rest_api_key = rest_api_key

    def _get_address(
        self,
        lat,
        lng
    ):
        url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x="+lng+"&y="+lat
        headers = {"Authorization": "KakaoAK "+ self.rest_api_key}
        api_json = requests.get(url, headers=headers)
        full_address = json.loads(api_json.text)

        return full_address

    def _get_items(
        self,
        index
    ):
        lat = self.latitude[index]
        lng = self.longitude[index]
        address = self._get_address(lat,lng)

        return


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