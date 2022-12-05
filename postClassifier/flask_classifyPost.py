import torch
import torch.nn as nn
from torch import cuda
import pandas as pd
import json
import sys
import os
from datetime import datetime
from transformers import BertModel, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn.functional as F
from preprocessor import PreProcessor
from utils import OneClassClassificationDataset_no_label, PostClassificationDataset_no_label, DataExporter
from kobert_tokenizer import KoBERTTokenizer
from flask import Flask, render_template, request, url_for

import requests
from bs4 import BeautifulSoup as bs
import json
import os.path

device = 'cpu'
ood_path = './saved_model/occ_standard/'
thm_path = './saved_model/thm_standard/'

app = Flask(__name__)

@app.route('/')
def home(is_trip="Yes/No Trip", theme="Mountain ..."):
    return render_template('inference.html', is_trip=is_trip, theme=theme)


def _predict(
    data,
    model,
    device
):
    model.eval()
    generated_labels = {"label":[]}
    
    input_text_ids = data['input_ids']
    input_mask     = data['attention_mask']

    ids  = input_text_ids.to(device, dtype = torch.long)
    #mask = input_mask.to(device, dtype = torch.long)

    print(ids.shape)        
    logits = model.forward(input_ids = ids, attention_mask = None).logits
    
    predicted_class_id = torch.argmax(F.softmax(logits,dim=1), dim=1).cpu().item()
    generated_labels["label"].extend([predicted_class_id])

    return generated_labels

def _inference(txt):
    
    
    MAX_LEN = 512
    

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    ood_model = BertForSequenceClassification.from_pretrained(ood_path)
    ood_model.to(device)

    thm_model = BertForSequenceClassification.from_pretrained(
        thm_path,
        num_labels = 21,
        output_attentions=False,
        output_hidden_states=False
    )
    thm_model.to(device)

    myPreProcessor = PreProcessor()

    input_text = myPreProcessor(txt)
    input_text = ' '.join(input_text.split())
    input_text = tokenizer(
        [txt],
        padding = 'max_length',
        max_length = MAX_LEN,
        truncation = True,
        return_tensors = "pt"
    )

    print(txt)
    #First determine if the post is travel related or not
    
    
    

    ood_labels = _predict(input_text, ood_model, device)
    print(ood_labels['label'][0])
    
    label_map = {
        0 : "mountain",
        1 : "luxury",
        2 : "history",
        3 : "well-being",
        4 : "ocean",
        5 : "cafe",
        6 : "park",
        7 : "gallery",
        8 : "architecture",
        9 : "budist-temple",
        10 : "family",
        11 : "school",
        12 : "pier",
        13 : "winter",
        14 : "sports and activities",
        15 : "cammping",
        16 : "island",
        17 : "couple",
        18 : "lake",
        19 : "waterfall" 
    }

    if ood_labels['label'][0] == 0 :
        is_trip_return = "This post is about Trip"
        
        thm_labels = _predict(input_text, thm_model, device)
        
        print(thm_labels)
        if thm_labels['label'][0] in label_map :
            theme_return = label_map[thm_labels['label'][0]]
        else : 
            theme_return = "Uknown Theme"
    else:
        is_trip_return = "This post is not about Trip"
        theme_return = "Not a trip, so no label"

    

    return is_trip_return, theme_return 


@app.route('/inference', methods=['POST','GET'])
def inference(is_trip ="", theme=""):
    if request.method == 'POST' :
        pass
    elif request.method == 'GET':
        txt = request.args.get('char1')
        is_trip, theme = _inference(txt)

        #is_trip = "test_trip" + txt
        #theme = "test_theme" + txt

        print(is_trip, theme)
        return render_template('inference.html', is_trip=is_trip, theme=theme )





if __name__ == '__main__':
    app.run(debug=True, threaded=True)