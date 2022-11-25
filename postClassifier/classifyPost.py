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
from utils import OneClassClassificationDataset_no_label, PostClassificationDataset_no_label, DataExporter
from kobert_tokenizer import KoBERTTokenizer

"""
specify input xl file path, and output path
"""

ood_path = './saved_model/occ_standard/'
thm_path = './saved_model/thm_standard/'


input_path = sys.argv[1]
output_path = sys.argv[2]
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache() 
epsilon = sys.float_info.epsilon   

def predict_with_no_gate(epoch, model, device, loader):
    model.eval()

    generated_labels = {"is_trip":[]}


    with torch.no_grad():
        for _, data in enumerate(loader, 0):

            ids  = data['input_text_ids'].to(device, dtype = torch.long)
            mask = data['input_mask'].to(device, dtype = torch.long)

            logits = model.forward(input_ids = ids, attention_mask = mask).logits

            predicted_class_id = torch.argmax(F.softmax(logits,dim=1), dim=1).cpu().item()

            generated_labels["is_trip"].extend([predicted_class_id])

        
    return generated_labels

def predict_with_gate(epoch, model, device, loader, gater = None, gater_list = None):
    model.eval()

    generated_labels = {"theme_id":[]}

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            if not gater(gater_list[_]) :
                continue

            ids  = data['input_text_ids'].to(device, dtype = torch.long)
            mask = data['input_mask'].to(device, dtype = torch.long)

            logits = model.forward(input_ids = ids, attention_mask = mask).logits

            predicted_class_id = torch.argmax(F.softmax(logits,dim=1), dim=1).cpu().item()

            generated_labels["theme_id"].extend([predicted_class_id])
        
    return generated_labels

def df_load_from_path(input_file_path):

    _is_xlsx = False
    _is_csv = False

    file_format = input_file_path.split(".")[1]
    if file_format == 'csv':
        _is_csv = True
        _is_xlsx = False
    if file_format == 'xlsx':
        _is_csv = False
        _is_xlsx = True
    else :
        AssertionError("File Format Uknown")

    if _is_csv and not _is_xlsx:
        return pd.read_csv(input_file_path)
    elif _is_xlsx and not _is_csv :
        return pd.read_excel(input_file_path)
    else :
        AssertionError("File Format Uknown")

def main(input_file_path, output_file_path):
    
    
    file_exists = os.path.exists(input_file_path)
    if not file_exists:
        AssertionError("No input File Found for Classification require path to csv, xlxs")

    SEED = 420               # random seed (default: 42)
    MAX_LEN = 512
    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
        }
    
    torch.backends.cudnn.deterministic = True
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    dfdataset = df_load_from_path(input_file_path)
    ooddataset = OneClassClassificationDataset_no_label(dfdataset, tokenizer, MAX_LEN)
   
    ood_loader = DataLoader(ooddataset, **val_params)
    
    print(len(dfdataset))
    print(dfdataset.head(10))

    ood_model = BertForSequenceClassification.from_pretrained(ood_path)
    ood_model.to(device)
    """
    First determine if the post is travel related or not
    """
    for epoch in range(1):
        generated_is_trip_label = predict_with_no_gate(epoch, ood_model, device, ood_loader)

    dfdataset['is_trip'] = pd.DataFrame.from_dict(generated_is_trip_label)
    dfthmdataset = dfdataset[dfdataset.is_trip != 1]
    dfthmdataset.reset_index(drop=True, inplace=True)

    thmdataset = PostClassificationDataset_no_label(dfthmdataset, tokenizer, MAX_LEN)
    thm_loader = DataLoader(thmdataset, **val_params)

    thm_model = BertForSequenceClassification.from_pretrained(thm_path,
                                                          num_labels = 21,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    thm_model.to(device)

    def is_trip_gater(is_a_trip_label):
        if is_a_trip_label == 0:
            return True
        else :
            return False

    """
    Extract tag information from post
    """

    for epoch in range(1):
        generated_theme_label = predict_with_no_gate(epoch, thm_model, device, thm_loader)

    dfthmdataset["theme_id"] = pd.DataFrame.from_dict(generated_theme_label)
    dfthmdataset = dfthmdataset[dfthmdataset.theme_id != 20]
    """
    Export to file
    """
    myDataExpoerter = DataExporter(dfthmdataset, "3def73060f55c3515922f19109dc469e")
    myDataExpoerter.add_theme_name_and_replace()
    myDataExpoerter.add_metro_local_address()
    myDataExpoerter.clean_datetime_and_replace()
    myDataExpoerter.convert_metro_and_local_name()
    myDataExpoerter.data = myDataExpoerter.data[myDataExpoerter.data.metroName != None]

    myDataExpoerter.add_rank_and_replace()
    print(myDataExpoerter.data)
    myDataExpoerter.data.rename(columns = {'place':'name','like':'likeNumber','text':'content', 'link':'photoUrl', 'datetime':'date'},inplace=True)
    myDataExpoerter.data = myDataExpoerter.data.drop(columns = ['is_trip','theme_id'])
    
    data_dict = myDataExpoerter.data.to_dict('index')
    data_list = [value for value in data_dict.values()]
    #print(data_list[0])
    #myDataExpoerter.data.to_json(output_file_path, force_ascii= False, orient='index')
    with open(output_file_path, "w", encoding='utf8') as json_file:
        result_as_json = json.dumps(data_list, ensure_ascii=False)
        result_as_json = result_as_json.replace('NaN','null')
        result_as_json = result_as_json.replace('None','null')
        json_file.write(result_as_json)


    print(myDataExpoerter.data)

    
if __name__ == '__main__':
    main(input_path, output_path)