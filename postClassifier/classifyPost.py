import torch
import torch.nn as nn
from torch import cuda
import pandas as pd
import sys
import wandb
from datetime import datetime
from transformers import BertModel, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import OneClassClassificationDataset_no_label, PostClassificationDataset_no_label, DataExporter
from kobert_tokenizer import KoBERTTokenizer

"""
specify input xl file path, and output path
"""
ood_path = '/saved_model/occ_standard'
thm_path = '/saved_model/thm_standard'

input_path = sys.argv[1]
output_path = sys.argv[2]
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache() 
epsilon = sys.float_info.epsilon   

def predict_with_no_gate(epoch, model, device, loader):
    model.eval()
    generated_labels = []

    with torch.no_grad():
        for _, data in enumerate(loader, 0):

            ids  = data['input_text_ids'].to(device, dtype = torch.long)
            mask = data['input_mask'].to(device, dtype = torch.long)

            logits = model.forward(input_ids = ids, attention_mask = mask).logits
            predicted_class_id = torch.argmax(F.softmax(logits,dim=1), dim=1)

            generated_labels.extend([predicted_class_id])
        
    return generated_labels

def predict_with_gate(epoch, model, device, loader, gater = None, gater_list = None):
    model.eval()
    generated_labels = []

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            if not gater(gater_list[_]) :
                continue

            ids  = data['input_text_ids'].to(device, dtype = torch.long)
            mask = data['input_mask'].to(device, dtype = torch.long)

            logits = model.forward(input_ids = ids, attention_mask = mask).logits
            predicted_class_id = torch.argmax(F.softmax(logits,dim=1), dim=1)

            generated_labels.extend([predicted_class_id])
        
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
    wandb.init(project="Post Classification Final")
    
    config = wandb.config           # Initialize config
    config.TRAIN_BATCH_SIZE = 32    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1     # input batch size for testing (default: 1)
    config.TRAIN_EPOCHS =  50       # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1  
    config.LEARNING_RATE = 4.00e-05 # learning rate (default: 0.01)
    config.SEED = 420               # random seed (default: 42)
    config.MAX_LEN = 512
    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    torch.backends.cudnn.deterministic = True
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    dfdataset = df_load_from_path(input_file_path)
    ooddataset = OneClassClassificationDataset_no_label(dfdataset, tokenizer, config.MAX_LEN)
    thmdataset = PostClassificationDataset_no_label(dfdataset, tokenizer, config.MAX_LEN)
    ood_loader = DataLoader(ooddataset, **val_params)
    thm_loader = DataLoader(thmdataset, **val_params)

    ood_model = BertForSequenceClassification.from_pretrained(ood_path)
    thm_model = BertForSequenceClassification.from_pretrained(thm_path,
                                                          num_labels = 21,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    """
    First determine if the post is travel related or not
    """
    generated_is_trip_label = []
    for epoch in range(config.VAL_EPOCHS):
        generated_is_trip_label = predict_with_no_gate(epoch, ood_model, device, ood_loader)

    def is_trip_gater(is_a_trip_label):
        if is_a_trip_label == 0:
            return True
        else :
            return False

    """
    Extract tag information from post
    """
    generated_theme_label = []
    for epoch in range(config.VAL_EPOCHS):
        generated_theme_label = predict_with_gate(epoch, ood_model, device, ood_loader, is_trip_gater, generated_is_trip_label)

    """
    Export to file
    """
    myDataExpoerter = DataExporter()
    dfdataset.to_csv(dataset_name+'final_data.csv')
    
    
    
    print('Generated File')

    
if __name__ == '__main__':
    main(input_path, output_path)