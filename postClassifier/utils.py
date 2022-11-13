from preprocessor import PreProcessor
import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.utils.data import DataLoader, Dataset

class OneClassClassificationDataset(Dataset):

    def __init__(
        self, 
        dataframe, 
        tokenizer, 
        input_max_len,
        labels_dict = {'__label1__' : 0, '__label2__' : 1}, 
    ):
        self.tokenizer  = tokenizer 
        self.data       = dataframe
        self.input_text = self.data['REVIEW_TEXT']
        self.labels     = self.data['LABEL']
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
        input_text = PreProcessor(input_text)
        input_text = ' '.join(input_text.split())
        input_text = self.tokenizer.encode_plus(input_text)

        input_text_ids = input_text['input_ids'].squeeze()
        input_mask     = input_text['attention_mask']
        
        labels_y       = self.labels[index]
        labels_y       = self.labels_dict[labels_y]
        labels_y       = torch.tensor([labels_y])
        
        return {
            'input_text_ids' : input_text_ids.to(dtype=torch.long),
            'input_mask'     : input_mask.to(dtype=torch.long),
            'labels_y'       : labels_y.to(dtype=torch.long)
        }

class DataExporter():
    """
        Export Datas 
    """
    
    def __init__(
        self,
        num_workers
    ):
        self.num_workers = num_workers

        

    
