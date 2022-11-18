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
from kobert_tokenizer import KoBERTTokenizer

from model import KoBERTforSequenceClassification
from utils import PostClassificationDataset

save_path = './saved_model/thm_' + datetime.now().strftime("%y%m%d_%H%M%S")
saved_path  = './saved_model/thm_'

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache() 
epsilon = sys.float_info.epsilon   

def train(
    epoch,
    tokenizer,
    model,
    device,
    loader,
    optimizer
):
    model.train()
    total_correct = 0
    total_len     = 0
    
    for _, data in enumerate(loader, 0):
        optimizer.zero_grad()
        
        y    = data['labels_y'].to(device, dtype = torch.long)
        ids  = data['input_text_ids'].to(device, dtype = torch.long)
        mask = data['input_mask'].to(device, dtype = torch.long)
        
        outputs = model.forward(input_ids = ids, attention_mask = mask, labels=y)
        
        loss = outputs[0]
        logits = outputs.logits
        pred  = torch.argmax(F.softmax(logits,dim=1),dim=1)
        
        correct = pred.eq(y)
        total_correct += correct.sum().item()
        total_len += len(y)
        loss.backward()
        optimizer.step()
        
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})
        
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
    
    print(f'Epoch: {epoch}, Accuracy: {total_correct/total_len:.3f}')

def validate(epoch, tokenizer, model, device, loader):
    model.eval()

    total_correct_test = 0
    
    tp = 0 
    fp = 0
    tn = 0
    fn = 0
    
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y    = data['labels_y'].to(device, dtype = torch.long)
            ids  = data['input_text_ids'].to(device, dtype = torch.long)
            mask = data['input_mask'].to(device, dtype = torch.long)
  
            logits = model.forward(input_ids = ids, attention_mask = mask).logits
            predicted_class_id = torch.argmax(F.softmax(logits,dim=1), dim=1)

            if predicted_class_id.item() == 1 :
                if predicted_class_id.item() == y.item() :
                    tp += 1
                else:
                    fp += 1
            else :
                if predicted_class_id.item() == y.item() :
                    tn += 1
                else:
                    fn += 1 
             
            if _%100==0:
                print(f'Completed {_}')
    
    precision = tp/(tp+fp+epsilon) + epsilon
    recall    = tp/(tp+fn+epsilon) + epsilon 
    f1        = 2/(1/precision+1/recall)
    acc       = (tp+tn)/len(loader)

    return precision, recall, f1, acc
    
def main():
    wandb.init(project="Post Theme")
    
    config = wandb.config           # Initialize config
    config.TRAIN_BATCH_SIZE = 32    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1     # input batch size for testing (default: 1)
    config.TRAIN_EPOCHS =  10       # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1  
    config.LEARNING_RATE = 4.00e-05 # learning rate (default: 0.01)
    config.SEED = 420               # random seed (default: 42)
    config.MAX_LEN = 512

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    torch.backends.cudnn.deterministic = True
    
    dfdataset  = pd.read_excel('testingData/instagram_post.xlsx')
    dftrainset = dfdataset.sample(frac=0.8,random_state=config.SEED)
    dftestset  = dfdataset.drop(dftrainset.index)
    dftrainset.reset_index(drop=True, inplace=True)
    dftestset.reset_index(drop=True, inplace=True)
    
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    training_set = PostClassificationDataset(dftrainset, tokenizer, config.MAX_LEN)
    test_set     = PostClassificationDataset(dftestset , tokenizer, config.MAX_LEN)
    
    train_loader = DataLoader(training_set, **train_params)
    test_loader  = DataLoader(test_set, **val_params)
    
    print(dftrainset.sample(10))
    print("TRAIN Dataset: {}".format(dftrainset.shape))
    
    model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1',
                                                          num_labels = 21,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)   
    
    for epoch in range(config.TRAIN_EPOCHS):
        print(str(epoch + 1), ': ',"Training")
        train(epoch, tokenizer, model, device, train_loader, optimizer)
        print(str(epoch + 1), ': ',"Validating")
        precision, recall, f1, acc = validate(epoch, tokenizer, model, device, test_loader)
        print(f"precision:{precision:.3f}\nrecall:{recall:.3f}\nf1:{f1:.3f}\nacc:{acc:.3f}")

    print("saving parameter")
    model.save_pretrained(save_path)
       
if __name__ == '__main__':
    main()