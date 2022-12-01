import pandas as pd
import torch
import wandb
import sys
import os
from datetime import datetime
from model import HybridRecSystem
from utils import HybridRecDataset, UserMap, SpotMap
from torch.utils.data import DataLoader, Dataset
from torch import cuda

input_paths = {
    "user_taste" : sys.argv[1],
    "spot"       : sys.argv[2],
    "visited"    : sys.argv[3],
    "liked"      : sys.argv[4]
}
save_path = './saved_model/hybrid_' + datetime.now().strftime("%y%m%d_%H%M%S")

device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'
print(device)
torch.cuda.empty_cache() 
epsilon = sys.float_info.epsilon 

def train(
    epoch,
    model,
    device,
    loader,
    optimizer
):
    model.train()
    loss_function = torch.nn.MSELoss()
    for _,data in enumerate(loader, 0):
        
        user_id      = data['user_id'].to(device, dtype = torch.long)
        user_feature = data['user_feature'].to(device, dtype = torch.long)
        item_id      = data['item_id'].to(device, dtype = torch.long)
        item_feature = data['item_feature'].to(device, dtype = torch.long)
        rating       = data['rating'].to(device, dtype = torch.float32)

        predicted_rating = model.forward(user_id,item_id,user_feature,item_feature)
        loss = loss_function(predicted_rating, rating)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

def valid(
    epoch,
    model,
    device,
    loader,
):
    model.eval()
    total_loss = 0.0
    loss_function = torch.nn.MSELoss()
    for _,data in enumerate(loader, 0):
        
        user_id      = data['user_id'].to(device, dtype = torch.long)
        user_feature = data['user_feature'].to(device, dtype = torch.long)
        item_id      = data['item_id'].to(device, dtype = torch.long)
        item_feature = data['item_feature'].to(device, dtype = torch.long)
        rating       = data['rating'].to(device, dtype = torch.float32)

        predicted_rating = model.forward(user_id,item_id,user_feature,item_feature)
        loss = loss_function(predicted_rating, rating)
        total_loss += loss.item()
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

    return total_loss/len(loader)

def main(input_paths):

    wandb.init(project="hybridRec_blackcoffee")
    
    config = wandb.config           # Initialize config
    config.TRAIN_BATCH_SIZE = 2     # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1     # input batch size for testing (default: 1)
    config.TRAIN_EPOCHS =  10       # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1  
    config.LEARNING_RATE = 4.00e-05 # learning rate (default: 0.01)
    config.SEED = 420               # random seed (default: 42)

    ## LOAD Dataset
    df_userTaste = pd.read_json(input_paths["user_taste"])
    df_spot      = pd.read_json(input_paths["spot"])
    df_visited   = pd.read_json(input_paths["visited"])
    df_liked     = pd.read_json(input_paths["liked"])
    
    user_map_object = UserMap()
    user_map_object.from_dfuserTaste_make_map(df_userTaste)
    user_map        = user_map_object.user_map
    number_of_users = user_map_object.number_of_users
    
    item_map_object = SpotMap()
    item_map_object.from_dfspots_make_map(df_spot)
    item_map        = item_map_object.spot_map
    number_of_items = item_map_object.number_of_spots

    user_map_object.export_to_pickle(save_path)
    item_map_object.export_to_pickle(save_path)

    df_userTaste = df_userTaste.replace({"id" : user_map})
    df_spot      = df_spot.replace({"id" : item_map})
    df_visited   = df_visited.replace({"user_id" : user_map, "spot_id" : item_map})
    df_liked     = df_liked.replace({"user_id" : user_map, "spot_id" : item_map})

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
    }
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
            '경기'         : 29,
            '강원'         : 30,
            '충북'         : 31,
            '충남'         : 32,
            '전북'         : 33,
            '전남'         : 34,
            '경북'         : 35,
            '경남'         : 36,
            '제주'         : 37,
            'unknown_loc'  : 38
    }
    user_feature_size = len(dict_user_feature_map)
    item_feature_size = len(dict_item_feature_map)

    model = HybridRecSystem(
        number_of_users,
        number_of_items,
        user_feature_size,
        item_feature_size,
        embedding_size = 50,
        n_hidden = 20
    )
    model.to(device)
    #df_train_liked   = df_liked.sample(frac=0.8,random_state=200)
    #df_test_liked    = df_liked.drop(df_train_liked.index)
    #df_train_visited = df_visited.sample(frac=0.8,random_state=200)
    #df_test_visited  = df_visited.drop(df_train_visited.index)

    df_train_liked   = df_liked
    df_test_liked    = df_liked
    df_train_visited = df_visited
    df_test_visited  = df_visited
    
    #print(df_train_liked,df_test_liked,df_train_visited,df_test_visited)

    HybridRecTrainDataset = HybridRecDataset(
        df_userTaste = df_userTaste,
        df_spotFeature = df_spot,
        df_userlikedspot = df_train_liked,
        df_uservisitedspot = df_train_visited,
        dict_user_feature_map = dict_user_feature_map,
        dict_item_feature_map = dict_item_feature_map
    )

    HybridRecValidDataset = HybridRecDataset(
        df_userTaste = df_userTaste,
        df_spotFeature = df_spot,
        df_userlikedspot = df_test_liked,
        df_uservisitedspot = df_test_visited,
        dict_user_feature_map = dict_user_feature_map,
        dict_item_feature_map = dict_item_feature_map
    )
    
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
    HybridRecTrainLoader = DataLoader(HybridRecTrainDataset, **train_params)
    HybridRecValidLoader = DataLoader(HybridRecValidDataset, **train_params)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.TRAIN_EPOCHS):
        print(str(epoch + 1), ': ',"Training")
        train(epoch, model, device, HybridRecTrainLoader, optimizer)
        print(str(epoch + 1), ': ',"Validating")
        total_loss = valid(epoch, model, device, HybridRecValidLoader)
        print(f"total_valid_loss:{total_loss}")

    model.save_trained(save_path)

    new_model = HybridRecSystem()
    new_model.load_trained(save_path)

    print(model.number_of_item, new_model.number_of_item)

if __name__ == '__main__':
    main(input_paths)