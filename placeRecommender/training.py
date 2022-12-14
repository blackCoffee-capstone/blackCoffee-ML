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

#input_paths["liked"] = "./testingData/test_wish_spot.json"
over_write_standard = eval(sys.argv[5])
save_path = './saved_model/hybrid_' + datetime.now().strftime("%y%m%d_%H%M%S")
standard_path = './saved_model/hybrid_standard'

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
        min_max_normalized_rating = data['min_max_normalized_rating'].to(device, dtype = torch.float32)
        
        #print("shapes",user_id, user_feature)

        predicted_rating = model.forward(user_id,item_id,user_feature,item_feature)
        batch_size_out, hidden_feature_out = predicted_rating.shape
        predicted_rating = predicted_rating.view([batch_size_out])
        #print(predicted_rating, min_max_normalized_rating)
        loss = loss_function(predicted_rating, min_max_normalized_rating)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if _%500==0:
            print(f'Epoch: {epoch+1}, Loss:  {loss.item()}')

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
        batch_size_out, hidden_feature_out = predicted_rating.shape
        
        predicted_rating = predicted_rating.view([batch_size_out])
        loss = loss_function(predicted_rating, rating)
        total_loss += loss.item()
        if _%500==0:
            print(f'Epoch: {epoch+1}, Loss:  {loss.item()}')

    return total_loss/len(loader)

def main(input_paths, do_over_write_standard):

    wandb.init(project="hybridRec_blackcoffee")
    
    config = wandb.config           # Initialize config
    config.TRAIN_BATCH_SIZE = 1     # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1     # input batch size for testing (default: 1)
    config.TRAIN_EPOCHS =  65       # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1  
    config.LEARNING_RATE = 5.00e-05 # learning rate (default: 0.01)
    config.SEED = 423                # random seed (default: 42)

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

    print(f"# of User : {number_of_users} , #of Item : {number_of_items}")

    df_userTaste = df_userTaste.replace({"id" : user_map})
    df_spot      = df_spot.replace({"id" : item_map})
    df_visited   = df_visited.replace({"user_id" : user_map, "spot_id" : item_map})
    df_liked     = df_liked.replace({"user_id" : user_map, "spot_id" : item_map})

    dict_user_feature_map = {
            '???'           : 0,
            '?????????'       : 1,
            '??????'         : 2,
            '??????'         : 3,
            '??????'         : 4,
            '??????'         : 5,
            '??????'         : 6,
            '?????????'       : 7,
            '??????'         : 8,
            '??????'         : 9,
            '??????'         : 10,
            '??????'         : 11,
            '????????????'     : 12,
            '??????'         : 13,
            '????????????'     : 14,
            '??????'         : 15,
            '???'           : 16,
            '??????'         : 17,
            '?????????'       : 18,
            '??????'         : 19,
            'unknown_theme': 20
    }
    dict_item_feature_map = {
            '???'           : 0,
            '?????????'       : 1,
            '??????'         : 2,
            '??????'         : 3,
            '??????'         : 4,
            '??????'         : 5,
            '??????'         : 6,
            '?????????'       : 7,
            '??????'         : 8,
            '??????'         : 9,
            '??????'         : 10,
            '??????'         : 11,
            '????????????'     : 12,
            '??????'         : 13,
            '????????????'     : 14,
            '??????'         : 15,
            '???'           : 16,
            '??????'         : 17,
            '?????????'       : 18,
            '??????'         : 19,
            'unknown_theme': 20,
            '??????'         : 21,
            '??????'         : 22,
            '??????'         : 23,
            '??????'         : 24,
            '??????'         : 25,
            '??????'         : 26,
            '??????'         : 27,
            '??????'         : 28,
            '??????'         : 29,
            '??????'         : 30,
            '??????'         : 31,
            '??????'         : 32,
            '??????'         : 33,
            '??????'         : 34,
            '??????'         : 35,
            '??????'         : 36,
            '??????'         : 37,
            'unknown_loc'  : 38
    }
    user_feature_size = len(dict_user_feature_map)
    item_feature_size = len(dict_item_feature_map)

    
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

    max_user_id = int(HybridRecTrainDataset.max_user_id)
    max_item_id = int(HybridRecTrainDataset.max_item_id)

    model = HybridRecSystem(
        int(max_user_id+1),
        int(max_item_id+1),
        user_feature_size,
        item_feature_size,
        embedding_size = 50,
        n_hidden = 20
    )
    model.to(device)

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
    user_map_object.export_to_pickle(save_path)
    item_map_object.export_to_pickle(save_path)

    print("saved parameters, user and spot maps at :",save_path)
    
    if do_over_write_standard:
        model.save_trained(standard_path)
        user_map_object.export_to_pickle(standard_path)
        item_map_object.export_to_pickle(standard_path)
        print("saved parameters, user and spot maps at :",standard_path)
        

if __name__ == '__main__':
    main(input_paths, over_write_standard)