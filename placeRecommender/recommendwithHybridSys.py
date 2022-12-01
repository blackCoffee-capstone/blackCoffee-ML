import json
import sys
import numpy as np
import torch
from model import HybridRecSystem
from utils import HybridRecDataset, RecCandidateDataset, SpotMap, UserMap, TargetUser
from torch.utils.data import DataLoader, Dataset

input_path = sys.argv[1]
output_path = sys.argv[2]
target_id = int(sys.argv[3])
loading_model_path = sys.argv[4]
device = "cpu"
list_recommendation_length = 10

def recommendHybrid(
    model,
    device,
    loader
):
    model.eval()
    ## for padding item's rating
    ratings = torch.tensor([-1.0])

    for _,data in enumerate(loader, 0):
        
        user_id      = data['user_id'].to(device, dtype = torch.long)
        user_feature = data['user_feature'].to(device, dtype = torch.long)
        item_id      = data['item_id'].to(device, dtype = torch.long)
        item_feature = data['item_feature'].to(device, dtype = torch.long)

        predicted_rating = model.forward(user_id,item_id,user_feature,item_feature)
        
        ratings = torch.cat([ratings, predicted_rating], dim=0)

    return ratings

def valid(
    epoch,
    model,
    device,
    loader,
):
    model.eval()
    ## for padding item's rating
    ratings = torch.tensor([-1.0])

    for _,data in enumerate(loader, 0):
        
        user_id      = data['user_id'].to(device, dtype = torch.long)
        user_feature = data['user_feature'].to(device, dtype = torch.long)
        item_id      = data['item_id'].to(device, dtype = torch.long)
        item_feature = data['item_feature'].to(device, dtype = torch.long)

        predicted_rating = model.forward(user_id,item_id,user_feature,item_feature)
        
        ratings = torch.cat([ratings, predicted_rating], dim=0)

    return ratings

def main(input_file_path, output_file_path, model_path):

    ## Load user_map, item_map
    userMap_object = UserMap()
    userMap_object.load_from_pickle(model_path)
    spotMap_object = SpotMap()
    spotMap_object.load_from_pickle(model_path)

    ## GET target_user_id, user_feature from input_file
    ## If user_id not in map get the most simlar user id
    taraget_user_object = TargetUser()
    taraget_user_object.get_target_user_id_and_feature(input_file_path, UserMap, 0.8)  
    user_id = taraget_user_object.user_id
    user_feature = taraget_user_object.user_feature

    _params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }

    list_candidate_dataset = RecCandidateDataset()
    list_candidate_dataset.generate_rec_candidate(user_id, user_feature, spotMap_object)
    list_candidate_loader = DataLoader(list_candidate_dataset, **_params)

    model = HybridRecSystem()
    model.load_trained(model_path)

    ## Get Every Spots ratings index-0 rating is just padding with rating value -1
    every_predicted_ratings = recommendHybrid(model, device, list_candidate_loader)
    
    ## Use argsort and sort item_id by ratings item_id 0 is padding which will be located at last
    every_item_ordered_by_ratings = torch.argsort(every_predicted_ratings, descending=True)
    
    ## Remove Padding item_id
    sorted_items = every_item_ordered_by_ratings.tolist()
    sorted_items[:-1] 
    
    if len(sorted_items) > list_recommendation_length:
        list_rec = sorted_items[0:list_recommendation_length]
    else :
        list_rec = sorted_items

    map_rec = []
    ## TODO:
    ## GEO_NAMES
    geo_names = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']

    for geo_name in geo_names:
        geo_candidate_list = spotMap_object.filter_by_spot_features([geo_name])
        for item_id in sorted_items:
            if item_id in geo_candidate_list:
                map_rec.append(item_id)
                break
    
    
    dict_recommendations = {
        "listRecommendation" : list_rec,
        "mapRecommendation" : map_rec
    }

    print(dict_recommendations)
    ## Export Result from json
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(dict_recommendations, ensure_ascii=False).replace('None','null'))

    return

if __name__ == '__main__':
    main(input_path, output_path, loading_model_path)