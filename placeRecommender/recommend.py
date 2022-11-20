import json
import sys
import numpy as np
from model import PlaceRecommender
from utils import RecommendationDataset

input_path = sys.argv[1]
output_path = sys.argv[2]

def main(input_file_path, output_file_path):
    ## Load Data from Json
    myRecommendationDataset = RecommendationDataset()
    myRecommendationDataset.load_data_from_json(input_file_path)

    ## Load Recommender
    userNum = myRecommendationDataset.userNum
    spotNum = myRecommendationDataset.spotNum
    location_spot_matrix = np.transpose(np.array(myRecommendationDataset.get_spot_loc_matrix(), dtype=float))
    theme_spot_matrix = np.transpose(np.array(myRecommendationDataset.get_spot_theme_matrix(), dtype=float))
    user_theme_matrix = np.array(myRecommendationDataset.get_user_theme_matrix(), dtype=float)
    empty_user_item_matrix = np.zeros([userNum,spotNum],dtype=float)

    #print(userNum,spotNum)
    #print(location_spot_matrix.shape)
    #print(myRecommendationDataset.user_taste)
    #print(myRecommendationDataset.spots[0]['RecommendationsSpotResponseDto']['themes'])
    #print(type(myRecommendationDataset.spots))
    #print(theme_spot_matrix.shape)
    #print(user_theme_matrix.shape)
    #print(empty_user_item_matrix.shape)

    recommender = PlaceRecommender(
        userNum,
        spotNum,
        user_theme_matrix,
        theme_spot_matrix,
        location_spot_matrix,
        empty_user_item_matrix,
        weighted_user_item_matrix= empty_user_item_matrix
    )
    taraget_user_id = 1

    recommender.gen_weight()
    #print(recommender.weighted_user_item_matrix)
    #print(recommender.location_item_matrix)
    ## Create Recommender
    dict_recommendations = {
        "locationRecommendation" : {},
        "themeRecommendation" : {}
    }
    
    ## Run Recommendatoin,
    ## From Recommended item_ids generate list of spots (containing spot informations)
    for loc, recommendations in recommender.recommend_by_loc(taraget_user_id).items():
        dict_recommendations["locationRecommendation"][loc] = myRecommendationDataset.from_npArray_get_spot(recommendations[0])
    
    for theme, recommendations in recommender.recommend_by_theme(taraget_user_id).items():
        dict_recommendations["themeRecommendation"][theme] = myRecommendationDataset.from_npArray_get_spot(recommendations[0])

    print(dict_recommendations)
    ## Export Result from json
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(dict_recommendations, ensure_ascii=False))

    return


if __name__ == '__main__':
    main(input_path, output_path)