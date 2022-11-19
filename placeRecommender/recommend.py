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
    recommender = PlaceRecommender()
    taraget_user_id = 1

    ## Create Recommender
    every_spot_ndarray = recommender.

    dict_recommendations = {
        "locationRecommendation" : {},
        "themeRecommendation" : {}
    }

    ## Run Recommendatoin,
    ## From Recommended item_ids generate list of spots (containing spot informations)
    for loc, recommendations in recommender.recommend_by_loc(taraget_user_id).items():
        dict_recommendations["locationRecommendation"][loc] = myRecommendationDataset.from_npArray_get_spot(recommendations)
    
    for theme, recommendations in recommender.recommend_by_theme(taraget_user_id).items():
        dict_recommendations["themeRecommendation"][theme] = myRecommendationDataset.from_npArray_get_spot(recommendations)

    ## Export Result from json
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dumps(dict_recommendations, ensure_ascii=False)

    return


if __name__ == '__main__':
    main(input_path, output_path)