import json
import sys
import numpy as np
from model import PlaceRecommender
from utils import RecommendationDataset, newRecommendationDataset

input_path = sys.argv[1]
output_path = sys.argv[2]
target_id = int(sys.argv[3])

def main(input_file_path, output_file_path, target_id):
    ## Load Data from Json
    myRecommendationDataset = newRecommendationDataset()
    myRecommendationDataset.load_data_from_json(input_file_path)

    ## Load Recommender

    userNum = myRecommendationDataset.userNum
    spotNum = myRecommendationDataset.spotNum
    location_spot_matrix = np.transpose(np.array(myRecommendationDataset.get_spot_loc_matrix(), dtype=float))
    theme_spot_matrix = np.transpose(np.array(myRecommendationDataset.get_spot_theme_matrix(), dtype=float))
    user_theme_matrix = np.array(myRecommendationDataset.get_user_theme_matrix(), dtype=float)
    empty_user_item_matrix = np.zeros([userNum,spotNum],dtype=float)

    print(user_theme_matrix)

    recommender = PlaceRecommender(
        userNum,
        spotNum,
        user_theme_matrix,
        theme_spot_matrix,
        location_spot_matrix,
        empty_user_item_matrix,
        weighted_user_item_matrix= empty_user_item_matrix
    )
    
    mapped_target_id = myRecommendationDataset.get_user_id(target_id)

    recommender.gen_weight()
    print(recommender.weighted_user_item_matrix)
    #print(recommender.location_item_matrix)
    ## Create Recommender
    dict_recommendations = {
        "locationRecommendation" : {},
        "themeRecommendation" : {},
        "listRecommendation" : [],
        "mapRecommendation" : []
    }

    ## Run Recommendatoin,
    ## From Recommended item_ids generate list of spots (containing spot informations)
    reocmmendation_num_per_loc_criteria = 1
    reocmmendation_num_per_criteria = 10
    
    for loc, recommendations in recommender.recommend_by_loc(mapped_target_id, reocmmendation_num_per_loc_criteria).items():
        
        #print(recommendations)

        location_name = ""
        for name, index in myRecommendationDataset.loc_dict.items():
            if index == loc:
                location_name = name
        
        if location_name == 'Uknown' :
            continue
        
        spot_ids = myRecommendationDataset.from_npArray_get_spot_id(recommendations[0])
        dict_recommendations["locationRecommendation"][location_name] = spot_ids
        
        if spot_ids[0] not in dict_recommendations["mapRecommendation"]:
            dict_recommendations["mapRecommendation"].append(spot_ids[0])
        
    
    for theme, recommendations in recommender.recommend_by_theme(mapped_target_id, reocmmendation_num_per_criteria).items():
        
        print(recommendations)

        theme_name = ""
        for name, index in myRecommendationDataset.theme_dict.items():
            if index == theme:
                theme_name = name

        if theme_name == 'Uknown' :
            continue

        dict_recommendations["themeRecommendation"][theme_name] = myRecommendationDataset.from_npArray_get_spot_id(recommendations[0])

    dict_recommendations["listRecommendation"] = myRecommendationDataset.from_npArray_get_spot_id(recommender.recommend(mapped_target_id, reocmmendation_num_per_criteria)[0])

    print(dict_recommendations)
    ## Export Result from json
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(dict_recommendations, ensure_ascii=False).replace('None','null'))

    return


if __name__ == '__main__':
    main(input_path, output_path, target_id)