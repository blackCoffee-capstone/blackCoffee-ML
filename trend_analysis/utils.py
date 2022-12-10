import pandas as pd
import json

class Parser():
    def __init__(self):
        None

    def read_json_make_dataframe(
        self,
        input_path : str
    ) -> pd.DataFrame:
        
        with open(input_path, 'r') as file:
            input_data = json.load(file)

        dict_data = {
            "spot_id" : [],
            "week" : [],
            "like" : []
        }

        for spot_data in input_data:
            spot_id = spot_data["spotId"]
            sns_posts = spot_data['snsPosts'] 
            for sns_post in sns_posts:
                dict_data["spot_id"].append(spot_id)
                dict_data["week"].append(sns_post["week"])
                dict_data["like"].append(int(sns_post["likeNumber"]))
        
        return pd.DataFrame(dict_data)
