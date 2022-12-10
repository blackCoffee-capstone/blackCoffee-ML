
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import sys
import copy
import datetime
## pip install pyscopg2-binary

class Analyzer():
    def __init__(
        self,
        df_data : pd.DataFrame
    ) -> None:
        self.data = df_data
        self.first_week = df_data.min()["week"]
        self.list_of_weeks = sorted(self.data.week.unique())

    def _get_spots_weekly_likes(
        self,
        week : str
    ) -> pd.DataFrame:
        
        weekly_data = self.data.loc[self.data["week"] == week]
        
        return weekly_data

    def _process_weekly_likes(
        self,
        weekly_statistics : dict,
        week_index : int,
        weekly_likes : pd.DataFrame
    ) -> dict:


        for row in weekly_likes:
            spot_name = row[0]
            spot_id   = row[1]
            likes     = row[2]

            if spot_id in weekly_statistics:
                weekly_statistics[spot_id][week_index] = likes

            else : 
                weekly_statistics[spot_id] = {
                    week_index : likes
                }

        return weekly_statistics

    def get_weekly_statistics(self) -> pd.Series:  
        
        weekly_statistics =  self.data.groupby(["spot_id","week"], group_keys=False)['like'].sum()
        
        return weekly_statistics
    

    def sort_spot_id_by_likes(
        self,
        week : str,
    ) -> dict:

        
        weekly_statistics =  self._get_spots_weekly_likes(week).groupby(["spot_id","week"], group_keys=False)['like'].sum()
        list_of_weeks    = self.list_of_weeks
        list_of_spot_ids = self.data.spot_id.unique()
        
        spots_like = {}
        for spot_and_week, like in weekly_statistics.to_dict().items():
            spot_id = spot_and_week[0]
            spots_like[spot_id] = like
        
        sorted_spots_like = dict(sorted(spots_like.items(), key=lambda item: item[1], reverse=True))

        sorted_spot_ids_with_rank = []
        
        rank = 1
        for spot_id in sorted_spots_like:
            sorted_spot_ids_with_rank.append({
                "spotId": int(spot_id),
                "rank" : rank
            })
            rank = rank + 1
        
        return sorted_spot_ids_with_rank

            

    def _fill_in_spot_weekly_trends(
        self,
        spot_weekly_trends : pd.Series
    ) -> pd.Series:
        list_of_weeks = self.list_of_weeks
        
        copy_trend = spot_weekly_trends.copy()
        
        for week in list_of_weeks:
            if week not in spot_weekly_trends:
                copy_trend[week] = 0
        
        return copy_trend.sort_index()


    def _from_previous_linear_regression_get_err(
        self,
        spot_weekly_trends : pd.Series,
        last_week_index : int
    ) -> LinearRegression:
        
        list_of_weeks = self.list_of_weeks
        X = np.array(list(range(1, len(list_of_weeks)+1))[:last_week_index])
        X = X.reshape(-1, 1)
        y = spot_weekly_trends.iloc[:last_week_index]
        
        try:
            y_true = spot_weekly_trends[list_of_weeks[last_week_index]]
        except:
            print("err!",spot_weekly_trends, last_week_index, self.list_of_weeks)
            y_true = 0


        model = LinearRegression()
        model.fit(X = X, y = y)

        return y_true - model.predict([[len(list_of_weeks)]])

    def sort_spot_id_by_buzz(
        self,
        week : str,
    ) -> dict:
        
        last_week_index = self.list_of_weeks.index(week)

        weekly_statistics = self.get_weekly_statistics()
        list_of_weeks    = self.list_of_weeks
        list_of_spot_ids = self.data.spot_id.unique()

        prediction_err = {}

        for spot_id in list_of_spot_ids:
            spot_weekly_trend = weekly_statistics[spot_id]
            spot_weekly_trend = self._fill_in_spot_weekly_trends(spot_weekly_trend)
            prediction_err[spot_id] = self._from_previous_linear_regression_get_err(spot_weekly_trend, last_week_index)
        
        sorted_prediction_err = dict(sorted(prediction_err.items(), key=lambda item: item[1], reverse=True))
        
       

        sorted_spot_ids_with_rank = []
        
        rank = 1
        for spot_id in sorted_prediction_err:
            sorted_spot_ids_with_rank.append({
                "spotId": int(spot_id),
                "rank" : rank
            })
            rank = rank + 1
        
        return sorted_spot_ids_with_rank

    def calculate_trend_rank(
        self,
        week : str,
    ):  
        list_of_weeks    = self.list_of_weeks
        week_index= self.list_of_weeks.index(week)
        
        if week_index == 0:
            ## Not Enough data for actual trend analysis
            ## Sort Rank by like number
            sorted_spot_ids_with_rank = self.sort_spot_id_by_likes(week)

        elif week_index == 1:
            ## Not Enough Data for Linear Regression 
            ## Sort Rank by Like difference
            # sorted_spot_ids_with_rank = self.sort_spot_id_by_likes_difference(week)
            sorted_spot_ids_with_rank = self.sort_spot_id_by_likes(week)
        
        
        elif week_index  >= 2:
            sorted_spot_ids_with_rank = self.sort_spot_id_by_buzz(week)

        return sorted_spot_ids_with_rank
