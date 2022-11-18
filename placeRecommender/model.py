import numpy as np

class PlaceRecommender():
    def __init__(
        self,
        initial_taste_weight = 0.8,
        saved_spot_weight = 0.4,
    ):
        self.initial_taste_weight = initial_taste_weight
        self.saved_spot_weight = saved_spot_weight

    def _