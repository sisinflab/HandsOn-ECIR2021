"""
Created on April 1, 2020
Define Recommender Model.
@author Felice Antonio Merra (felice.merra@poliba.it)
"""
from abc import ABC

import tensorflow as tf


class RecommenderModel(tf.keras.Model, ABC):
    """
    This class represents a recommender model.
    You can load a pretrained model by specifying its ckpt path and use it for training/testing purposes.
    """

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, rec):
        self.rec = rec
        self.data = data
        self.num_items = data.num_items
        self.num_users = data.num_users
        self.path_output_rec_result = path_output_rec_result
        self.path_output_rec_weight = path_output_rec_weight

    def train(self):
        pass