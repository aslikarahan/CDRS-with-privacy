import numpy as np
import random

class additive_scheme():

    def __init__(self, num_users, num_items, sparsity):
        self.num_users = num_users
        self.num_items = num_items
        self.sparsity = sparsity


    def generate_rating_matrix(self):
        array = np.random.randint(10, size=(self.num_users, self.num_items))
        self.sparsity = 1 - np.count_nonzero(array)/(self.num_users* self.num_items)
        return array

    def generate_additive_shares(self, main_array):
        share_array = np.random.uniform(size=(self.num_users, self.num_items))
        share_1 = np.multiply(main_array, share_array)
        share_2 = np.subtract(main_array, share_1)
        return share_1, share_2

    def combine_latent_vectors(self, user_latent_vector_1, user_latent_vector_2, item_latent_vector_1, item_latent_vector_2, d):
        order = random.shuffle(range(0,d))
        #
        # for i in order:
        #     if i > np.shape(user_latent_vector_1)[0]:






