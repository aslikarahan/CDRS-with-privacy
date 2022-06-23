import pickle

import numpy as np
from funk_svd.dataset import fetch_ml_ratings
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = fetch_ml_ratings(variant='100k')
train, test = train_test_split(df, test_size=0.2, random_state=42)

latent_vector_sizes = [3, 5, 10, 15, 30, 50, 100]
epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]
sensitivity = 4
half_rating_sensitivity = 2
# base + noise
adversary_predictions_base_noise_mae = {}
adversary_predictions_base_noise_rmse = {}

# average of two parties
adversary_predictions_average_of_two_mae_1 = {}
adversary_predictions_average_of_two_rmse_1 = {}

# plus minus trick
adversary_predictions_plus_minus_mae_1 = {}
adversary_predictions_plus_minus_rmse_1 = {}

# Base + noise (Laplacian)
print("Base + noise (Laplacian)")
# start = time.time()
for epsilon in epsilons:
    noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
    base = train.copy(deep=True)
    base.rating = base.rating + noise
    base.rating = np.clip(base.rating, 1, 5)
    mae = mean_absolute_error(train['rating'], base.rating)
    rmse = mean_squared_error(train['rating'], base.rating, squared=False)
    adversary_predictions_base_noise_mae[epsilon] = mae
    adversary_predictions_base_noise_rmse[epsilon] = rmse

# Average of two parties
print("Average of two parties")
for epsilon in epsilons:
    noise_1 = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
    noise_2 = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
    base_1 = train.copy(deep=True)
    base_1.rating = base_1.rating + noise_1
    base_2 = train.copy(deep=True)
    base_2.rating = base_2.rating + noise_2

    base_1.rating = np.clip((base_1.rating + base_2.rating) / 2, 1, 5)

    mae1 = mean_absolute_error(train['rating'], base_1.rating)
    rmse1 = mean_squared_error(train['rating'], base_1.rating, squared=False)

    adversary_predictions_average_of_two_mae_1[epsilon] = mae1
    adversary_predictions_average_of_two_rmse_1[epsilon] = rmse1

# Plus minus trick
print("Plus minus trick")
# start = time.time()
for epsilon in epsilons:
    noise_1 = np.abs(np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0]))
    noise_2 = np.abs(np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0]))
    base_1 = train.copy(deep=True)
    base_1.rating = base_1.rating + noise_1
    base_2 = train.copy(deep=True)
    base_2.rating = base_2.rating - noise_2

    base_1.rating = np.clip((base_1.rating + base_2.rating) / 2, 1, 5)

    mae1 = mean_absolute_error(train['rating'], base_1.rating)
    rmse1 = mean_squared_error(train['rating'], base_1.rating, squared=False)

    adversary_predictions_plus_minus_mae_1[epsilon] = mae1
    adversary_predictions_plus_minus_rmse_1[epsilon] = rmse1

# additive shares with (r/2 + noise, r/2 - noise)

handle = open('pickles/adversary_collaborating_mae.pickle', 'wb')
pickle.dump(adversary_predictions_base_noise_mae, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(adversary_predictions_average_of_two_mae_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(adversary_predictions_plus_minus_mae_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

handle = open('pickles/adversary_collaborating_rmse.pickle', 'wb')
pickle.dump(adversary_predictions_base_noise_rmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(adversary_predictions_average_of_two_rmse_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(adversary_predictions_plus_minus_rmse_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()
