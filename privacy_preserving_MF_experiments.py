import pickle
import time

import numpy as np
from funk_svd import SVD
from funk_svd.dataset import fetch_ml_ratings
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

df = fetch_ml_ratings(variant='100k')
train, test = train_test_split(df, test_size=0.2, random_state=42)

latent_vector_sizes = [3, 5, 10, 15, 30, 50, 100]
# latent_vector_sizes = [5, 15]
epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]
# epsilons = [0.5, 1]
sensitivity = 4
half_rating_sensitivity = 2

# baseline experiments
results_baseline = []
predictions_baseline = {}

# base + noise
results_base_noise_laplacian = {}
predictions_base_noise_laplacian = {}

# average of two parties
results_average_of_two_laplacian = {}
predictions_average_of_two_laplacian = {}

# plus minus trick
results_plus_minus_of_two_laplacian = {}
predictions_plus_minus_of_two_laplacian = {}

# additive shares with (noise, r-noise)
results_additive_n_r_laplacian = {}
predictions_additive_n_r_laplacian = {}

# additive shares with (r/2 + noise, r/2 - noise)
results_additive_n_2_laplacian = {}
predictions_additive_n_2_laplacian = {}

# Baseline
print("Baseline")
start = time.time()
for factors in latent_vector_sizes:
    svd_run = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                  shuffle=False, min_rating=1, max_rating=5)
    svd_run.fit(X=train, X_val=train)
    pred2 = svd_run.predict(test)

    # mae2 = mean_absolute_error(test['rating'], pred2)
    mae2 = mean_squared_error(test['rating'], pred2, squared=False)
    results_baseline.append(mae2)
    predictions_baseline[factors] = pred2

end = time.time()
print(f'Took {end - start:.1f} sec')

# Base + noise (Laplacian)
print("Base + noise (Laplacian)")
start = time.time()
for factors in latent_vector_sizes:
    for epsilon in epsilons:
        noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
        base = train.copy(deep=True)
        base.rating = base.rating + noise
        svd_run = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                      shuffle=False, min_rating=1, max_rating=5)
        svd_run.fit(X=base, X_val=base)
        pred2 = svd_run.predict(test)
        # mae2 = mean_absolute_error(test['rating'], pred2)
        mae2 = mean_squared_error(test['rating'], pred2, squared=False)
        if epsilon not in results_base_noise_laplacian.keys():
            results_base_noise_laplacian[epsilon] = []
        results_base_noise_laplacian[epsilon].append(mae2)
        predictions_base_noise_laplacian[(factors, epsilon)] = pred2
end = time.time()
print(f'Took {end - start:.1f} sec')

# Average of two parties
print("Average of two parties")
start = time.time()
for factors in latent_vector_sizes:
    for epsilon in epsilons:
        noise_1 = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
        noise_2 = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
        base_1 = train.copy(deep=True)
        base_1.rating = base_1.rating + noise_1
        base_2 = train.copy(deep=True)
        base_2.rating = base_2.rating + noise_2
        svd1 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1, max_rating=5)

        svd2 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1, max_rating=5)
        svd1.fit(X=base_1, X_val=base_1)
        svd2.fit(X=base_2, X_val=base_2)
        svd_real_deal = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                            shuffle=False, min_rating=1, max_rating=5)

        svd_real_deal.bi_ = (svd1.bi_ + svd2.bi_) / 2
        svd_real_deal.bu_ = (svd1.bu_ + svd2.bu_) / 2
        svd_real_deal.user_mapping_ = svd1.user_mapping_
        svd_real_deal.item_mapping_ = svd1.item_mapping_
        svd_real_deal.pu_ = (svd1.pu_ + svd2.pu_) / 2
        svd_real_deal.qi_ = (svd1.qi_ + svd2.qi_) / 2
        svd_real_deal.global_mean_ = (svd1.global_mean_ + svd2.global_mean_) / 2
        pred = svd_real_deal.predict(test)
        # mae = mean_absolute_error(test['rating'], pred)
        mae = mean_squared_error(test['rating'], pred, squared=False)
        if epsilon not in results_average_of_two_laplacian.keys():
            results_average_of_two_laplacian[epsilon] = []
        results_average_of_two_laplacian[epsilon].append(mae)
        predictions_average_of_two_laplacian[(factors, epsilon)] = pred
end = time.time()
print(f'Took {end - start:.1f} sec')

# Plus minus trick
print("Plus minus trick")
start = time.time()
for factors in latent_vector_sizes:
    for epsilon in epsilons:
        noise_1 = np.abs(np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0]))
        noise_2 = np.abs(np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0]))
        base_1 = train.copy(deep=True)
        base_1.rating = base_1.rating + noise_1
        base_2 = train.copy(deep=True)
        base_2.rating = base_2.rating - noise_2
        svd1 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1, max_rating=5)

        svd2 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1, max_rating=5)
        svd1.fit(X=base_1, X_val=base_1)
        svd2.fit(X=base_2, X_val=base_2)
        svd_real_deal = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                            shuffle=False, min_rating=1, max_rating=5)
        svd_real_deal.bi_ = (svd1.bi_ + svd2.bi_) / 2
        svd_real_deal.bu_ = (svd1.bu_ + svd2.bu_) / 2
        svd_real_deal.user_mapping_ = svd1.user_mapping_
        svd_real_deal.item_mapping_ = svd1.item_mapping_
        svd_real_deal.pu_ = (svd1.pu_ + svd2.pu_) / 2
        svd_real_deal.qi_ = (svd1.qi_ + svd2.qi_) / 2
        svd_real_deal.global_mean_ = (svd1.global_mean_ + svd2.global_mean_) / 2
        pred = svd_real_deal.predict(test)
        # mae = mean_absolute_error(test['rating'], pred)
        mae = mean_squared_error(test['rating'], pred, squared=False)
        if epsilon not in results_plus_minus_of_two_laplacian.keys():
            results_plus_minus_of_two_laplacian[epsilon] = []
        results_plus_minus_of_two_laplacian[epsilon].append(mae)
        predictions_plus_minus_of_two_laplacian[(factors, epsilon)] = pred
end = time.time()
print(f'Took {end - start:.1f} sec')

# additive shares with (noise, r-noise)
print("additive shares with (noise, r-noise)")
start = time.time()
for factors in latent_vector_sizes:
    for epsilon in epsilons:
        noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
        base_1 = train.copy(deep=True)
        base_1.rating = noise
        base_2 = train.copy(deep=True)
        base_2.rating = base_2.rating - noise
        svd1 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1, max_rating=5)
        svd2 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1, max_rating=5)
        svd1.fit(X=base_1, X_val=base_1)
        svd2.fit(X=base_2, X_val=base_2)
        svd_real_deal = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                            shuffle=False, min_rating=1, max_rating=5)
        svd_real_deal.bi_ = svd1.bi_ + svd2.bi_
        svd_real_deal.bu_ = svd1.bu_ + svd2.bu_
        svd_real_deal.user_mapping_ = svd1.user_mapping_
        svd_real_deal.item_mapping_ = svd1.item_mapping_
        svd_real_deal.pu_ = np.concatenate((svd1.pu_, svd2.pu_), axis=1)
        svd_real_deal.qi_ = np.concatenate((svd1.qi_, svd2.qi_), axis=1)
        svd_real_deal.global_mean_ = svd1.global_mean_ + svd2.global_mean_
        pred = svd_real_deal.predict(test)
        # mae = mean_absolute_error(test['rating'], pred)
        mae = mean_squared_error(test['rating'], pred, squared=False)
        if epsilon not in results_additive_n_r_laplacian.keys():
            results_additive_n_r_laplacian[epsilon] = []
        results_additive_n_r_laplacian[epsilon].append(mae)
        predictions_additive_n_r_laplacian[(factors, epsilon)] = pred
end = time.time()
print(f'Took {end - start:.1f} sec')

# additive shares with (r/2 + noise, r/2 - noise)
print("additive shares with (r/2 + noise, r/2 - noise)")
start = time.time()
for factors in latent_vector_sizes:
    for epsilon in epsilons:
        noise = np.abs(np.random.laplace(loc=0, scale=half_rating_sensitivity / epsilon, size=train.shape[0]))
        base_1 = train.copy(deep=True)
        base_1.rating = base_1.rating / 2 + noise
        base_2 = train.copy(deep=True)
        base_2.rating = base_2.rating / 2 - noise
        svd1 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1, max_rating=5)
        svd2 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1, max_rating=5)
        svd1.fit(X=base_1, X_val=base_1)
        svd2.fit(X=base_2, X_val=base_2)
        svd_real_deal = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
                            shuffle=False, min_rating=1, max_rating=5)
        svd_real_deal.bi_ = svd1.bi_ + svd2.bi_
        svd_real_deal.bu_ = svd1.bu_ + svd2.bu_
        svd_real_deal.user_mapping_ = svd1.user_mapping_
        svd_real_deal.item_mapping_ = svd1.item_mapping_
        svd_real_deal.pu_ = np.concatenate((svd1.pu_, svd2.pu_), axis=1)
        svd_real_deal.qi_ = np.concatenate((svd1.qi_, svd2.qi_), axis=1)
        svd_real_deal.global_mean_ = svd1.global_mean_ + svd2.global_mean_
        pred = svd_real_deal.predict(test)
        # mae = mean_absolute_error(test['rating'], pred)
        mae = mean_squared_error(test['rating'], pred, squared=False)
        if epsilon not in results_additive_n_2_laplacian.keys():
            results_additive_n_2_laplacian[epsilon] = []
        results_additive_n_2_laplacian[epsilon].append(mae)
        predictions_additive_n_2_laplacian[(factors, epsilon)] = pred
end = time.time()
print(f'Took {end - start:.1f} sec')

handle = open('results_rmse.pickle', 'wb')
pickle.dump(results_baseline, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(results_base_noise_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(results_average_of_two_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(results_plus_minus_of_two_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(results_additive_n_r_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(results_additive_n_2_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

handle = open('predictions.pickle', 'wb')
pickle.dump(predictions_baseline, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(predictions_base_noise_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(predictions_average_of_two_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(predictions_plus_minus_of_two_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(predictions_additive_n_r_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(predictions_additive_n_2_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

