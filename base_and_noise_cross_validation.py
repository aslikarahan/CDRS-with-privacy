import pickle
import random
import time
import numpy as np
from funk_svd import SVD
from funk_svd.dataset import fetch_ml_ratings
from scipy import spatial
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def intra_domain_similarity_calculation(svd, user_ids):
    similarities = {}
    for user1 in user_ids:
        for user2 in user_ids:
            similarities[(user1, user2)] = 1-float(spatial.distance.cosine(svd.pu_[svd.user_mapping_[user1]], svd.pu_[svd.user_mapping_[user2]]))
    return similarities

def inter_domain_similarity_calculation(svd_source, svd_target, user_ids):
    similarities = {}
    for user in user_ids:
        similarities[user] = 1-float(spatial.distance.cosine(svd_source.pu_[svd_source.user_mapping_[user]], svd_target.pu_[svd_target.user_mapping_[user]]))
    return similarities

def predict_pair_with_cosine(new_user_id, item_id, similarity_dict, interdomain_coef, user_ids, target_ratings, target_svd, average_target, average_source):
    sum_similarity_abs = 0
    sum_rating_bias = 0
    for overlapping_user in user_ids:
        if overlapping_user == new_user_id:
            continue
        sim = similarity_dict[(new_user_id, overlapping_user)]*interdomain_coef[overlapping_user]
        sum_similarity_abs = sum_similarity_abs + abs(sim)
        if  target_ratings.loc[(target_ratings['u_id'] == overlapping_user) & (target_ratings['i_id'] == item_id)].rating.values.size == 0:
            r_i_v = target_svd.predict_pair(overlapping_user, item_id, clip=True)
        else:
            r_i_v = target_ratings.loc[(target_ratings['u_id'] == overlapping_user) & (target_ratings['i_id'] == item_id)].rating.values[0]
        average_of_overlapping_user = average_target[average_target.u_id ==overlapping_user].rating.values[0]
        sum_rating_bias = sum_rating_bias + sim*(r_i_v-average_of_overlapping_user)

    average_of_new_user = average_source[average_source.u_id ==new_user_id].rating.values[0]
    return average_of_new_user + sum_rating_bias/sum_similarity_abs


def get_test_set_for_new_user(target_dataframe, user_id):
    return target_dataframe.loc[(target_dataframe['u_id'] == user_id)]


random.seed(a=42)
df = fetch_ml_ratings(variant='100k')

# Clean up items and users with not enough ratings
minimum_number_or_ratings_item = 10
minimum_number_or_ratings_user = 20
df = df[~df.i_id.isin(df.i_id.value_counts()[df.i_id.value_counts()<minimum_number_or_ratings_item].index)]
df = df[~df.u_id.isin(df.u_id.value_counts()[df.u_id.value_counts()<minimum_number_or_ratings_user].index)]

# Initialize parameters
number_or_users_per_domain = 50
number_or_overlapping_users = 3

# Sample a number of users for the source domain
source_user_ids = random.sample(list(df.u_id.unique()), number_or_users_per_domain)
source = df[df.u_id.isin(source_user_ids)]

# Select a number of user to be the overlapping users
overlapping_user_ids = random.sample(list(source.u_id.unique()), number_or_overlapping_users)
# target_sampling_set = df[~df.u_id.isin(source_user_ids)]
# target_user_ids = random.sample(list(df[~df.u_id.isin(source_user_ids)].u_id.unique()), number_or_users_per_domain-number_or_overlapping_users)

# Select a number of non-overlapping users to be in the target domain
target_user_ids = random.sample(list(set(df.u_id.unique())-set(source_user_ids)), number_or_users_per_domain-number_or_overlapping_users)
target = df[df.u_id.isin(target_user_ids + overlapping_user_ids)]

overlapping_target = target[target.u_id.isin(overlapping_user_ids)]
overlapping_source = source[source.u_id.isin(overlapping_user_ids)]

# Initialize items to not overlap
source_item_ids = random.sample(list(overlapping_source.i_id.unique()), int(overlapping_source.i_id.unique().size/2))
target_item_ids =list(set(overlapping_source.i_id.unique())-set(source_item_ids))

target = target[target.i_id.isin(target_item_ids)]
source = source[source.i_id.isin(source_item_ids)]

overlapping_target = target[target.u_id.isin(overlapping_user_ids)]
overlapping_source = source[source.u_id.isin(overlapping_user_ids)]

non_overlapping_target = target[~target.u_id.isin(overlapping_user_ids)]
non_overlapping_source = source[~source.u_id.isin(overlapping_user_ids)]

if len(set(target.i_id.unique()).intersection(set(source.i_id.unique()))) == 0:
    print("No item overlap")

if set(overlapping_user_ids)==set(source.u_id.unique()).intersection(set(target.u_id.unique())):
    print(f"User overlap is correct, number of overlap is {len(set(source.u_id.unique()).intersection(set(target.u_id.unique())))}")

print(f"SOURCE: number of ratings: {source.shape[0]}, number of users {source.u_id.unique().size}, number of items {source.i_id.unique().size}")
print(f"TARGET: number of ratings: {target.shape[0]}, number of users {target.u_id.unique().size}, number of items {target.i_id.unique().size}")


sparsity_original = df.shape[0]/((df.i_id.unique().size)*(df.u_id.unique().size))
sparsity_source = source.shape[0]/((source.i_id.unique().size)*(source.u_id.unique().size))
sparsity_target = target.shape[0]/((target.i_id.unique().size)*(target.u_id.unique().size))

print(f"Original sparsity: {sparsity_original}")
print(f"Source sparsity: {sparsity_source}")
print(f"Target sparsity: {sparsity_target}")

factor = 2
svd_source = SVD(lr=0.001, reg=0.005, n_epochs=1000, n_factors=2, early_stopping=True,
              shuffle=False, min_rating=1, max_rating=5)
svd_source.fit(X=source, X_val=source)
intra_domain_similarity = intra_domain_similarity_calculation(svd_source, overlapping_user_ids)

svd_target = SVD(lr=0.001, reg=0.005, n_epochs=1000, n_factors=2, early_stopping=True,
                 shuffle=False, min_rating=1, max_rating=5)
svd_target.fit(X=target, X_val=target)

inter_domain_similarity = inter_domain_similarity_calculation(svd_source, svd_target, overlapping_user_ids)

source_averages = (source.groupby("u_id").mean("rating")).reset_index()
target_averages = (target.groupby("u_id").mean("rating")).reset_index()

# predict_pair_with_cosine(new_user_id, item_id, similarity_dict, interdomain_coef, user_ids, target_ratings, target_svd, average_target, average_source)

predict_pair_with_cosine(272, 14, intra_domain_similarity, inter_domain_similarity, overlapping_user_ids, target, svd_target, target_averages, source_averages)

latent_vector_sizes = [5, 15, 30, 50, 100]

# user_id = 851
# item_id = 146
#
#
#
# if  target.loc[(target['u_id'] == user_id) & (df['i_id'] == item_id)].rating.values.size == 0:
#     r_i_v = svd_target.predict_pair(user_id, item_id, clip=True)
# else:
#     r_i_v = target.loc[(target['u_id'] == 851) & (df['i_id'] == 147)].rating.values[0]
#
#
# latent_vector_sizes = [5, 15]
# # epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]
# # # epsilons = [0.5, 1]
# # sensitivity = 4
# half_rating_sensitivity = 2
#
# # baseline experiments
# results_baseline = []
# predictions_baseline = {}
#
# # base + noise
# results_base_noise_laplacian = {}
# predictions_base_noise_laplacian = {}
#
# # average of two parties
# results_average_of_two_laplacian = {}
# predictions_average_of_two_laplacian = {}
#
# # plus minus trick
# results_plus_minus_of_two_laplacian = {}
# predictions_plus_minus_of_two_laplacian = {}
#
# # additive shares with (noise, r-noise)
# results_additive_n_r_laplacian = {}
# predictions_additive_n_r_laplacian = {}
#
# # additive shares with (r/2 + noise, r/2 - noise)
# results_additive_n_2_laplacian = {}
# predictions_additive_n_2_laplacian = {}
#
# # Baseline
# print("Baseline")
# start = time.time()
# for factors in latent_vector_sizes:
#     svd_run = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                   shuffle=False, min_rating=1, max_rating=5)
#     svd_run.fit(X=train, X_val=train)
#     pred2 = svd_run.predict(test)
#
#     mae2 = mean_absolute_error(test['rating'], pred2)
#     results_baseline.append(mae2)
#     predictions_baseline[factors] = pred2
#
# end = time.time()
# print(f'Took {end - start:.1f} sec')
#
# # Base + noise (Laplacian)
# print("Base + noise (Laplacian)")
# start = time.time()
# for factors in latent_vector_sizes:
#     for epsilon in epsilons:
#         noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
#         base = train.copy(deep=True)
#         base.rating = base.rating + noise
#         svd_run = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                       shuffle=False, min_rating=1, max_rating=5)
#         svd_run.fit(X=base, X_val=base)
#         pred2 = svd_run.predict(test)
#         mae2 = mean_absolute_error(test['rating'], pred2)
#         if epsilon not in results_base_noise_laplacian.keys():
#             results_base_noise_laplacian[epsilon] = []
#         results_base_noise_laplacian[epsilon].append(mae2)
#         predictions_base_noise_laplacian[(factors, epsilon)] = pred2
# end = time.time()
# print(f'Took {end - start:.1f} sec')
#
# # Average of two parties
# print("Average of two parties")
# start = time.time()
# for factors in latent_vector_sizes:
#     for epsilon in epsilons:
#         noise_1 = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
#         noise_2 = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
#         base_1 = train.copy(deep=True)
#         base_1.rating = base_1.rating + noise_1
#         base_2 = train.copy(deep=True)
#         base_2.rating = base_2.rating + noise_2
#         svd1 = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                    shuffle=False, min_rating=1, max_rating=5)
#
#         svd2 = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                    shuffle=False, min_rating=1, max_rating=5)
#         svd1.fit(X=base_1, X_val=base_1)
#         svd2.fit(X=base_2, X_val=base_2)
#         svd_real_deal = SVD(lr=0.001, reg=0.005, n_epochs=1000, n_factors=factors, early_stopping=True,
#                             shuffle=False, min_rating=1, max_rating=5)
#
#         svd_real_deal.bi_ = (svd1.bi_ + svd2.bi_) / 2
#         svd_real_deal.bu_ = (svd1.bu_ + svd2.bu_) / 2
#         svd_real_deal.user_mapping_ = svd1.user_mapping_
#         svd_real_deal.item_mapping_ = svd1.item_mapping_
#         svd_real_deal.pu_ = (svd1.pu_ + svd2.pu_) / 2
#         svd_real_deal.qi_ = (svd1.qi_ + svd2.qi_) / 2
#         svd_real_deal.global_mean_ = (svd1.global_mean_ + svd2.global_mean_) / 2
#         pred = svd_real_deal.predict(test)
#         mae = mean_absolute_error(test['rating'], pred)
#         if epsilon not in results_average_of_two_laplacian.keys():
#             results_average_of_two_laplacian[epsilon] = []
#         results_average_of_two_laplacian[epsilon].append(mae)
#         predictions_average_of_two_laplacian[(factors, epsilon)] = pred
# end = time.time()
# print(f'Took {end - start:.1f} sec')
#
# # Plus minus trick
# print("Plus minus trick")
# start = time.time()
# for factors in latent_vector_sizes:
#     for epsilon in epsilons:
#         noise_1 = np.abs(np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0]))
#         noise_2 = np.abs(np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0]))
#         base_1 = train.copy(deep=True)
#         base_1.rating = base_1.rating + noise_1
#         base_2 = train.copy(deep=True)
#         base_2.rating = base_2.rating - noise_2
#         svd1 = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                    shuffle=False, min_rating=1, max_rating=5)
#
#         svd2 = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                    shuffle=False, min_rating=1, max_rating=5)
#         svd1.fit(X=base_1, X_val=base_1)
#         svd2.fit(X=base_2, X_val=base_2)
#         svd_real_deal = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                             shuffle=False, min_rating=1, max_rating=5)
#         svd_real_deal.bi_ = (svd1.bi_ + svd2.bi_) / 2
#         svd_real_deal.bu_ = (svd1.bu_ + svd2.bu_) / 2
#         svd_real_deal.user_mapping_ = svd1.user_mapping_
#         svd_real_deal.item_mapping_ = svd1.item_mapping_
#         svd_real_deal.pu_ = (svd1.pu_ + svd2.pu_) / 2
#         svd_real_deal.qi_ = (svd1.qi_ + svd2.qi_) / 2
#         svd_real_deal.global_mean_ = (svd1.global_mean_ + svd2.global_mean_) / 2
#         pred = svd_real_deal.predict(test)
#         mae = mean_absolute_error(test['rating'], pred)
#         if epsilon not in results_plus_minus_of_two_laplacian.keys():
#             results_plus_minus_of_two_laplacian[epsilon] = []
#         results_plus_minus_of_two_laplacian[epsilon].append(mae)
#         predictions_plus_minus_of_two_laplacian[(factors, epsilon)] = pred
# end = time.time()
# print(f'Took {end - start:.1f} sec')
#
# # additive shares with (noise, r-noise)
# print("additive shares with (noise, r-noise)")
# start = time.time()
# for factors in latent_vector_sizes:
#     for epsilon in epsilons:
#         noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=train.shape[0])
#         base_1 = train.copy(deep=True)
#         base_1.rating = noise
#         base_2 = train.copy(deep=True)
#         base_2.rating = base_2.rating - noise
#         svd1 = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                    shuffle=False, min_rating=1, max_rating=5)
#         svd2 = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                    shuffle=False, min_rating=1, max_rating=5)
#         svd1.fit(X=base_1, X_val=base_1)
#         svd2.fit(X=base_2, X_val=base_2)
#         svd_real_deal = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                             shuffle=False, min_rating=1, max_rating=5)
#         svd_real_deal.bi_ = svd1.bi_ + svd2.bi_
#         svd_real_deal.bu_ = svd1.bu_ + svd2.bu_
#         svd_real_deal.user_mapping_ = svd1.user_mapping_
#         svd_real_deal.item_mapping_ = svd1.item_mapping_
#         svd_real_deal.pu_ = np.concatenate((svd1.pu_, svd2.pu_), axis=1)
#         svd_real_deal.qi_ = np.concatenate((svd1.qi_, svd2.qi_), axis=1)
#         svd_real_deal.global_mean_ = svd1.global_mean_ + svd2.global_mean_
#         pred = svd_real_deal.predict(test)
#         mae = mean_absolute_error(test['rating'], pred)
#         if epsilon not in results_additive_n_r_laplacian.keys():
#             results_additive_n_r_laplacian[epsilon] = []
#         results_additive_n_r_laplacian[epsilon].append(mae)
#         predictions_additive_n_r_laplacian[(factors, epsilon)] = pred
# end = time.time()
# print(f'Took {end - start:.1f} sec')
#
# # additive shares with (r/2 + noise, r/2 - noise)
# print("additive shares with (r/2 + noise, r/2 - noise)")
# start = time.time()
# for factors in latent_vector_sizes:
#     for epsilon in epsilons:
#         noise = np.abs(np.random.laplace(loc=0, scale=half_rating_sensitivity / epsilon, size=train.shape[0]))
#         base_1 = train.copy(deep=True)
#         base_1.rating = base_1.rating / 2 + noise
#         base_2 = train.copy(deep=True)
#         base_2.rating = base_2.rating / 2 - noise
#         svd1 = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                    shuffle=False, min_rating=1, max_rating=5)
#         svd2 = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                    shuffle=False, min_rating=1, max_rating=5)
#         svd1.fit(X=base_1, X_val=base_1)
#         svd2.fit(X=base_2, X_val=base_2)
#         svd_real_deal = SVD(lr=0.01, reg=0.01, n_epochs=1000, n_factors=factors, early_stopping=True,
#                             shuffle=False, min_rating=1, max_rating=5)
#         svd_real_deal.bi_ = svd1.bi_ + svd2.bi_
#         svd_real_deal.bu_ = svd1.bu_ + svd2.bu_
#         svd_real_deal.user_mapping_ = svd1.user_mapping_
#         svd_real_deal.item_mapping_ = svd1.item_mapping_
#         svd_real_deal.pu_ = np.concatenate((svd1.pu_, svd2.pu_), axis=1)
#         svd_real_deal.qi_ = np.concatenate((svd1.qi_, svd2.qi_), axis=1)
#         svd_real_deal.global_mean_ = svd1.global_mean_ + svd2.global_mean_
#         pred = svd_real_deal.predict(test)
#         mae = mean_absolute_error(test['rating'], pred)
#         if epsilon not in results_additive_n_2_laplacian.keys():
#             results_additive_n_2_laplacian[epsilon] = []
#         results_additive_n_2_laplacian[epsilon].append(mae)
#         predictions_additive_n_2_laplacian[(factors, epsilon)] = pred
# end = time.time()
# print(f'Took {end - start:.1f} sec')
#
# handle = open('results.pickle', 'wb')
# pickle.dump(results_baseline, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(results_base_noise_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(results_average_of_two_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(results_plus_minus_of_two_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(results_additive_n_r_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(results_additive_n_2_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# handle.close()
#
# handle = open('predictions.pickle', 'wb')
# pickle.dump(predictions_baseline, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(predictions_base_noise_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(predictions_average_of_two_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(predictions_plus_minus_of_two_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(predictions_additive_n_r_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(predictions_additive_n_2_laplacian, handle, protocol=pickle.HIGHEST_PROTOCOL)
# handle.close()
