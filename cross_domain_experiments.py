import pickle
import random
import time

import numpy as np
from funk_svd import SVD
from funk_svd.dataset import fetch_ml_ratings
from scipy import spatial
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

def similarity_per_new_user(interdomain, intradomain, user_ids):
    similarities = {}
    for user1 in user_ids:
        for user2 in user_ids:
            if user1 == user2:
                continue
            similarities[(user1, user2)] = intradomain[(user1, user2)]*interdomain[user2]
    return similarities

def predict_pair_with_cosine_without_predictions(new_user_id, item_id, similarity_pre_calculation, user_ids, target_ratings, target_svd, average_target, average_source):
    sum_similarity_abs = 0
    sum_rating_bias = 0
    for overlapping_user in user_ids:
        if overlapping_user == new_user_id:
            continue
        # sim = similarity_pre_calculation[(new_user_id, overlapping_user)]
        if  target_ratings.loc[(target_ratings['u_id'] == overlapping_user) & (target_ratings['i_id'] == item_id)].rating.values.size == 0:
            # r_i_v = target_svd.predict_pair(overlapping_user, item_id, clip=True)
            continue
        else:
            r_i_v = target_ratings.loc[(target_ratings['u_id'] == overlapping_user) & (target_ratings['i_id'] == item_id)].rating.values[0]

        sim = similarity_pre_calculation[(new_user_id, overlapping_user)]
        average_of_overlapping_user = average_target[overlapping_user]
        sum_similarity_abs = sum_similarity_abs + abs(sim)
        sum_rating_bias = sum_rating_bias + sim*(r_i_v-average_of_overlapping_user)

    average_of_new_user = average_source[new_user_id]
    if sum_similarity_abs == 0:
        pred = average_of_new_user
    else:
        pred = average_of_new_user + sum_rating_bias/sum_similarity_abs
    pred = 5 if pred > 5 else pred
    pred = 1 if pred < 1 else pred
    return pred

def predict_pair_with_cosine(new_user_id, item_id, similarity_pre_calculation, user_ids, target_ratings, target_svd, average_target, average_source):
    sum_similarity_abs = 0
    sum_rating_bias = 0
    for overlapping_user in user_ids:
        if overlapping_user == new_user_id:
            continue
        if  target_ratings.loc[(target_ratings['u_id'] == overlapping_user) & (target_ratings['i_id'] == item_id)].rating.values.size == 0:
            r_i_v = target_svd.predict_pair(overlapping_user, item_id, clip=True)
        else:
            r_i_v = target_ratings.loc[(target_ratings['u_id'] == overlapping_user) & (target_ratings['i_id'] == item_id)].rating.values[0]

        sim = similarity_pre_calculation[(new_user_id, overlapping_user)]
        average_of_overlapping_user = average_target[overlapping_user]
        sum_similarity_abs = sum_similarity_abs + abs(sim)
        sum_rating_bias = sum_rating_bias + sim*(r_i_v-average_of_overlapping_user)

    average_of_new_user = average_source[new_user_id]
    pred = average_of_new_user + sum_rating_bias/sum_similarity_abs
    pred = 5 if pred > 5 else pred
    pred = 1 if pred < 1 else pred

    return pred

def predict(X, similarity_pre_calculation, user_ids, target_ratings, target_svd, average_target, average_source):
    return [
        predict_pair_with_cosine(u_id, i_id, similarity_pre_calculation, user_ids, target_ratings, target_svd, average_target, average_source)
        for u_id, i_id in zip(X['u_id'], X['i_id'])
    ]

def preprocess(overlap_percentage):
    random.seed(a=42)
    # df = fetch_ml_ratings(variant='1m')
    df = fetch_ml_ratings(variant='100k')

    # Clean up items and users with not enough ratings
    minimum_number_or_ratings_item = 10
    minimum_number_or_ratings_user = 20

    df = df[~df.i_id.isin(df.i_id.value_counts()[df.i_id.value_counts()<minimum_number_or_ratings_item].index)]
    df = df[~df.u_id.isin(df.u_id.value_counts()[df.u_id.value_counts()<minimum_number_or_ratings_user].index)]
    sparsity_original = df.shape[0]/((df.i_id.unique().size)*(df.u_id.unique().size))

    # Initialize parameters
    number_or_users_per_domain = 450
    number_or_overlapping_users = int(number_or_users_per_domain*overlap_percentage)

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

    sparsity_source = source.shape[0]/((source.i_id.unique().size)*(source.u_id.unique().size))
    sparsity_target = target.shape[0]/((target.i_id.unique().size)*(target.u_id.unique().size))

    print(f"Original sparsity: {sparsity_original}")
    print(f"Source sparsity: {sparsity_source}")
    print(f"Target sparsity: {sparsity_target}")

    return source, target, overlapping_user_ids, overlapping_target, overlapping_source

start_regular = time.time()
latent_vector_sizes = [2, 4, 6, 10, 16, 30, 50, 100]
user_overlap_percentages = [0.05, 0.1, 0.2, 0.3]
maes= {}
rmses= {}
for percentage in user_overlap_percentages:
    source, target, overlapping_user_ids, overlapping_target, overlapping_source = preprocess(percentage)
    maes[percentage] = []
    rmses[percentage] = []
    for factor in latent_vector_sizes:
        # print(f"n_factors={factor*2}")
        svd_source = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factor, early_stopping=True,
                      shuffle=False, min_rating=1, max_rating=5)
        svd_source.fit(X=source, X_val=source)

        svd_target = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factor, early_stopping=True,
                         shuffle=False, min_rating=1, max_rating=5)
        svd_target.fit(X=target, X_val=target)

        # print("Trainings done, similarity calculation starting")
        intra_domain_similarity = intra_domain_similarity_calculation(svd_source, overlapping_user_ids)
        inter_domain_similarity = inter_domain_similarity_calculation(svd_source, svd_target, overlapping_user_ids)
        similarity_pre_calculation = similarity_per_new_user(inter_domain_similarity, intra_domain_similarity, overlapping_user_ids)

        # print("Similarity done, average calculation starting")
        source_averages = (source.groupby("u_id").mean("rating")).drop(labels="i_id", axis = 1).to_dict()['rating']
        target_averages = (target.groupby("u_id").mean("rating")).drop(labels="i_id", axis = 1).to_dict()['rating']

        # print("Averages done, prediction starting")
        start = time.time()
        results = predict(overlapping_target, similarity_pre_calculation, overlapping_user_ids, target, svd_target, target_averages, source_averages)
        end = time.time()
        print(f'Took {end - start:.1f} sec for test set size {overlapping_target.shape[0]}')
        # print(mean_absolute_error(overlapping_target['rating'], results))
        maes[percentage].append(mean_absolute_error(overlapping_target['rating'], results))
        rmses[percentage].append(mean_squared_error(overlapping_target['rating'], results, squared=False))
#
end_regular = time.time()
print(f'All regular one took {end_regular - start_regular:.1f} sec')
factor = 6
epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]
pp_maes= {}
pp_rmses= {}
start_pp = time.time()
for percentage in user_overlap_percentages:
    source, target, overlapping_user_ids, overlapping_target, overlapping_source = preprocess(percentage)
    pp_maes[percentage] = []
    pp_rmses[percentage] = []
    for epsilon in epsilons:
        # print("privacy preserving version")
        sensitivity = 4
        half_rating_sensitivity = 2

        noise = np.abs(np.random.laplace(loc=0, scale=half_rating_sensitivity / epsilon, size=source.shape[0]))
        source_1 = source.copy(deep=True)
        source_1.rating = source_1.rating / 2 + noise
        source_2 = source.copy(deep=True)
        source_2.rating = source_2.rating / 2 - noise

        source_svd_1 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factor, early_stopping=True,
                           shuffle=False, min_rating=1, max_rating=5)
        source_svd_2 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factor, early_stopping=True,
                           shuffle=False, min_rating=1, max_rating=5)

        source_svd_1.fit(X=source_1, X_val=source_1)
        source_svd_2.fit(X=source_2, X_val=source_2)


        svd_source_pp = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factor*2, early_stopping=True,
                            shuffle=False, min_rating=1, max_rating=5)

        svd_source_pp.bi_ = source_svd_1.bi_ + source_svd_2.bi_
        svd_source_pp.bu_ = source_svd_1.bu_ + source_svd_2.bu_
        svd_source_pp.user_mapping_ = source_svd_1.user_mapping_
        svd_source_pp.item_mapping_ = source_svd_1.item_mapping_
        svd_source_pp.pu_ = np.concatenate((source_svd_1.pu_, source_svd_2.pu_), axis=1)
        svd_source_pp.qi_ = np.concatenate((source_svd_1.qi_, source_svd_2.qi_), axis=1)
        svd_source_pp.global_mean_ = source_svd_1.global_mean_ + source_svd_2.global_mean_


        noise = np.abs(np.random.laplace(loc=0, scale=half_rating_sensitivity / epsilon, size=target.shape[0]))
        target_1 = target.copy(deep=True)
        target_1.rating = target_1.rating / 2 + noise
        target_2 = target.copy(deep=True)
        target_2.rating = target_2.rating / 2 - noise

        target_svd_1 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factor, early_stopping=True,
                           shuffle=False, min_rating=1, max_rating=5)
        target_svd_2 = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factor, early_stopping=True,
                           shuffle=False, min_rating=1, max_rating=5)

        target_svd_1.fit(X=target_1, X_val=target_1)
        target_svd_2.fit(X=target_2, X_val=target_2)

        svd_target_pp = SVD(lr=0.001, reg=0.01, n_epochs=1000, n_factors=factor*2, early_stopping=True,
                            shuffle=False, min_rating=1, max_rating=5)
        svd_target_pp.bi_ = target_svd_1.bi_ + target_svd_2.bi_
        svd_target_pp.bu_ = target_svd_1.bu_ + target_svd_2.bu_
        svd_target_pp.user_mapping_ = target_svd_1.user_mapping_
        svd_target_pp.item_mapping_ = target_svd_1.item_mapping_
        svd_target_pp.pu_ = np.concatenate((target_svd_1.pu_, target_svd_2.pu_), axis=1)
        svd_target_pp.qi_ = np.concatenate((target_svd_1.qi_, target_svd_2.qi_), axis=1)
        svd_target_pp.global_mean_ = target_svd_1.global_mean_ + target_svd_2.global_mean_

        # print("PP trainings done, pp similarity calculation starting")
        intra_domain_similarity_pp = intra_domain_similarity_calculation(svd_source_pp, overlapping_user_ids)
        inter_domain_similarity_pp = inter_domain_similarity_calculation(svd_source_pp, svd_target_pp, overlapping_user_ids)
        similarity_pre_calculation_pp = similarity_per_new_user(inter_domain_similarity_pp, intra_domain_similarity_pp, overlapping_user_ids)
        source_averages = (source.groupby("u_id").mean("rating")).drop(labels="i_id", axis = 1).to_dict()['rating']
        target_averages = (target.groupby("u_id").mean("rating")).drop(labels="i_id", axis = 1).to_dict()['rating']
        # print("PP prediction starting")

        start = time.time()
        results = predict(overlapping_target, similarity_pre_calculation_pp, overlapping_user_ids, target, svd_target_pp, target_averages, source_averages)
        end = time.time()
        print(f'Took {end - start:.1f} sec for test set size {overlapping_target.shape[0]}')
        pp_maes[percentage].append(mean_absolute_error(overlapping_target['rating'], results))
        pp_rmses[percentage].append(mean_squared_error(overlapping_target['rating'], results, squared=False))
        print(mean_absolute_error(overlapping_target['rating'], results))

end_pp = time.time()
print(f'All pp one took {end_pp - start_pp:.1f} sec')

handle = open('pickles/cdrs_mae.pickle', 'wb')
pickle.dump(maes, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(pp_maes, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

handle = open('pickles/cdrs_rmse.pickle', 'wb')
pickle.dump(rmses, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(pp_rmses, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()




