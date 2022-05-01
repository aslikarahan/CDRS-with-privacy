import random

from funk_svd.dataset import fetch_ml_ratings
from funk_svd import SVD
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

def add_random_noise(x, noise_range):
    noise = random.choice([i for i in range((-1)*noise_range, noise_range) if i not in [0, x]])
    # noise = random.randint(-5, 5)
    # while noise == 0 or noise == x:
    #     noise = random.randint(-5, 5)
    return noise, x-noise

df = fetch_ml_ratings(variant='100k')
train = df.sample(frac=0.8, random_state=7)
share1 = train.copy(deep=True)
share2 = train.copy(deep=True)
test = df.drop(train.index.tolist())
# noise_range = 5
# noise_generator = lambda x: random.choice([i for i in range((-1)*noise_range, noise_range) if i not in [0, x]])

# latent_vector_sizes = [3, 5, 10, 15, 20, 30, 50, 100, 200, 300, 400, 500]
latent_vector_sizes = [5, 15, 30, 50, 100]
# noise_range = [1, 2, 3, 5, 10, 30, 100, 200]
noise_range = [1, 2, 3, 5, 10, 30, 100]

results_shared = {}
results_baseline = {}
predicted_ratings_shared = {}
predicted_ratings_baseline = {}
histogram_latent_value = 100
for range in noise_range:
    results_shared[range] = []
    results_baseline[range] = []
    for factors in latent_vector_sizes:

        noise = np.random.randint(-1 * range, range, train.shape[0])
        share1.rating = noise
        share2.rating = train.rating - noise

        svd1 = SVD(lr=0.001, reg=0.005, n_epochs=10000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1-range, max_rating=5+range)

        svd2 = SVD(lr=0.001, reg=0.005, n_epochs=10000, n_factors=factors, early_stopping=True,
                   shuffle=False, min_rating=1-range, max_rating=5+range)

        svd1.fit(X=share1, X_val=share1)
        svd2.fit(X=share2, X_val=share2)

        svd_real_deal = SVD(lr=0.001, reg=0.005, n_epochs=1000, n_factors=factors, early_stopping=True,
                            shuffle=False, min_rating=1, max_rating=5)

        svd_real_deal.bi_ = svd1.bi_ + svd2.bi_
        svd_real_deal.bu_ = svd1.bu_ + svd2.bu_
        svd_real_deal.user_mapping_ = svd1.user_mapping_
        svd_real_deal.item_mapping_ = svd1.item_mapping_
        svd_real_deal.pu_ = np.concatenate((svd1.pu_, svd2.pu_), axis=1)
        svd_real_deal.qi_ = np.concatenate((svd1.qi_, svd2.qi_), axis=1)
        svd_real_deal.global_mean_ = svd1.global_mean_ + svd2.global_mean_

        pred = svd_real_deal.predict(test)

        mae = mean_absolute_error(test['rating'], pred)

        results_shared[range].append(mae)
        if factors == histogram_latent_value:
            predicted_ratings_shared[range] = pred

        print(results_shared)

for factors in latent_vector_sizes:
    svd_run = SVD(lr=0.001, reg=0.005, n_epochs=1000, n_factors=factors, early_stopping=True,
                  shuffle=False, min_rating=1, max_rating=5)
    svd_run.fit(X=train, X_val=train)

    pred2 = svd_run.predict(test)
    mae2 = mean_absolute_error(test['rating'], pred2)
    if factors == histogram_latent_value:
        predicted_ratings_baseline[range] = pred2

    # print(f'without sharing MAE: {mae2:.2f}')
    results_baseline[range].append(mae2)

print(results_shared)
print(results_baseline)



# results_shared = {1: [0.7557329237658681, 0.7548398667363889, 0.7541647448582807, 0.7532857851448758, 0.7962368631583806, 0.8249769371445125, 0.8598997023794765, 0.8541888069025799, 0.8304514853429271, 0.8168501745059531, 0.8206944832281341, 0.8185279062905945], 2: [0.7595606514688434, 0.757491980419039, 0.8372769219457372, 0.8942812489683847, 0.949261889097706, 1.0151074842270857, 1.1307648712773615, 1.0753013389720187, 0.9784571796644216, 0.9373770312284785, 0.9119302526487477, 0.9088357554267655], 3: [0.7590870510126799, 0.8532761480475097, 0.9707014894491324, 1.0605189017362946, 1.232141287982531, 1.3130113652685746, 1.351281753093395, 1.24981401877961, 1.1101941186829596, 1.0410692153021421, 1.020679807120775, 0.9931760191288916], 5: [0.9278071971431973, 1.1610514034955068, 1.3580044162312508, 1.4580844109744568, 1.5261326436162104, 1.5965523190186157, 1.6030320530047923, 1.4871072759203605, 1.314611884388565, 1.2447836682909919, 1.1902401525739046, 1.1575886560053097], 10: [1.384954122995591, 1.5305008001579359, 1.6538087847859833, 1.7350019294862493, 1.7720503093679147, 1.810893719321367, 1.817368166464803, 1.7059512849795866, 1.5642452328768817, 1.5038733689239792, 1.460410552707563, 1.4217032970468724], 30: [1.7660978986559999, 1.8101480047882188, 1.8975884224497055, 1.898116121224231, 1.916407027858568, 1.9476259359540309, 1.9432362728919936, 1.895914272309259, 1.8231635730714835, 1.7803183390178707, 1.7673653335490085, 1.743258847787133], 100: [1.922154574716528, 1.946941159840945, 1.9488305108852648, 1.969493830379458, 1.9828558944659043, 1.9950821957442055, 1.985575920144259, 1.9669743694271455, 1.9495015609450679, 1.927999050895325, 1.9203306787057064, 1.9022605333017022], 200: [1.9526403433137824, 1.9634113782693021, 1.9908904177992328, 1.975361632189693, 1.9817837438282648, 1.9860755821906004, 1.9834474147584629, 1.9699628322665652, 1.9673072790240989, 1.9879553180625444, 1.9658511260815092, 1.9722428913983163]}
# results_baseline = {1: [], 2: [], 3: [], 5: [], 10: [], 30: [], 100: [], 200: [0.75451564338991, 0.7543289700187106, 0.7537679110115801, 0.7519553321659956, 0.7494709824214703, 0.7782355632923809, 0.79099449942283, 0.8021651913920186, 0.7826705429691674, 0.7802581208667567, 0.7774695896107505, 0.7784534009472087]}

plt.title("Additive share separation")
for range in noise_range:
    plt.plot(  latent_vector_sizes, results_shared[range], linewidth=1, marker =".", label= range)


plt.plot(  latent_vector_sizes, results_baseline[noise_range[noise_range.__len__()-1]], linewidth=1, marker =".", label= 'no noise')

#
# show legend
plt.ylim(0, 2.1)

plt.legend()

# show graph
plt.show()


plt.title("Additive share separation")
for range in noise_range:
    plt.plot(  latent_vector_sizes, results_shared[range], linewidth=1, marker =".", label= range)


plt.plot(  latent_vector_sizes, results_baseline[noise_range[noise_range.__len__()-1]], linewidth=1, marker =".", label= 'no noise')

#
# show legend

plt.legend()

# show graph
plt.show()

for range in noise_range:
    bins = np.linspace(-1, 6, 100)
    plt.title("Noise range {}".format(range))

    plt.hist(predicted_ratings_shared[range], bins, alpha=0.5, label='noisy shared pred')
    plt.hist(predicted_ratings_baseline[noise_range[noise_range.__len__()-1]], bins, alpha=0.5, label='baseline prediction')
    plt.hist(test['rating'], bins, alpha=0.5, label='real ratings')
    plt.legend(loc='upper right')
    plt.show()