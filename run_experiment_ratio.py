from funk_svd.dataset import fetch_ml_ratings
from funk_svd import SVD
import numpy as np

from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


df = fetch_ml_ratings(variant='100k')
latent_vector_sizes = [3, 5, 10, 15, 20, 30, 50, 100, 200, 300, 400, 500]
noise_range = [(0.01, 0.99), (-0.99, 0.99), (-2, 2)]

train = df.sample(frac=0.8, random_state=7)
share1 = train.copy(deep=True)
share2 = train.copy(deep=True)
test = df.drop(train.index.tolist())
results_shared = {}
results_baseline = {}
for range in noise_range:
    results_shared[range] = []
    results_baseline[range] = []
    for factors in latent_vector_sizes:
        noise = np.random.uniform(range[0], range[1], train.shape[0])
        while noise.any() == 0:
            noise = np.random.uniform(range[0], range[1], train.shape[0])

        share1.rating = noise * train.rating
        share2.rating = train.rating - share1.rating

        svd1 = SVD(lr=0.01, reg=0.005, n_epochs=1000, n_factors=factors, early_stopping=True,
                  shuffle=False, min_rating=1, max_rating=5)

        svd2 = SVD(lr=0.01, reg=0.005, n_epochs=1000, n_factors=factors, early_stopping=True,
                  shuffle=False, min_rating=1, max_rating=5)

        svd1.fit(X=share1, X_val= share1)
        svd2.fit(X=share2, X_val= share2)

        svd_combining= SVD(lr=0.001, reg=0.005, n_epochs=1000, n_factors=factors, early_stopping=False,
                           shuffle=False, min_rating=1, max_rating=5)

        svd_combining.bi_= svd1.bi_ + svd2.bi_
        svd_combining.bu_= svd1.bu_ + svd2.bu_
        svd_combining.user_mapping_ = svd1.user_mapping_
        svd_combining.item_mapping_ = svd1.item_mapping_
        svd_combining.pu_ = np.concatenate((svd1.pu_, svd2.pu_), axis = 1)
        svd_combining.qi_ = np.concatenate((svd1.qi_, svd2.qi_), axis = 1)
        svd_combining.global_mean_ = svd1.global_mean_ + svd2.global_mean_


        pred = svd_combining.predict(test)
        mae = mean_absolute_error(test['rating'], pred)
        results_shared[range].append(mae)

        # print(f'Test MAE: {mae:.2f}')
        svd_actual= SVD(lr=0.001, reg=0.005, n_epochs=1000, n_factors=100, early_stopping=True,
                        shuffle=False, min_rating=1, max_rating=5)
        svd_actual.fit(X=train, X_val = train)

        pred2 = svd_actual.predict(test)
        mae2 = mean_absolute_error(test['rating'], pred2)
        results_baseline[range].append(mae2)


print(results_shared)
print(results_baseline)

results_shared = {(0.01, 0.99): [0.8063942372156856, 0.8487368807852578, 0.9527608989883962, 1.0353314158359093, 1.0972820362870674, 1.1606209023026066, 1.2175078761008091, 1.0615549894596024, 0.9386437741924144, 0.9083042777695569, 0.8873066951290021, 0.883164682330976], (-0.99, 0.99): [0.9788087474057137, 1.071838208647227, 1.2439679224807683, 1.356558231741555, 1.4433842862338526, 1.527780287834923, 1.5647935554511183, 1.3421673247247026, 1.15959146585404, 1.098088138001825, 1.0555203920883545, 1.0284552687834148], (-2, 2): [1.2204913352253473, 1.3518215539722755, 1.538194017764615, 1.616939833433814, 1.672451108546449, 1.7393475187658525, 1.7799776168122814, 1.591349228159296, 1.4234210338236135, 1.3412195646633112, 1.294831170867124, 1.2505005768934327]}
results_baseline = {(0.01, 0.99): [0.800420471275871, 0.7959799548760997, 0.7990004698646187, 0.8016986891561931, 0.792708829689732, 0.8029468954011453, 0.7969000113772541, 0.799194005206248, 0.7986528506405118, 0.7963689397714294, 0.7968015075431194, 0.7926096531212399], (-0.99, 0.99): [0.8014126146911147, 0.8009638445431635, 0.8039609378528052, 0.8001103716777718, 0.8022819309311763, 0.7960946939353589, 0.7974788783960394, 0.8041941188892819, 0.8023792231028503, 0.7966506494750658, 0.8017050861943953, 0.7979323510959963], (-2, 2): [0.8021094990330447, 0.8015010299564589, 0.8017744112507671, 0.801722439102316, 0.8051469107494664, 0.8030260472360523, 0.799281152086249, 0.7984380338838845, 0.8006977493560833, 0.7957391681540518, 0.795863221838368, 0.8001049381447345]}



plt.title("Multiplicative share separation")
plt.plot(  latent_vector_sizes, results_shared[noise_range[0]], linewidth=1, marker =".", label= noise_range[0])
plt.plot( latent_vector_sizes, results_shared[noise_range[1]], linewidth=1, marker =".", label= noise_range[1])
plt.plot( latent_vector_sizes, results_shared[noise_range[2]], linewidth=1, marker =".", label= noise_range[2])
plt.plot(  latent_vector_sizes, results_baseline[noise_range[0]], linewidth=1, marker =".", label= 'no noise')

# plt.plot( 'x_values', 'y2_values', data=df, marker='', color='olive', linewidth=2)
# plt.plot( 'x_values', 'y3_values', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
#
# show legend
plt.legend()


# show graph
plt.show()


plt.title("Multiplicative share separation")
plt.plot(  latent_vector_sizes, results_shared[noise_range[0]], linewidth=1, marker =".", label= noise_range[0])
plt.plot( latent_vector_sizes, results_shared[noise_range[1]], linewidth=1, marker =".", label= noise_range[1])
plt.plot( latent_vector_sizes, results_shared[noise_range[2]], linewidth=1, marker =".", label= noise_range[2])
plt.plot(  latent_vector_sizes, results_baseline[noise_range[0]], linewidth=1, marker =".", label= 'no noise')

#
# show legend
plt.ylim(0, 1.8)

plt.legend()

# show graph
plt.show()




