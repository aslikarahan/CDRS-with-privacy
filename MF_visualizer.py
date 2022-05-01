import pickle

from funk_svd.dataset import fetch_ml_ratings
from matplotlib import pyplot as plt
import numpy as np

# latent_vector_sizes = [5, 15, 30, 50, 100, 200]
from sklearn.model_selection import train_test_split

df = fetch_ml_ratings(variant='100k')

train, test = train_test_split(df, test_size=0.2, random_state=42)

latent_vector_sizes = [3, 5, 10, 15, 30, 50, 100]
noise_range = [1, 2, 3, 5, 10, 30, 100]
epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]

handle = open('results_mae.pickle', 'rb')
results_baseline = pickle.load(handle)
results_base_noise_laplacian = pickle.load(handle)
results_average_of_two_laplacian = pickle.load(handle)
results_plus_minus_of_two_laplacian = pickle.load(handle)
results_additive_n_r_laplacian = pickle.load(handle)
results_additive_n_2_laplacian = pickle.load(handle)
handle.close()

handle = open('predictions.pickle', 'rb')
predictions_baseline = pickle.load(handle)
predictions_base_noise_laplacian = pickle.load(handle)
predictions_average_of_two_laplacian = pickle.load(handle)
predictions_plus_minus_of_two_laplacian = pickle.load(handle)
predictions_additive_n_r_laplacian = pickle.load(handle)
predictions_additive_n_2_laplacian = pickle.load(handle)
handle.close()

plt.figure(figsize=(15, 15), dpi=80)
plt.xlabel('Latent Vector Size')
plt.title('Increasing the Redundancy')
plt.ylabel('MAE')
plt.xticks(latent_vector_sizes)

for epsilon in epsilons:
    plt.plot(latent_vector_sizes, results_base_noise_laplacian[epsilon], linewidth=1, marker =".", label= "Base and noise: {:.1f}".format(epsilon))
    plt.plot(latent_vector_sizes, results_average_of_two_laplacian[epsilon], linewidth=1, marker =".", label= "Average of two and noise: {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_plus_minus_of_two_laplacian[epsilon], linewidth=1, marker =".", label= "Plus minus trick noise: {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_r_laplacian[epsilon], linewidth=1, marker ="v", label= "(noise, r-noise): {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_2_laplacian[epsilon], linewidth=1, marker ="s", label= "(r/2 + noise, r/2 - noise): {:.1f}".format(epsilon))
plt.plot(  latent_vector_sizes, results_baseline, linewidth=1, marker =".", label= 'no noise')

#
# show legend
plt.ylim(0, 2.1)

plt.legend()

# show graph
plt.show()

plt.figure(figsize=(15, 15), dpi=150)
plt.xlabel('Epsilons',fontsize=18)
bar_chart_factor = 2

plt.title('Errors with Latent Vector Size: {}'.format(latent_vector_sizes[bar_chart_factor]))
plt.ylabel('MAE',fontsize=18)



ind = np.arange(len(epsilons))
width = 0.1       # the width of the bars

epsilon_strings = [str(epsilon) for epsilon in epsilons]

plt.bar(ind-2*width, results_baseline[bar_chart_factor], label= "No noise", width=width)
plt.bar(ind-width, [results_base_noise_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "Base and noise", width=width)
plt.bar(ind, [results_average_of_two_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "Average of two", width=width)
plt.bar(ind+width, [results_plus_minus_of_two_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "Plus minus trick", width=width)
plt.bar(ind+2*width, [results_additive_n_r_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "(noise, r-noise)", width=width)
plt.bar(ind+3*width, [results_additive_n_2_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "(r/2 + noise, r/2 - noise)", width=width)


plt.xticks(ind + width / 6, epsilon_strings)


# show legend
plt.ylim(0, 2.1)

plt.legend(prop={'size': 18})

# show graph
plt.show()

#
# histogram_factor = 15
# for epsilon in epsilons:
#     bins = np.linspace(-1, 6, 100)
#     plt.title("Epsilon {:.1f}".format(epsilon))
#     plt.hist(predictions_baseline[histogram_factor], bins, alpha=0.5, label= "No noise")
#     # plt.hist(predictions_base_noise_laplacian[(histogram_factor, epsilon)], bins, alpha=0.5, label= "Base and noise")
#     # plt.hist(predictions_average_of_two_laplacian[(histogram_factor, epsilon)], bins, alpha=0.5, label= "Average of two")
#     plt.hist(predictions_plus_minus_of_two_laplacian[(histogram_factor, epsilon)], bins, alpha=0.5, label= "Plus minus trick")
#     # plt.hist(predictions_additive_n_r_laplacian[(histogram_factor, epsilon)], bins, alpha=0.5, label= "(noise, r-noise)")
#     plt.hist(predictions_additive_n_2_laplacian[(histogram_factor, epsilon)], bins, alpha=0.5, label= "(r/2 + noise, r/2 - noise)")
#     plt.hist(test['rating'], bins, alpha=0.5, label='real ratings')
#     plt.legend(loc='upper right')
#     plt.show()