import pickle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)

handle = open('cdrs_mae.pickle', 'rb')
maes = pickle.load(handle)
pp_maes = pickle.load(handle)
handle.close()


handle = open('cdrs_rmse.pickle', 'rb')
rmses = pickle.load(handle)
pp_rmses = pickle.load(handle)
handle.close()


latent_vector_sizes = [2, 4, 6, 10, 16, 30, 50, 100]
user_overlap_percentages = [0.05, 0.1, 0.2, 0.3]
epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]


factor = 6

# plt.figure(figsize=(15, 15), dpi=100)

plt.xlabel('Latent Vector Size')
plt.title('Latent Factor Dimension vs Error')
plt.ylabel('MAE')
plt.xticks(latent_vector_sizes)

for percentage in user_overlap_percentages:
    plt.plot(latent_vector_sizes, maes[percentage], linewidth=1, marker =".", label= "Percentage: {:.2f}".format(percentage))
#
# show legend
plt.ylim(0.8, 0.95)

plt.legend()

# show graph
plt.show()


for overlap in user_overlap_percentages:
    plt.title('MAE with User Overlap {}%'.format(int(overlap*100)))
    ind = np.arange(len(epsilons))
    width = 0.1       # the width of the bars

    epsilon_strings = [str(epsilon) for epsilon in epsilons]

    plt.bar(ind, [pp_maes[overlap][epsilons.index(epsilon)] for epsilon in epsilons], label= "PP-CDRS", width=width)
    plt.bar(ind+width, [maes[overlap][latent_vector_sizes.index(factor)] for epsilon in epsilons], label= "Baseline", width=width)

    plt.xticks(ind + width / 2, epsilon_strings)


    # show legend
    plt.ylim(0, 1.1)

    plt.xlabel('Epsilon')
    plt.ylabel('MAE')
    plt.legend()
    # plt.legend(prop={'size': 12})

    # show graph
    plt.show()