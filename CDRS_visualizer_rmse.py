import pickle

import numpy as np
from matplotlib import pyplot as plt

SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)

handle = open('pickles/cdrs_rmse.pickle', 'rb')
maes = pickle.load(handle)
pp_maes = pickle.load(handle)
handle.close()

latent_vector_sizes = [2, 4, 6, 10, 16, 30, 50, 100]
user_overlap_percentages = [0.05, 0.1, 0.2, 0.3]
epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]
factor = 6
plt.xlabel('Latent Vector Size')
plt.title('Latent Factor Dimension vs Error - CDRS without Privacy')
plt.ylabel('RMSE')
plt.xticks(latent_vector_sizes)
for percentage in user_overlap_percentages:
    plt.plot(latent_vector_sizes, maes[percentage], linewidth=1, marker=".",
             label="Percentage: {:.2f}".format(percentage))
plt.ylim(0, 1.5)
plt.legend()
# plt.savefig(f"img/cdrs/rmse/normal.png")
plt.show()

for overlap in user_overlap_percentages:
    plt.title('RMSE with User Overlap {}%'.format(int(overlap * 100)))
    ind = np.arange(len(epsilons))
    width = 0.1
    epsilon_strings = [str(epsilon) for epsilon in epsilons]
    plt.bar(ind, [pp_maes[overlap][epsilons.index(epsilon)] for epsilon in epsilons], label="PP-CDRS", width=width)
    plt.bar(ind + width, [maes[overlap][latent_vector_sizes.index(factor)] for epsilon in epsilons], label="Baseline",
            width=width)
    plt.xticks(ind + width / 2, epsilon_strings)
    plt.ylim(0, 1.5)
    plt.tick_params(axis='y', which='minor', bottom=False)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlabel('Epsilon')
    plt.ylabel('RMSE')
    plt.legend()
    # plt.savefig(f"img/cdrs/rmse/overlap{int(overlap*100)}.png")
    plt.show()
