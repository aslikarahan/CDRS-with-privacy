import pickle

import numpy as np
from funk_svd.dataset import fetch_ml_ratings
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

SMALL_SIZE = 10
linewidth_custom = 1
LEGEND_SCALE = 2
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)

df = fetch_ml_ratings(variant='100k')

train, test = train_test_split(df, test_size=0.2, random_state=42)

latent_vector_sizes = [3, 5, 10, 15, 30, 50, 100]
noise_range = [1, 2, 3, 5, 10, 30, 100]
epsilons = [0.1, 0.5, 1, 3]

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
line_styles = ["dashdot", "dotted", "dashed", "solid"]

handle = open('pickles/results_rmse.pickle', 'rb')
results_baseline = pickle.load(handle)
results_base_noise_laplacian = pickle.load(handle)
results_average_of_two_laplacian = pickle.load(handle)
results_plus_minus_of_two_laplacian = pickle.load(handle)
results_additive_n_r_laplacian = pickle.load(handle)
results_additive_n_2_laplacian = pickle.load(handle)
handle.close()

handle = open('pickles/predictions.pickle', 'rb')
predictions_baseline = pickle.load(handle)
predictions_base_noise_laplacian = pickle.load(handle)
predictions_average_of_two_laplacian = pickle.load(handle)
predictions_plus_minus_of_two_laplacian = pickle.load(handle)
predictions_additive_n_r_laplacian = pickle.load(handle)
predictions_additive_n_2_laplacian = pickle.load(handle)
handle.close()

plt.xlabel('Latent Factor Dimension')
plt.title('Effect of Increasing the Redundancy')
plt.ylabel('RMSE')
plt.xticks(latent_vector_sizes)

for epsilon in epsilons:
    color_index = epsilons.index(epsilon)
    col = colors[color_index]
    plt.plot(latent_vector_sizes, results_base_noise_laplacian[epsilon], linewidth=linewidth_custom, color=col,
             linestyle=line_styles[0], label="Single party, \u03B5: {:.1f}".format(epsilon))
    plt.plot(latent_vector_sizes, results_average_of_two_laplacian[epsilon], linewidth=linewidth_custom, color=col,
             linestyle=line_styles[1], label="Two parties, \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_plus_minus_of_two_laplacian[epsilon], linewidth=1, label= "Two parties, opposite sign noise with \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_r_laplacian[epsilon], linewidth=1, marker ="v", label= "(noise, r-noise): {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_2_laplacian[epsilon], linewidth=1, label= "(r/2 + noise, r/2 - noise): {:.1f}".format(epsilon))
plt.plot(latent_vector_sizes, results_baseline, linewidth=linewidth_custom, color=colors[4], linestyle=line_styles[3],
         label='Without noise')

lines = [Line2D([0], [0], color=colors[epsilons.index(eps)]) for eps in epsilons]
lines.append(Line2D([0], [0], color='black', linestyle=line_styles[0]))
lines.append(Line2D([0], [0], color='black', linestyle=line_styles[1]))
lines.append(Line2D([0], [0], color=colors[4], linestyle=line_styles[3]))

labels = ["\u03B5: {:.1f}".format(eps) for eps in epsilons]
labels.append("Single party")
labels.append("Two parties")
labels.append("Without noise")
plt.legend(lines, labels, prop={'size': SMALL_SIZE - LEGEND_SCALE}, loc='lower right')
plt.ylim(0, 2.5)
# plt.savefig("img/rmse/redundancy1.png")
plt.show()

plt.xlabel('Latent Factor Dimension')
plt.title('Effect of Increasing the Redundancy: Noise with Opposite Signs')
plt.ylabel('RMSE')
plt.xticks(latent_vector_sizes)

for epsilon in epsilons:
    color_index = epsilons.index(epsilon)
    col = colors[color_index]
    # plt.plot(latent_vector_sizes, results_base_noise_laplacian[epsilon], linewidth=1, label= "Single party, \u03B5 {:.1f}".format(epsilon))
    plt.plot(latent_vector_sizes, results_average_of_two_laplacian[epsilon], linewidth=1, color=col,
             linestyle=line_styles[0], label="Two parties, \u03B5: {:.1f}".format(epsilon))
    plt.plot(latent_vector_sizes, results_plus_minus_of_two_laplacian[epsilon], linewidth=1, color=col,
             linestyle=line_styles[1], label="Two parties, opposite sign noise, \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_r_laplacian[epsilon], linewidth=1, marker ="v", label= "(noise, r-noise): {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_2_laplacian[epsilon], linewidth=1, label= "(r/2 + noise, r/2 - noise): {:.1f}".format(epsilon))
plt.plot(latent_vector_sizes, results_baseline, linewidth=1, color=colors[4], linestyle=line_styles[3],
         label='Without noise')

lines = [Line2D([0], [0], color=colors[epsilons.index(eps)]) for eps in epsilons]
lines.append(Line2D([0], [0], color='black', linestyle=line_styles[0]))
lines.append(Line2D([0], [0], color='black', linestyle=line_styles[1]))
lines.append(Line2D([0], [0], color=colors[4], linestyle=line_styles[3]))

labels = ["\u03B5: {:.1f}".format(eps) for eps in epsilons]
labels.append("Two parties")
labels.append("Two parties, opposite sign noise")
labels.append("Without noise")
plt.legend(lines, labels, prop={'size': SMALL_SIZE - LEGEND_SCALE}, loc='lower right')
plt.ylim(0, 2.5)
# plt.savefig("img/rmse/redundancy2.png")
plt.show()

plt.xlabel('Latent Factor Dimension')
plt.title('Effect of Additive Separation')
plt.ylabel('RMSE')
plt.xticks(latent_vector_sizes)

for epsilon in epsilons:
    color_index = epsilons.index(epsilon)
    col = colors[color_index]
    plt.plot(latent_vector_sizes, results_base_noise_laplacian[epsilon], linewidth=1, color=col,
             linestyle=line_styles[0], label="Single party, \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_average_of_two_laplacian[epsilon], linewidth=1, label= "Two parties, \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_plus_minus_of_two_laplacian[epsilon], linewidth=1, label= "Two parties, opposite sign noise with \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_r_laplacian[epsilon], linewidth=1, marker ="v", label= "(noise, r-noise): {:.1f}".format(epsilon))
    plt.plot(latent_vector_sizes, results_additive_n_2_laplacian[epsilon], linewidth=1, color=col,
             linestyle=line_styles[1], label="Additive separation, \u03B5 {:.1f}".format(epsilon))
plt.plot(latent_vector_sizes, results_baseline, linewidth=1, color=colors[4], linestyle=line_styles[3],
         label='Without noise')

lines = [Line2D([0], [0], color=colors[epsilons.index(eps)]) for eps in epsilons]
lines.append(Line2D([0], [0], color='black', linestyle=line_styles[0]))
lines.append(Line2D([0], [0], color='black', linestyle=line_styles[1]))
lines.append(Line2D([0], [0], color=colors[4], linestyle=line_styles[3]))

labels = ["\u03B5: {:.1f}".format(eps) for eps in epsilons]
labels.append("Single party")
labels.append("Additive separation")
labels.append("Without noise")
plt.legend(lines, labels, prop={'size': SMALL_SIZE - LEGEND_SCALE}, loc='lower right')
plt.ylim(0, 2.5)
# plt.savefig("img/rmse/additive.png")
plt.show()

latent_vector_sizes = [3, 5, 10, 15, 30, 50, 100]
noise_range = [1, 2, 3, 5, 10, 30, 100]
epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]

plt.xlabel('\u03B5')
bar_chart_factor = 2
plt.title('Errors with Latent Vector Size: {}'.format(latent_vector_sizes[bar_chart_factor]))
plt.ylabel('RMSE')
ind = np.arange(len(epsilons))
width = 0.15  # the width of the bars
epsilon_strings = [str(epsilon) for epsilon in epsilons]
plt.bar(ind - 2 * width, results_baseline[bar_chart_factor], label="Without noise", width=width)
plt.bar(ind - width, [results_base_noise_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons],
        label="Single party with noise", width=width)
plt.bar(ind, [results_average_of_two_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons],
        label="Increasing the redundancy", width=width)
plt.bar(ind + width, [results_plus_minus_of_two_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons],
        label="Increasing the redundancy, opposite signs", width=width)
plt.bar(ind + 2 * width, [results_additive_n_2_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons],
        label="Additive separation", width=width)

plt.xticks(ind + width / 5, epsilon_strings)
plt.tick_params(axis='y', which='minor', bottom=False)
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on()
plt.ylim(0, 2.5)
plt.legend(prop={'size': SMALL_SIZE - LEGEND_SCALE})

# plt.savefig("img/rmse/comparison_all.png")
plt.show()
