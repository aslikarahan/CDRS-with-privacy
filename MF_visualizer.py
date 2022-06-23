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

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)

df = fetch_ml_ratings(variant='100k')
train, test = train_test_split(df, test_size=0.2, random_state=42)

latent_vector_sizes = [3, 5, 10, 15, 30, 50, 100]
noise_range = [1, 2, 3, 5, 10, 30, 100]
epsilons = [0.1, 0.5, 1, 3]

colors = ["tab:blue", "tab:orange","tab:green","tab:red","tab:purple"]
line_styles = [ "dashdot", "dotted", "dashed","solid"]

handle = open('pickles/results_mae.pickle', 'rb')
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

handle = open('pickles/adversary_mae.pickle', 'rb')
adversary_predictions_base_noise_mae = pickle.load(handle)
adversary_predictions_average_of_two_mae_1 = pickle.load(handle)
adversary_predictions_plus_minus_mae_1 = pickle.load(handle)
adversary_predictions_half_mae = pickle.load(handle)
handle.close()

plt.xlabel('Latent Factor Dimension')
plt.title('Effect of Increasing the Redundancy')
plt.ylabel('MAE')
plt.xticks(latent_vector_sizes)

for epsilon in epsilons:
    color_index = epsilons.index(epsilon)
    col = colors[color_index]
    plt.plot(latent_vector_sizes, results_base_noise_laplacian[epsilon], linewidth=linewidth_custom, color=col,
             linestyle=line_styles[0], label="Single party, \u03B5: {:.1f}".format(epsilon))
    plt.plot(latent_vector_sizes, results_average_of_two_laplacian[epsilon], linewidth=linewidth_custom, color=col ,linestyle = line_styles[1], label= "Two parties, \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_plus_minus_of_two_laplacian[epsilon], linewidth=1, label= "Two parties, opposite sign noise with \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_r_laplacian[epsilon], linewidth=1, marker ="v", label= "(noise, r-noise): {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_2_laplacian[epsilon], linewidth=1, label= "(r/2 + noise, r/2 - noise): {:.1f}".format(epsilon))
plt.plot(  latent_vector_sizes, results_baseline, linewidth=linewidth_custom,color=colors[4], linestyle = line_styles[3],label= 'Without noise')

lines = [Line2D([0], [0], color=colors[epsilons.index(eps)]) for eps in epsilons]
lines.append(Line2D([0], [0], color='black', linestyle = line_styles[0]))
lines.append(Line2D([0], [0], color='black', linestyle = line_styles[1]))
lines.append(Line2D([0], [0], color=colors[4], linestyle = line_styles[3]))

labels = ["\u03B5: {:.1f}".format(eps) for eps in epsilons]
labels.append("Single party")
labels.append("Two parties")
labels.append("Without noise")
plt.legend(lines, labels, prop={'size': SMALL_SIZE-LEGEND_SCALE}, loc='lower right')
plt.ylim(0, 2.1)
# plt.savefig("img/redundancy1.png")
plt.show()

plt.xlabel('Latent Factor Dimension')
plt.title('Effect of Increasing the Redundancy: Noise with Opposite Signs')
plt.ylabel('MAE')
plt.xticks(latent_vector_sizes)

for epsilon in epsilons:
    color_index = epsilons.index(epsilon)
    col = colors[color_index]
    # plt.plot(latent_vector_sizes, results_base_noise_laplacian[epsilon], linewidth=1, label= "Single party, \u03B5 {:.1f}".format(epsilon))
    plt.plot(latent_vector_sizes, results_average_of_two_laplacian[epsilon], linewidth=1,color=col , linestyle = line_styles[0],label= "Two parties, \u03B5: {:.1f}".format(epsilon))
    plt.plot(latent_vector_sizes, results_plus_minus_of_two_laplacian[epsilon], linewidth=1, color=col , linestyle = line_styles[1],label= "Two parties, opposite sign noise, \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_r_laplacian[epsilon], linewidth=1, marker ="v", label= "(noise, r-noise): {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_2_laplacian[epsilon], linewidth=1, label= "(r/2 + noise, r/2 - noise): {:.1f}".format(epsilon))
plt.plot(  latent_vector_sizes, results_baseline, linewidth=1, color=colors[4], linestyle = line_styles[3], label= 'Without noise')

lines = [Line2D([0], [0], color=colors[epsilons.index(eps)]) for eps in epsilons]
lines.append(Line2D([0], [0], color='black', linestyle = line_styles[0]))
lines.append(Line2D([0], [0], color='black', linestyle = line_styles[1]))
lines.append(Line2D([0], [0], color=colors[4], linestyle = line_styles[3]))

labels = ["\u03B5: {:.1f}".format(eps) for eps in epsilons]
labels.append("Two parties")
labels.append("Two parties, opposite sign noise")
labels.append("Without noise")
plt.legend(lines, labels, prop={'size': SMALL_SIZE-LEGEND_SCALE}, loc='lower right')
plt.ylim(0, 2.1)
# plt.savefig("img/redundancy2.png")
plt.show()

plt.xlabel('Latent Factor Dimension')
plt.title('Effect of Additive Separation')
plt.ylabel('MAE')
plt.xticks(latent_vector_sizes)

for epsilon in epsilons:
    color_index = epsilons.index(epsilon)
    col = colors[color_index]
    plt.plot(latent_vector_sizes, results_base_noise_laplacian[epsilon], linewidth=1,color=col  ,linestyle = line_styles[0], label= "Single party, \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_average_of_two_laplacian[epsilon], linewidth=1, label= "Two parties, \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_plus_minus_of_two_laplacian[epsilon], linewidth=1, label= "Two parties, opposite sign noise with \u03B5 {:.1f}".format(epsilon))
    # plt.plot(latent_vector_sizes, results_additive_n_r_laplacian[epsilon], linewidth=1, marker ="v", label= "(noise, r-noise): {:.1f}".format(epsilon))
    plt.plot(latent_vector_sizes, results_additive_n_2_laplacian[epsilon], linewidth=1,color=col ,linestyle = line_styles[1], label= "Additive separation, \u03B5 {:.1f}".format(epsilon))
plt.plot(  latent_vector_sizes, results_baseline, linewidth=1, color=colors[4], linestyle = line_styles[3],label= 'Without noise')

lines = [Line2D([0], [0], color=colors[epsilons.index(eps)]) for eps in epsilons]
lines.append(Line2D([0], [0], color='black', linestyle = line_styles[0]))
lines.append(Line2D([0], [0], color='black', linestyle = line_styles[1]))
lines.append(Line2D([0], [0], color=colors[4], linestyle = line_styles[3]))

labels = ["\u03B5: {:.1f}".format(eps) for eps in epsilons]
labels.append("Single party")
labels.append("Additive separation")
labels.append("Without noise")
plt.legend(lines, labels, prop={'size': SMALL_SIZE-LEGEND_SCALE}, loc='lower right')
plt.ylim(0, 2.1)
# plt.savefig("img/additive.png")
plt.show()

latent_vector_sizes = [3, 5, 10, 15, 30, 50, 100]
noise_range = [1, 2, 3, 5, 10, 30, 100]
epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]

plt.xlabel('\u03B5')
bar_chart_factor = 2
plt.title('Errors with Latent Vector Size: {}'.format(latent_vector_sizes[bar_chart_factor]))
plt.ylabel('MAE')
ind = np.arange(len(epsilons))
width = 0.15  # the width of the bars
epsilon_strings = [str(epsilon) for epsilon in epsilons]
plt.bar(ind-2*width, results_baseline[bar_chart_factor], label= "Without noise", width=width)
plt.bar(ind-width, [results_base_noise_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "Single party with noise", width=width)
plt.bar(ind, [results_average_of_two_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "Increasing the redundancy", width=width)
plt.bar(ind+width, [results_plus_minus_of_two_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "Increasing the redundancy, opposite signs", width=width)
# plt.bar(ind+2*width, [results_additive_n_r_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "(noise, r-noise)", width=width)
plt.bar(ind+2*width, [results_additive_n_2_laplacian[epsilon][bar_chart_factor] for epsilon in epsilons], label= "Additive separation", width=width)

plt.xticks(ind + width / 5, epsilon_strings)
# plt.yticks(np.linspace(0,2, 10))
plt.tick_params(axis='y', which='minor', bottom=False)
plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5, alpha = 0.5)
plt.minorticks_on()
plt.ylim(0, 2.1)
plt.legend(prop={'size': SMALL_SIZE-LEGEND_SCALE})
# plt.savefig("img/comparison_all.png")
plt.show()

epsilons = [1, 1.5, 2, 3]

histogram_factor = 10
for epsilon in epsilons:
    bins = np.linspace(1, 5, 100)
    ax1 = plt.subplot(4, 1, 1)
    ax1.hist(predictions_baseline[histogram_factor], bins, color="orange", alpha=0.9, label="Without noise")
    # plt.hist(predictions_base_noise_laplacian[(histogram_factor, epsilon)], bins, alpha=0.5, label= "Base and noise")
    # plt.hist(predictions_average_of_two_laplacian[(histogram_factor, epsilon)], bins, alpha=0.5, label= "Average of two")
    # plt.hist(test['rating'], bins, alpha=0.5, color="black", label='Actual Ratings')
    plt.legend(loc='upper right')
    plt.legend(prop={'size': SMALL_SIZE})
    ax2 = plt.subplot(4, 1, 2)
    ax2.hist(predictions_plus_minus_of_two_laplacian[(histogram_factor, epsilon)], bins, color="green", alpha=0.5,
             label="Inc. redundancy, opposite signs")

    plt.legend(loc='upper right')
    plt.legend(prop={'size': SMALL_SIZE})
    ax3 = plt.subplot(4, 1, 3)
    ax3.hist(predictions_additive_n_2_laplacian[(histogram_factor, epsilon)], bins, color="red", alpha=0.5,
             label="Additive separation")
    plt.legend(loc='upper right')
    plt.legend(prop={'size': SMALL_SIZE})
    ax4 = plt.subplot(4, 1, 4)
    ax4.hist(test['rating'], bins, alpha=0.5, color="black", label='Actual Ratings')
    plt.legend(loc='upper right')
    plt.legend(prop={'size': SMALL_SIZE})
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax3.set_xticks([1, 2, 3, 4, 5])
    ax4.set_xticks([1, 2, 3, 4, 5])

    ax1.get_shared_x_axes().join(ax1, ax2, ax3, ax4)
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    fig = plt.gcf()

    fig.suptitle("Rating Histogram with \u03B5 {:.1f}".format(epsilon), fontsize=14)
    fig.supxlabel('Predicted Rating Value', fontsize=10)
    fig.supylabel('Number of Ratings', fontsize=10)

    # plt.savefig(f"img/histogram{epsilon}.png")
    plt.show()

epsilons = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]

plt.xlabel('\u03B5')
bar_chart_factor = 2
plt.title('Adversary Errors in Predicting Ratings')
plt.ylabel('MAE')

ind = np.arange(len(epsilons))
width = 0.15  # the width of the bars
epsilon_strings = [str(epsilon) for epsilon in epsilons]
plt.bar(ind - 3 * width / 2, [adversary_predictions_base_noise_mae[epsilon] for epsilon in epsilons],
        label="Single party with noise", width=width)
plt.bar(ind - width / 2, [adversary_predictions_average_of_two_mae_1[epsilon] for epsilon in epsilons],
        label="Increasing the redundancy", width=width)
plt.bar(ind + width / 2, [adversary_predictions_plus_minus_mae_1[epsilon] for epsilon in epsilons],
        label="Increasing the redundancy, opposite signs", width=width)
plt.bar(ind + 3 * width / 2, [adversary_predictions_half_mae[epsilon] for epsilon in epsilons],
        label="Additive separation", width=width)

plt.xticks(ind + width / 4, epsilon_strings)
# plt.yticks(np.linspace(0,2, 10))
plt.tick_params(axis='y', which='minor', bottom=False)
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on()
plt.ylim(0, 2.1)
plt.legend(prop={'size': SMALL_SIZE - LEGEND_SCALE})

# plt.savefig("img/comparison_adversary_mae.png")
plt.show()

# collaborating
handle = open('pickles/adversary_collaborating_mae.pickle', 'rb')
adversary_predictions_base_noise_mae = pickle.load(handle)
adversary_predictions_average_of_two_mae_1 = pickle.load(handle)
adversary_predictions_plus_minus_mae_1 = pickle.load(handle)
handle.close()

plt.xlabel('\u03B5')
bar_chart_factor = 2
plt.title('Adversary Errors in Predicting Ratings When Parties Collaborate')
plt.ylabel('MAE')

ind = np.arange(len(epsilons))
width = 0.15  # the width of the bars
epsilon_strings = [str(epsilon) for epsilon in epsilons]
plt.bar(ind - 3 * width / 2, [adversary_predictions_base_noise_mae[epsilon] for epsilon in epsilons],
        label="Single party with noise", width=width)
plt.bar(ind - width / 2, [adversary_predictions_average_of_two_mae_1[epsilon] for epsilon in epsilons],
        label="Increasing the redundancy", width=width)
plt.bar(ind + width / 2, [adversary_predictions_plus_minus_mae_1[epsilon] for epsilon in epsilons],
        label="Increasing the redundancy, opposite signs", width=width)
plt.bar(ind + 3 * width / 2, [0.005 for epsilon in epsilons], label="Additive separation", width=width)

plt.xticks(ind + width / 4, epsilon_strings)
plt.tick_params(axis='y', which='minor', bottom=False)
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on()
plt.ylim(0, 2.1)
plt.legend(prop={'size': SMALL_SIZE - LEGEND_SCALE})

# plt.savefig("img/comparison_collaborating_mae.png")
plt.show()
