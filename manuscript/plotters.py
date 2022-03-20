import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
# plt.rcParams.update({
#     "text.usetex": True,})
    # "font.family": "sans-serif",
    # "font.sans-serif": ["Helvetica"]})

def plot_error_fn_compressions(results):
    iter_range = results["iter_range"]
    optimal_errors_norm = results['opt_error_norm']
    even_errors_norm = results['even_error_norm']
    tce_all = results["tce_all"]
    fig, ax = plt.subplots(2, 1)

    colors = sns.color_palette("hls", 8)
    # ax[0].scatter(iter_range, optimal_errors, color=colors[0], label='Algorithmic')
    # ax[0].plot(iter_range, optimal_errors, color=colors[0], label='Algorithmic')
    # ax[0].scatter(iter_range, even_errors, color=colors[3], label='Even')
    # ax[0].plot(iter_range, even_errors, color=colors[3], label='Even')
    ax[1].scatter(iter_range, optimal_errors_norm, color=colors[0], label='Algorithmic')
    ax[1].plot(iter_range, optimal_errors_norm, color=colors[0], label='Algorithmic')
    ax[1].scatter(iter_range, even_errors_norm, color=colors[3], label='Even')
    ax[1].plot(iter_range, even_errors_norm, color=colors[3], label='Even')
    ax[0].scatter(iter_range, tce_all, color=colors[3], label='TCE')
    ax[0].plot(iter_range, tce_all, color=colors[3], label='TCE')
    # plt.xticks(np.linspace(0, iter_range+1, 10))
    # plt.xticks(list(int(np.arange(0, iter_range+1))))
    ax[1].set_xlabel('Iterations')
    ax[0].set_ylabel('Total chosen error')
    ax[1].set_ylabel('Total normalized error')
    # plt.xticks([0, 5, 10, 15, 19])
    plt.legend(loc='upper left')

def plot_one_round(results):
    """
    :param results: dict? for now
    :return:
    """
    temp_t = results['temp_t']
    temp_inf = results['temp_inf']
    algo_t = results['algo_t']
    algo_inf = results['algo_inf']
    even_t = results['even_t']
    even_inf = results['even_inf']
    algo_boundary_times = results['algo_boundary_times']
    temp_boundary_times = results['temp_boundary_times']
    even_boundary_times = results['even_boundary_times']
    total_algo_error = results['total_algo_error']
    total_even_error = results['total_even_error']
    total_chosen_error = results['total_chosen_error']
    print("STARTING PLOTTING")
    colors = sns.color_palette("hls", 8)
    colors = sns.color_palette()
    colors = sns.color_palette("Paired")
    # colors = ["#C54CC5", "#00A4D4", "#FFC626"]
    # colors = ["crimson", "#00A4D4", "#FFC626", 'blue']
    type_colors = {'temp': colors[1], 'even': colors[3], 'algo': colors[0]}
    type_colors = {'temp': 'grey', 'even': colors[7], 'algo': 'c'}
    type_colors = {'temp': 'grey', 'even': "#FFC626", 'algo': "#00A4D4"}
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax = axs[0]
    ax.plot(temp_t, temp_inf, label='Temporal', color=type_colors['temp'], lw=2.5, alpha=1.0)
    # ax.plot(opt_t, opt_inf, label='Algorithmic-terminal', color=colors[1], lw=2, alpha=0.6, ls='--')
    # ax.plot(opt_t_h, opt_inf_h, label='Algorithmic-halftime', color=colors[3], lw=2, alpha=0.6, ls='--')
    ax.plot(algo_t, algo_inf, label='Algorithmic', color=type_colors['algo'], lw=2, alpha=1.0, ls='--')
    # ax.plot(rand_t, rand_inf, label='Random', color=colors[6], lw=2, alpha=0.6, ls='-.')
    ax.plot(even_t, even_inf, label='Even', color=type_colors['even'], lw=2, alpha=1.0, ls='-.')
    ## vertical lines to show compression
    max_infected_buffer = max(max(temp_inf), max(algo_inf)) + 2
    ax.vlines(temp_boundary_times, ymin=0, ymax=max_infected_buffer / 3, ls='-',
              color=type_colors['temp'], lw=0.5, alpha=1.0)
    # ax.vlines(opt_net.get_time_network_map().keys(), ymin=2*max_infected_buffer/3, ymax=max_infected_buffer, ls='-', color=colors[1], lw=2, alpha=1.0)
    # ax.vlines(opt_net_h.get_time_network_map().keys(), ymin=2*max_infected_buffer/3, ymax=max_infected_buffer, ls='-', color=colors[3], lw=1, alpha=0.95)
    ax.vlines(algo_boundary_times, ymin=2 * max_infected_buffer / 3, ymax=max_infected_buffer,
              ls='--', color=type_colors['algo'], lw=1, alpha=0.95)
    # ax.vlines(rand_net.get_time_network_map().keys(), ymin=max_infected_buffer/3, ymax=2*max_infected_buffer/3, ls='--', color=colors[6], lw=1, alpha=0.95)
    ax.vlines(even_boundary_times, ymin=max_infected_buffer / 3,
              ymax=2 * max_infected_buffer / 3, ls='-.', color=type_colors['even'], lw=1, alpha=0.95)
    # ax.xlabel('Time')
    ax.set_ylabel('Infected nodes')
    # ax.set_xticks(list(temporal_network.get_time_network_map().keys())[::4])
    axs[0].legend(loc='lower right', frameon=False)
    # plt.show()
    ax = axs[1]
    # plt.figure('error')
    # for the plot, have it be normalized error?
    # TODO 3/3: THIS ONE IS STILL NORMALIZING
    # ax.plot(opt_t_c, (-np.array(temp_inf)+np.array(opt_inf))/np.array(temp_inf), label=f'Optimal terminal: {total_optimal_error_nm}', color=colors[1], ls='--')
    # ax.plot(opt_t_c, (-np.array(temp_inf)+np.array(opt_inf_h))/np.array(temp_inf), label=f'Optimal halftime: {total_optimal_h_error_nm}', color=colors[3], ls='--')
    ax.plot(algo_t, (-np.array(temp_inf) + np.array(algo_inf)) / np.array(temp_inf),
            label=f'Algorithmic: {total_algo_error}\n TCE: {total_chosen_error}',
            color=type_colors['algo'], ls='--', lw=2, alpha=1.0)
    ax.fill_between(algo_t, (-np.array(temp_inf) + np.array(algo_inf)) / np.array(temp_inf),
                    color=type_colors['algo'], alpha=0.5)
    # ax.plot(opt_t, (-np.array(temp_inf)+np.array(opt_inf)), label=f'Algorithmic {total_optimal_error}', color=type_colors['even', ls='--')
    # ax.plot(rand_t, (-np.array(temp_inf)+np.array(rand_inf))/np.array(temp_inf), label=f'Random: {total_random_error_nm}', color=colors[6], ls='-.')
    # ax.plot(rand_t, (-np.array(temp_inf)+np.array(rand_inf)), label=f'Random {total_random_error}', color=colors[6], ls='-.')
    ax.plot(even_t, (-np.array(temp_inf) + np.array(even_inf)) / np.array(temp_inf),
            label=f'Even {total_even_error}',
            color=type_colors['even'], ls='-.', lw=2)
    ax.fill_between(even_t, (-np.array(temp_inf) + np.array(even_inf)) / np.array(temp_inf), color=type_colors['even'], alpha=0.5)
    # TODO do actual integral
    # ax.plot(even_t, (-np.array(temp_inf)+np.array(even_inf)), label=f'Even {total_even_error}', color=colors[3], ls='-.')
    ax.plot(algo_t, np.zeros(len(algo_t)), color='k', ls='-', alpha=1.0)
    ax.set_xlabel('Time')
    # ax.set_ylabel('Normalized error')
    ax.set_ylabel('Normalized error')
    axs[1].legend(frameon=False)
    # plt.show()
    plt.tight_layout()
    # fig.set_size_inches(5,5)
    print("ENDING PLOT")


def plot_heatmap(matrix, betas, compressions, title='Difference of Algorithmic-Even Total Integrated Error'):
    # cmap = sns.diverging_palette(220, 20, as_cmap=True)
    colorlist = ["#C54CC5",  "#FFC626", "#00A4D4",]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    c = np.linspace(0, 80, 12)
    ax = sns.heatmap(matrix, cmap=newcmp, center=0, square=True)
    ax.set_xlabel('Number of compressions')
    ax.set_xticklabels(compressions)
    ax.set_yticklabels(betas, rotation=45)
    ax.set_ylabel('Beta')
    ax.set_title(title)

def plot_tce(matrix, betas, compressions):
    fig, ax = plt.subplots()
    for b, beta in enumerate(betas):
        ax.plot(compressions, matrix[b], label=beta)
    ax.legend(frameon=False)


def plot_comparison_and_error(results, betas, compressions, title=""):
    colorlist = ["#C54CC5",  "#FFC626", "#00A4D4",]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    ax = sns.heatmap(results["difference_matrix"], cmap=newcmp, center=0, linecolor='white', linewidths=.5,
                     cbar_kws={'label': 'algorithmic error minus even error', 'orientation': 'horizontal'})
    ax.set_xlabel('Number of compressions')
    ax.set_xticklabels(compressions)
    ax.set_yticklabels(betas, rotation=45)
    ax.set_ylabel('Beta')
    ax.set_title(title)
    y_ticks = []
    # ax2 = ax.twinx()
    for b in range(len(betas)-1):
        matrix = np.cumsum(results["tce"], axis=0)
        sns.lineplot(np.arange(10) + .5, ((-matrix[b] / np.max(matrix[b])) + 1 + b), color='k', lw=1)
        y_ticks.append(b)
    sns.lineplot(np.arange(10) + .5, ((-matrix[len(betas)-1] / np.max(matrix[len(betas)-1])) + 1 + (len(betas)-1)),
                 color='k', lw=1, label='Sum of aggregation error \n chosen by algorithm')
    ax.legend(frameon=False)


