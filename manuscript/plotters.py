import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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