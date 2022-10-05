import pandas as pd
import manuscript.network_generators as netgen
from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import json
import constants

CONF_DAT = False
HOSP_DAT = True

if CONF_DAT:
    num_snapshots = 200
    # num_snapshots = 1000 # use .000065
    # num_snapshots = 4000 # use .0002
    beta = .000015
    conference_snapshots = netgen.data_network_snapshots('../raw_data/tij_InVS.dat',
                                                         int(constants.CONF_TOTAL_TIME/num_snapshots), beta,
                                                         int(num_snapshots/2), constants.CONF_START_TIME)

    results = one_round(TemporalNetwork(conference_snapshots), int(constants.CONF_TOTAL_TIME/num_snapshots), beta,
                        len(conference_snapshots), iters=180)
    plot_one_round(results)


if HOSP_DAT:
    """
    Dataset network statistics for hospital data
    """
    # netgen.dataset_statistics('../raw_data/detailed_list_of_contacts_Hospital.dat_')
    # plt.show()
    """
    Split data into snapshots with specified number of snapshots
    """
    # Durations with 200 snapshots: 1737
    num_snapshots = 200
    hospital_snapshots = netgen.data_network_snapshots('../raw_data/detailed_list_of_contacts_Hospital.dat_',
                                                       int(constants.HOSP_TOTAL_TIME/num_snapshots), .000015,
                                                       int(num_snapshots/2),
                                                       constants.HOSP_START_TIME)

    """
    Solve for the temporal solution given the number of snapshots and beta to assess base dynamics
    """
    res = run_temporal(TemporalNetwork(hospital_snapshots), int(constants.HOSP_TOTAL_TIME/num_snapshots), .000015,
                       num_snapshots)
    plt.scatter(res[0], res[1])
    plt.show()

    """
    Assessing compression from 1,000 snapshots to 10 vs 200
    """
    # num_snapshots = 1000
    # results_one_round = one_round(TemporalNetwork(hospital_snapshots), int(total_time/num_snapshots), .000015, len(hospital_snapshots), iters=980)
    # results_one_round = one_round(TemporalNetwork(hospital_snapshots), int(total_time/num_snapshots), .00009, len(hospital_snapshots), iters=800)
    # f = open('hospital_data_1000_to_200.json', "w")
    # json.dump(results_one_round, f)
    # f.close()
    # plot_one_round(results_one_round)
    # plt.show()

    """
    Panel (a) for Fig. 4 experiment.
    Solving for pairwise error between snapshots at 3 granularities of initial pre-compressed snapshots
    """
    # # Getting pairwise errors for 3 granularities
    # layer_nums = [200, 1000, 4000]
    # tau = int(constants.HOSP_TOTAL_TIME/num_snapshots)*.000015 # MEANS .000015 = tau / int(total_time/num_snapshots)
    # # to get beta for the others, do new_beta = tau / int(total_time/num_snapshots)
    # beta_1000 = tau / int(constants.HOSP_TOTAL_TIME/1000)
    # beta_4000 = tau / int(constants.HOSP_TOTAL_TIME/4000)
    # betas = [.000015, beta_1000, beta_4000]
    #
    # for i in range(len(layer_nums)):
    #     layer_num = layer_nums[i]
    #     b = betas[i]
    #     hospital_snapshots = netgen.data_network_snapshots('../raw_data/detailed_list_of_contacts_Hospital.dat_',
    #                                                        int(constants.HOSP_TOTAL_TIME / layer_num), b,
    #                                                        int(layer_num / 2), 140)
    #     st, eps = plot_pairwise_error(hospital_snapshots)
    #     res = pd.DataFrame()
    #     res["st_times"] = st
    #     res["pariwise_eps"] = eps
    #     res.to_csv(f"./results/draft2/pairwise_error_{layer_num}_layers_updated.csv")
    # #############

    """
    Panel (b) for Fig. 4 experiment.
    Solving single round of 200 snapshots -> 10 snapshots for Fig.4 in manuscript panel (b)
    """
    results_one_round = one_round(TemporalNetwork(hospital_snapshots), int(constants.HOSP_TOTAL_TIME/num_snapshots),
                                  .000015, len(hospital_snapshots), iters=190, order=1, norm=2)
    f = open('results/draft2/hospital_data_updated.json', "w")
    json.dump(results_one_round, f)
    f.close()
    # results_one_round = json.load(open('results/hospital_data.json', "r"))
    plot_one_round(results_one_round)
    plt.show()

    """
    Panel (c) for Fig. 4 experiment.
    Finds normalized compression error against the fully temporal solution for even splits and algorithmic
    as a function of the final number of snapshots in the compressed network.
    Initial 200 pre-compressed snapshots, compressions from 100 to 200 steps.
    """
    # results = error_as_fn_of(TemporalNetwork(hospital_snapshots), .000015, np.arange(100, 200, 1))
    # result_array = np.array([results['iter_range'], results['even_error_norm'], results['opt_error_norm'], results['tce_all']])
    # np.savetxt('results/draft2/hospital_error_integrals_updated.txt', result_array, delimiter=',')
    # result_array = np.loadtxt('./results/draft2/hospital_error_integrals_updated.txt', delimiter=',')
    # results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2],
    #            'tce_all': result_array[3]}
    # plot_error_fn_compressions(results)
    # plt.show()

    """
    Robustness experiment for range of beta.
    Finds the difference between even compression error against temporal solution and algorithmic compression error.
    For a range of beta.
    """
    # Robustness experiment
    iter_range = np.arange(150,200,1)
    betas = np.array([.000005, .00001, .000015, .00002, .000025, .00003, .000035])
    # results = total_error_as_fn_of(betas, iter_range, TemporalNetwork(hospital_snapshots))
    # np.savetxt('hospital_difference.txt', results["difference_matrix"], delimiter=',')
    # np.savetxt('hospital_tce.txt', results["tce"], delimiter=',')
    results = {}
    results["difference_matrix"] = np.loadtxt('./results/hospital_difference.txt', delimiter=',')
    results["tce"] = np.loadtxt('./results/hospital_tce.txt', delimiter=',')
    plt.scatter(iter_range[1:], results["tce"][4][1:], marker='^', color='k', s=9)
    plt.box(False)
    plt.xlabel('Number of compressions')
    plt.semilogy()
    plt.tight_layout()
    plt.show()
    plot_comparison_and_error(results, betas, iter_range, constants.HOSP_TOTAL_TIME, orig_net_length=200)
    plt.show()