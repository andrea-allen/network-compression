import matplotlib.pyplot as plt
import pandas as pd
import manuscript.network_generators as netgen
from manuscript.simulation_helpers import *
from manuscript.plotters import *
import json
import constants

"""
Need to figure out the analysis plan for the next step here,
running another test on the data the full version, with error as a function of compression steps?
"""
## I think just need to run the error_as_a_function_of on another dataset
## to get the other version of Fig. 4 panel (c). That (or for multiple datasets) would then
## mean we could make some kind of giant figure that shows the results for a bunch of datasets.

## Probably still need to do some pre-compression on it

## Analysis plan:
## [done] Select 5-6 datasets
## [done] Run full error compression on them to get results on % of full error that even split vs. algorithmic gets
## TODO Plot results (all together, in a cool new figure)

## Then next: compare another compression method (multilayer compression) to some of the empirical datasets
## once I figure out how to do that


## Code to run for on other datasets for the equivalent of Fig. 4 panel (c) but for multiple datasets
"""
Panel (c) for Fig. 4 experiment.
Finds normalized compression error against the fully temporal solution for even splits and algorithmic
as a function of the final number of snapshots in the compressed network.
Initial 200 pre-compressed snapshots, compressions from 100 to 200 steps.

TODO: Define proper betas through experimentation to run full experiment
Multiple betas for results? Heatmap / matrix of results for multiple betas?
Code for that is also in the realdata_demos section, could be applied after
"""

CONF_DAT = True
HOSP_DAT = True
HS_DAT = True
HT09_DAT = True
OFFICE_DAT = True

import manuscript_fig_plotters

colors = sns.color_palette("Paired")
rd_bu = sns.color_palette('RdBu', 30)
icefire = sns.color_palette('icefire', 6)
cmr_map = sns.color_palette('CMRmap_r', 6)
type_colors = {'temp': 'grey', 'even': rd_bu[5], 'algo': rd_bu[27]}

fig, ax = plt.subplots(5, 1, figsize=(14,20))

boxplot_data = []
boxplot_labels = []


if OFFICE_DAT:
    # file_name = '../raw_data/tij_InVS15_office.dat_'
    # metadata = netgen.dataset_statistics(file_name)
    # num_snapshots = 200
    # # num_snapshots = 1000 # use .000065
    # # num_snapshots = 4000 # use .0002
    # beta = .00001
    # # beta = .000025
    # snapshots = netgen.data_network_snapshots(file_name,
    #                                                      int(metadata['total_time'] / num_snapshots), beta,
    #                                                      int(num_snapshots/2),
    #                                                      metadata['min_timestamp'])
    # results = error_as_fn_of(TemporalNetwork(snapshots), beta, np.arange(100, 200, 1))
    # result_array = np.array([results['iter_range'], results['even_error_norm'], results['opt_error_norm'], results['tce_all']])
    # np.savetxt('results/revisions1/office_error_integrals.txt', result_array, delimiter=',')
    result_array = np.loadtxt('./results/revisions1/office_error_integrals.txt', delimiter=',')
    results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2],
               'tce_all': result_array[3]}
    # plot_error_fn_compressions(results)
    # plt.semilogy()
    # plt.show()
    manuscript_fig_plotters.panel_c_idea2(ax[0], results, type_colors)
    boxplot_data.append(manuscript_fig_plotters.boxplot_revision(results))
    boxplot_labels.append('Office')


if HT09_DAT:
    # file_name = '../raw_data/ht09_contact_list.dat'
    # metadata = netgen.dataset_statistics(file_name)
    # num_snapshots = 200
    # # num_snapshots = 1000 # use .000065
    # # num_snapshots = 4000 # use .0002
    # beta = .00001
    # # beta = .000025
    # snapshots = netgen.data_network_snapshots(file_name,
    #                                                      int(metadata['total_time'] / num_snapshots), beta,
    #                                                      int(num_snapshots/2),
    #                                                      metadata['min_timestamp'])
    # results = error_as_fn_of(TemporalNetwork(snapshots), beta, np.arange(100, 200, 1))
    # result_array = np.array([results['iter_range'], results['even_error_norm'], results['opt_error_norm'], results['tce_all']])
    # np.savetxt('results/revisions1/ht09_error_integrals.txt', result_array, delimiter=',')
    result_array = np.loadtxt('./results/revisions1/ht09_error_integrals.txt', delimiter=',')
    results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2],
               'tce_all': result_array[3]}
    # plot_error_fn_compressions(results)
    # plt.semilogy()
    # plt.show()
    manuscript_fig_plotters.panel_c_idea2(ax[1], results, type_colors)
    boxplot_data.append(manuscript_fig_plotters.boxplot_revision(results))
    boxplot_labels.append('Ht09')


if HS_DAT:
    # hs_metadata = netgen.dataset_statistics('../raw_data/High-School_data_2013.csv')
    # # plt.show()
    # num_snapshots = 200
    # # num_snapshots = 1000 # use .000065
    # # num_snapshots = 4000 # use .0002
    # beta = .000015
    # # beta = .000025
    # snapshots = netgen.data_network_snapshots('../raw_data/High-School_data_2013.csv',
    #                                                      int(hs_metadata['total_time'] / num_snapshots), beta,
    #                                                      int(num_snapshots/2),
    #                                                      hs_metadata['min_timestamp'])
    # results = error_as_fn_of(TemporalNetwork(snapshots), beta, np.arange(100, 200, 1))
    # result_array = np.array([results['iter_range'], results['even_error_norm'], results['opt_error_norm'], results['tce_all']])
    # np.savetxt('results/revisions1/highschool_error_integrals.txt', result_array, delimiter=',')
    result_array = np.loadtxt('./results/revisions1/highschool_error_integrals.txt', delimiter=',')
    results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2],
               'tce_all': result_array[3]}
    # plot_error_fn_compressions(results)
    # plt.semilogy()
    # plt.show()


    manuscript_fig_plotters.panel_c_idea2(ax[2], results, type_colors)
    boxplot_data.append(manuscript_fig_plotters.boxplot_revision(results))
    boxplot_labels.append('High School')

if CONF_DAT:
    # file_name = '../raw_data/tij_InVS.dat'
    # metadata = netgen.dataset_statistics(file_name)
    # num_snapshots = 200
    # # num_snapshots = 1000 # use .000065
    # # num_snapshots = 4000 # use .0002
    # beta = .000025
    # snapshots = netgen.data_network_snapshots(file_name,
    #                                                      int(metadata['total_time'] / num_snapshots), beta,
    #                                                      int(num_snapshots/2),
    #                                                      metadata['min_timestamp'])
    # results = error_as_fn_of(TemporalNetwork(snapshots), beta, np.arange(100, 200, 1))
    # result_array = np.array([results['iter_range'], results['even_error_norm'], results['opt_error_norm'], results['tce_all']])
    # np.savetxt('results/revisions1/conf_error_integrals.txt', result_array, delimiter=',')
    result_array = np.loadtxt('./results/revisions1/conf_error_integrals.txt', delimiter=',')
    results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2],
               'tce_all': result_array[3]}
    # plot_error_fn_compressions(results)
    # plt.semilogy()
    # plt.show()
    manuscript_fig_plotters.panel_c_idea2(ax[3], results, type_colors)
    boxplot_data.append(manuscript_fig_plotters.boxplot_revision(results))
    boxplot_labels.append('Conference')

if HOSP_DAT:
    result_array = np.loadtxt('./results/draft2/hospital_error_integrals_updated.txt', delimiter=',')
    results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2],
               'tce_all': result_array[3]}
    # plot_error_fn_compressions(results)
    # plt.semilogy()
    # plt.show()
    manuscript_fig_plotters.panel_c_idea2(ax[4], results, type_colors)
    boxplot_data.append(manuscript_fig_plotters.boxplot_revision(results))
    boxplot_labels.append('Hospital')

plt.tight_layout()


# plt.savefig('results/revisions1/summary_fig.png')
plt.show()

plt.boxplot(boxplot_data,
            labels=boxplot_labels,
            notch=True,
            vert=True)

np.savetxt('./results/revisions1/boxplot_data.txt', np.array(boxplot_data), delimiter=',')
print(boxplot_labels)
labels = ['hospital', 'high school', 'conference', 'ht09', 'office']

# mdl_ensemble_results = np.array([layers, compression_factor, error_count])
mdl_ensemble_results = np.loadtxt('./results/revisions1/mdl_data_results.txt', delimiter=",")

# plt.scatter()
plt.show()

fig, ax = plt.subplots(1,1)
ax.violinplot(boxplot_data)
parts = ax.violinplot(
        boxplot_data, showmeans=False, showmedians=True, showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('white')
    # pc.set_edgecolor('black', alpha=0.5)
    pc.set_alpha(.5)
mdl_results_ordered = np.array([mdl_ensemble_results[1][4],
                                mdl_ensemble_results[1][3],
                                mdl_ensemble_results[1][1],
                                mdl_ensemble_results[1][2],
                                mdl_ensemble_results[1][0]])
ax.scatter(np.array([1,2,3,4,5]), mdl_results_ordered, marker="*", color="red", label="MDL compression factor", s=100)
ax.set_xticks(np.array([1,2,3,4,5]))
ax.set_xticklabels(boxplot_labels)
plt.show()







