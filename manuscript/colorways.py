import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

colors = sns.color_palette("Paired")
rd_bu = sns.color_palette('RdBu', 30)
icefire = sns.color_palette('icefire', 6)
cmr_map = sns.color_palette('CMRmap_r', 6)
type_colors = {'temp': 'grey', 'even': rd_bu[5], 'algo': rd_bu[27]}
type_colors = {'temp': 'grey', 'even': cmr_map[1], 'algo': cmr_map[5]}

def sample_colors(palette):
    fig, ax = plt.subplots(1, len(palette))
    for i in range(len(palette)-2):
        ax[i].scatter(np.arange(200), 100+ np.random.randint(0, 10, 200), color=palette[i])
    for i in range(len(palette)-2, len(palette)):
        xs = np.arange(20)
        ys = np.sin(xs)
        ys_2 = np.cos(xs)
        ax[i].plot(xs, ys, color=palette[i], ls="--")
        ax[i].plot(xs, ys_2, color=palette[i], ls=":")
        ax[i].fill_between(xs, ys, ys_2, color=palette[i], alpha=0.5)
    plt.show()

sample_colors(rd_bu)
sample_colors(icefire)
sample_colors(cmr_map)