# network-compression

This repository includes source code and supporting work for manuscript in preparation.
Some of the original data and results in raw form may not be included in the repository. Please reach out with questions.

Please cite this code and the original manuscript (in preparation) if used.


## src/temporalnetwork.py
Includes source code for temporal network and snapshot objects.

## src/compression.py
Objects for performing algorithmic compression of temporal networks.

## src/solvers.py
Source code for initializing and solving a deterministic system
of ordinary differential equations of an SI spreading process on
a temporal network.

## manuscript/
Contains supporting computations, experiments, and plotting code
for manuscript (in preparation).

Please contact Andrea Allen at andrea2allen (at) gmail.com for use questions.

Use examples: 

See `manuscript/synthetic_demos.py` for demonstration.

![Figure 1](./manuscript/readme_fig1.png)
The figure above shows an example application of the temporal compression framework
(see [`manuscript/synthetic_demos.py`](./manuscript/synthetic_demos.py) for the source code).
Here, we begin with 50 snapshots each with a slightly different random network
generated from a collection of motifs (Erdos-Renyi, Barbell graphs, configuration models, etc.)
inidicated by the grey lines in the top panel.

First using an even segmentation strategy to compress the snapshots, the red dashed lines
in the top panel show the new boundaries of the compressed snapshots. Then, using
our compression algorithm, the blue dashed lines show the boundaries of the compressed
networks under our algorithm.

The grey time series in the top panel is the solution of a disease spread process
on the original temporal network, where the network is switched at each time increment.
The blue and red dashed lines show the disease spread time series solution on the resulting
snapshots from the two compression regimes, even segmentation vs. our algorithm.

The lower panel shows the integral of the error between the red and blue time series and the
grey temporal solution. You can see that the algorithmic compression performs
much closer to the original fully temporal dynamics than an even compression.

## Data source

For validation of the method and data used in the manuscript, data was provided courtesy of the *SocioPatterns* project:
http://www.sociopatterns.org/datasets/

Full citations for the datasets used:

"Can co-location be used as a proxy for face-to-face contacts?", M. Génois and A. Barrat, EPJ Data Science 7, 11 (2018).

R. Mastrandrea, J. Fournet, A. Barrat,
Contact patterns in a high school: a comparison between data collected using wearable sensors, contact diaries and friendship surveys.
PLoS ONE 10(9): e0136497 (2015)

J. Fournet, A. Barrat, Contact patterns among high school students,
PLoS ONE 9(9):e107878 (2014). 

P. Vanhems et al., Estimating Potential Infection Transmission Routes in Hospital Wards Using Wearable Proximity Sensors, PLoS ONE 8(9): e73970 (2013). 

L. Isella et al.,  What’s in a crowd? Analysis of face-to-face behavioral networks, Journal of Theoretical Biology 271, 166 (2011).