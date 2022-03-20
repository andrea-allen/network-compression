import manuscript.network_generators as netgen

#Conference data:
total_time = 1016440 - 28820
num_layers = 400
beta = .00002
conference_snapshots = netgen.data_network_snapshots('../raw_data/tij_InVS.dat', int(total_time/num_layers), beta, num_layers, 28820)
print(len(conference_snapshots))
print(f'beta is: {conference_snapshots[10].beta}')
for snap in conference_snapshots:
    snap.beta = .2
print(f'beta is: {conference_snapshots[10].beta}')
