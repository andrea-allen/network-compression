from src.temporalnetwork import *
from numpy import linalg as LA


class Compressor:
    @staticmethod
    def compress(temporal_net, iterations=1, how='even', error_type='combined', order=3, norm=None):
        """
        Takes an ordered list of pairs and returns the compressed versions. Sole layers are returned as-is.
        :param error_type: 'terminal' or 'halftime' or 'combined' (default, best)
        :param how: 'optimal' or 'even'
        :param iterations: how many rounds of compression to perform
        :param temporal_net: TemporalNetwork object
        :return: Ordered list of layer pairs or singles
        """
        to_compress = int(iterations)
        if how.lower() == 'even':
            desired_num_snapshots = temporal_net.length - to_compress
            new_networks = TemporalNetwork(Compressor._even_compression(temporal_net, desired_num_snapshots))
            return new_networks
        current_net = temporal_net
        total_chosen_error = 0
        for r in range(iterations):
            new_net, chosen_error = Compressor._compress_round(current_net, how, error_type, order, norm)
            current_net = new_net
            total_chosen_error += chosen_error
        return current_net, total_chosen_error

    @staticmethod
    def _compress_round(temporal_net, how, error_type, order, norm):
        if how.lower() == 'optimal':
            new_networks, chosen_error = Compressor._optimal_compression(temporal_net, error_type, order, norm)
            return TemporalNetwork(new_networks), chosen_error
        elif how.lower() == 'random':
            new_networks = Compressor._random_compression(temporal_net)
            return TemporalNetwork(new_networks)

    @staticmethod
    def _optimal_compression(temporal_net, error_type='terminal', order=3, norm=None):
        all_pairs = temporal_net.get_ordered_pairs()
        epsilons = Compressor.pairwise_epsilon(all_pairs, error_type, order, norm)
        best_keys = [key for (key, val) in epsilons.items() if val == min(epsilons.values())]
        best_key = best_keys[0]
        # print(f"selecting best key {best_key} with epsilon {min(epsilons.values())}")
        chosen_error = min(epsilons.values())
        # print(f"num best keys was {len(best_keys)}")
        pairs = all_pairs
        new_networks = []
        snapshots_getting_compressed = [pairs[best_key][0]]
        the_snapshots = temporal_net.snapshots
        for id, layer in enumerate(the_snapshots):
            if the_snapshots[id] in snapshots_getting_compressed:
                new_networks.append(Compressor.aggregate(the_snapshots[id], the_snapshots[id + 1]))
            elif the_snapshots[id - 1] not in snapshots_getting_compressed:
                new_networks.append(the_snapshots[id])
        return new_networks, chosen_error

    @staticmethod
    def _random_compression(temporal_net):
        random_idx = np.random.randint(0, temporal_net.length - 1)
        new_networks = []
        for idx, snapshot in enumerate(temporal_net.snapshots):
            if idx == random_idx:
                new_networks.append(Compressor.aggregate(temporal_net.snapshots[idx], temporal_net.snapshots[idx + 1]))
            if idx - 1 == random_idx:
                pass
            elif idx != random_idx:
                new_networks.append(temporal_net.snapshots[idx])
        return new_networks

    @staticmethod
    def _even_compression(temporal_net, desired_num_layers):
        new_networks = []
        snapshots_boundaries = list([int(i) for i in np.linspace(0, temporal_net.length, desired_num_layers + 1)])
        idx = 0
        for b in range(len(snapshots_boundaries) - 1):
            current_aggregate = temporal_net.snapshots[snapshots_boundaries[b]]
            idx += 1
            while idx < snapshots_boundaries[b + 1]:
                current_aggregate = Compressor.aggregate(current_aggregate, temporal_net.snapshots[idx])
                idx += 1
            new_networks.append(current_aggregate)
        return new_networks

    @staticmethod
    def pairwise_epsilon(pairs, error_type='terminal', order=3, norm=None):
        epsilon_values = {}
        for idx, pair in enumerate(pairs):
            try:
                epsilon = Compressor.epsilon(pair[0], pair[1], error_type, order, norm)
                epsilon_values[idx] = epsilon
            except TypeError:
                pass  # if pair is a single
        return epsilon_values

    @staticmethod
    def aggregate(snapshot, other_snapshot):
        total_duration = other_snapshot.end_time - snapshot.start_time
        new_adjacency_matrix = ((snapshot.end_time - snapshot.start_time) * snapshot.A
                                + (
                                            other_snapshot.end_time - other_snapshot.start_time) * other_snapshot.A) / total_duration
        return Snapshot(snapshot.start_time, other_snapshot.end_time, snapshot.beta, new_adjacency_matrix)

    @staticmethod
    def commutator(A, B):
        return (A @ B) - (B @ A)

    @staticmethod
    def linear_difference(snapshot, other_snapshot):
        A = snapshot.scaled_matrix()
        B = other_snapshot.scaled_matrix()
        comm = Compressor.commutator(B, A)
        return comm/2

    @staticmethod
    def second_order_difference(snapshot, other_snapshot):
        A = snapshot.scaled_matrix()
        B = other_snapshot.scaled_matrix()
        comm = Compressor.commutator(B, A)
        return (comm / 2) + (B @ comm) / 2 + (A @ comm) / 2 + (comm @ B) / 2 + (comm @ A) / 2 + (comm @ comm) / 4


    @staticmethod
    def third_order_difference(snapshot, other_snapshot):
        A = snapshot.scaled_matrix()
        B = other_snapshot.scaled_matrix()
        BA = B @ A
        AB = A @ B
        diff_matrix = (1 / 2) * (BA - AB) \
                      + (1 / 12) * ((B @ BA) + (A @ AB)
                                    + (A @ (B @ B)) + (B @ (A @ A))) \
                      - (1 / 6) * ((B @ AB) + (A @ BA))
        return diff_matrix

    @staticmethod
    def epsilon(snapshot, other_snapshot, error_type='combined', order=3, norm=None):
        if error_type.lower() == 'combined':
            A = snapshot.scaled_matrix()
            if order==3:
                error_terminal = Compressor.third_order_difference(snapshot, other_snapshot)
            elif order==2:
                error_terminal = Compressor.second_order_difference(snapshot, other_snapshot)
            elif order==1:
                error_terminal = Compressor.linear_difference(snapshot, other_snapshot)
            P0 = snapshot.dd_normalized

            if norm is None:
                e_terminal = np.sum(np.abs(error_terminal).dot(P0))
            # OPERATOR NORM
            else:
                e_terminal = LA.norm(error_terminal, norm)

            agg_mat = (snapshot.duration * snapshot.A + other_snapshot.duration * other_snapshot.A) / (
                        snapshot.duration + other_snapshot.duration)
            if order==3:
                approx_A_temp = A + (A @ A) / 2 + (A@A@A)/6
            elif order==2:
                approx_A_temp = A + (A @ A) / 2
            elif order==1:
                approx_A_temp = A

            agg_scaled = snapshot.beta * snapshot.duration * agg_mat

            if order==3:
                approx_Agg = agg_scaled + (agg_scaled @ agg_scaled) / 2 + (agg_scaled@agg_scaled@agg_scaled)/6
            elif order==2:
                approx_Agg = agg_scaled + (agg_scaled @ agg_scaled) / 2
            elif order==1:
                approx_Agg = agg_scaled

            error = approx_A_temp - approx_Agg
            if norm is None:
                e_halftime = np.sum(np.abs(error).dot(P0))
            # OPERATOR NORM
            else:
                e_halftime = LA.norm(error, norm)
            if np.isnan(e_terminal) or np.isnan(e_halftime):
                print('STOP, nans')
            return (e_halftime + e_terminal) * (snapshot.duration + other_snapshot.duration)

        elif error_type.lower() == 'terminal':
            A = snapshot.scaled_matrix()
            B = other_snapshot.scaled_matrix()
            BA = B @ A
            AB = A @ B
            error = (1 / 2) * (BA - AB) \
                    + (1 / 12) * ((B @ BA) + (A @ AB)
                                  + (A @ (B @ B)) + (B @ (A @ A))) \
                    - (1 / 6) * ((B @ AB) + (A @ BA))
        elif error_type.lower() == 'halftime':

            agg_mat = (snapshot.duration * snapshot.A + other_snapshot.duration * other_snapshot.A) / (
                        snapshot.duration + other_snapshot.duration)
            A_scaled = snapshot.beta * snapshot.duration * snapshot.A
            approx_A_temp = A_scaled + (A_scaled @ A_scaled) / 2 + (A_scaled @ A_scaled @ A_scaled) / 6
            agg_scaled = snapshot.beta * snapshot.duration * agg_mat
            approx_Agg = agg_scaled + (agg_scaled @ agg_scaled) / 2 + (agg_scaled @ agg_scaled @ agg_scaled) / 6
            D = approx_A_temp - approx_Agg
            error = D

        P0 = snapshot.dd_normalized # THIS IS NOT THE DD, THIS IS THE NORMALIZED DEGREE OF THE NODE

        total_infect_diff = np.sum(np.abs(error).dot(P0))
        total_infect_diff = total_infect_diff * (snapshot.duration + other_snapshot.duration)

        return total_infect_diff
