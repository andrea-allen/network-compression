import unittest
from src.compression import *


class CompressorTests(unittest.TestCase):
    def test_compress_basecase(self):
        # SETUP
        array_one = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        array_two = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]])
        A = Snapshot(1, 2, .5,
                  array_one)
        A2 = Snapshot(2, 3, .5, array_one)
        B = Snapshot(3, 4, .5, array_two)
        B2 = Snapshot(4, 5, .5, array_two)
        temporal_net = TemporalNetwork([A, A2, B, B2])
        # ACT
        compressed_net, error_chosen = Compressor.compress(temporal_net, how='optimal')
        # ASSERT
        self.assertTrue(TemporalNetwork([Snapshot(1, 3, .5, array_one), B, B2]).equals(compressed_net))

    def test_compress(self):
        # SETUP
        array_one = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        array_two = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]])
        A = Snapshot(1, 2, .5, array_one)
        A2 = Snapshot(2, 3, .5, array_one)
        B = Snapshot(3, 4, .5, array_two)
        B2 = Snapshot(4, 5, .5, array_two)
        A3 = Snapshot(5, 6, .5, array_one)
        B3 = Snapshot(6, 7, .5, array_two)
        A4 = Snapshot(7, 8, .5, array_one)
        A5 = Snapshot(8, 9, .5, array_one)
        temporal_net = TemporalNetwork([A, A2, B, B2, A3, B3, A4, A5])
        combined_snapshot = Snapshot(1,3, .5, (A.A+B.A)/2)
        # ACT
        compressed_net, error_chosen = Compressor.compress(temporal_net, iterations=3, how='optimal')
        # ASSERT
        self.assertTrue(TemporalNetwork([Snapshot(1, 3, .5, array_one),
                                         Snapshot(3, 5, .5, array_two),
                                         A3, B3, Snapshot(7, 9, .5, array_one)]).equals(compressed_net))
        self.assertFalse(TemporalNetwork([A, combined_snapshot, combined_snapshot, combined_snapshot, A]).equals(compressed_net))

    def test_compress_odd(self):
        # SETUP
        array_one = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        array_two = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]])
        A = Snapshot(1, 2, .5, array_one)
        A2 = Snapshot(2, 3, .5, array_one)
        A3 = Snapshot(5, 6, .5, array_one)
        A4 = Snapshot(7, 8, .5, array_one)
        A5 = Snapshot(8, 9, .5, array_one)
        B = Snapshot(3, 4, .5, array_two)
        B2 = Snapshot(4, 5, .5, array_two)
        B3 = Snapshot(6, 7, .5, array_two)
        B4 = Snapshot(9, 10, .5, array_two)
        temporal_net = TemporalNetwork([A, A2, B, B2, A3, B3, A4, A5, B4])
        # ACT
        compressed_net, error_chosen = Compressor.compress(temporal_net, iterations=3, how='optimal')
        # ASSERT
        self.assertTrue(TemporalNetwork([Snapshot(1,3,.5,array_one),
                                         Snapshot(3,5,.5,array_two),
                                         A3,
                                         B3,
                                         Snapshot(7,9, .5, array_one),
                                         B4]).equals(compressed_net))

    def test_compress_odd_shift(self):
        # SETUP
        array_one = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        array_two = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]])

        temporal_net = TemporalNetwork([Snapshot(1, 2, .5, array_two),
                                        Snapshot(2,3, .5,array_one),
                                        Snapshot(3,4, .5,array_one),
                                        Snapshot(4,5, .5, array_two),
                                        Snapshot(5,6, .5, array_two),
                                        Snapshot(6,7, .5,array_one),
                                        Snapshot(7,8, .5, array_two),
                                        Snapshot(8,9, .5,array_one),
                                        Snapshot(9,10, .5,array_one)])

        expected_compressed_net = TemporalNetwork([Snapshot(1, 2, .5, array_two),
                                                   Snapshot(2, 4, .5, array_one),
                                                   Snapshot(4, 6, .5, array_two),
                                                   Snapshot(6, 7, .5, array_one),
                                                   Snapshot(7, 8, .5, array_two),
                                                   Snapshot(8, 10, .5, array_one)])
        # ACT
        compressed_net, total_error = Compressor.compress(temporal_net, iterations=3, how='optimal')
        # ASSERT
        self.assertTrue(expected_compressed_net.equals(compressed_net))

        # ACT
        compressed_net, total_error = Compressor.compress(temporal_net, iterations=4, how='optimal')
        # ASSERT
        self.assertTrue(TemporalNetwork([
            Snapshot(1, 2, .5, array_two),
            Snapshot(2, 4, .5, array_one),
            Snapshot(4, 6, .5, array_two),
            Snapshot(6, 8, .5, (array_one+array_two)/2),
            Snapshot(8,10,.5, array_one)])
                        .equals(compressed_net))

    def test_compress_twice(self):
        # SETUP
        some_array = np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]])
        other_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]])
        A = Snapshot(1, 2, 0.8,
                  some_array)
        B = Snapshot(2, 3, 0.8,
                  some_array)
        C = Snapshot(3, 4, 0.8,
                  other_array)
        D = Snapshot(4, 5, 0.8,
                  other_array)
        E = Snapshot(5, 6, 0.8,
                  some_array)
        F = Snapshot(6, 7, 0.8,
                  some_array)
        temporal_net = TemporalNetwork([A, B, C, D, E, F])

        # ACT
        compressed_net_once, _ = Compressor.compress(temporal_net, how='optimal')
        # ASSERT
        # should_be_network = TemporalNetwork([Snapshot(1, 3, 0.8, some_array),
        #                                      C, D,
        #                                      E, F])
        # Compresses the middle error based on the median strategy
        should_be_network = TemporalNetwork([Snapshot(1, 3, 0.8, some_array),
                                             C, D,
                                             E, F])
        self.assertTrue(should_be_network.equals(compressed_net_once))

        # ACT (compress again)
        compressed_net_twice, _ = Compressor.compress(compressed_net_once, how='optimal')

        # ASSERT
        should_be_network = TemporalNetwork([
            Snapshot(1, 3, 0.8, some_array),
            Snapshot(3,5, 0.8, other_array),
            E, F
        ])
        self.assertTrue(should_be_network.equals(compressed_net_twice))

    def test_compress_evenly(self):
        # SETUP
        some_array = np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]])
        other_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]])
        A = Snapshot(1, 2, 0.8,
                  some_array)
        B = Snapshot(2, 3, 0.8,
                  some_array)
        C = Snapshot(3, 4, 0.8,
                  other_array)
        D = Snapshot(4, 5, 0.8,
                  other_array)
        E = Snapshot(5, 6, 0.8,
                  some_array)
        F = Snapshot(6, 7, 0.8,
                  some_array)
        temporal_net = TemporalNetwork([A, B, C, D, E, F])

        # ACT
        compressed_net = Compressor.compress(temporal_net, iterations=3, how='even')
        # ASSERT
        should_be_network = TemporalNetwork([Snapshot(1,3, 0.8, some_array),
                                             Snapshot(3,5, 0.8, other_array),
                                             Snapshot(5,7, 0.8, some_array)])

        # ASSERT
        self.assertTrue(compressed_net.equals(should_be_network))

    def test_compress_iteratively(self):
        # SETUP
        some_array = np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]])
        other_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]])
        A = Snapshot(1, 2, 0.8,
                  some_array)
        B = Snapshot(2, 3, 0.8,
                  some_array)
        C = Snapshot(3, 4, 0.8,
                  other_array)
        D = Snapshot(4, 5, 0.8,
                  other_array)
        E = Snapshot(5, 6, 0.8,
                  some_array)
        F = Snapshot(6, 7, 0.8,
                  some_array)
        temporal_net = TemporalNetwork([A, B, C, D, E, F])

        # ACT
        compressed_net_level2, _ = Compressor.compress(temporal_net, iterations=2, how='optimal')
        compressed_net_again, _ = Compressor.compress(compressed_net_level2, iterations=2, how='optimal')

        compressed_level2_twice, _ = Compressor.compress(temporal_net, iterations=4, how='optimal')

        # ASSERT
        self.assertTrue(compressed_net_again.equals(compressed_level2_twice))




    def test_epsilon_trivial(self):
        A = Snapshot(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        B = Snapshot(2, 3, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))

        error = Compressor.epsilon(A, B)
        self.assertEqual(0, error)

    def test_epsilon_nontrivial(self):
        A = Snapshot(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        B = Snapshot(2, 3, 0.8,
                  np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]]))

        error = Compressor.epsilon(A, B)
        self.assertNotEqual(0, error)

    def test_aggregate(self):
        A = Snapshot(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        B = Snapshot(2, 3, 0.8,
                  np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]]))
        aggregated = Compressor.aggregate(A, B)
        self.assertTrue(aggregated.equals(Snapshot(A.start_time,
                                                B.end_time,
                                                0.8,
                                                (A.A + B.A)/2)))

    def test_aggregate_by_duration(self):
        A = Snapshot(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        B = Snapshot(2, 4, 0.8,
                  np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]]))
        aggregated = Compressor.aggregate(A, B)
        self.assertTrue(aggregated.equals(Snapshot(A.start_time,
                                                B.end_time,
                                                0.8,
                                                (2*B.A + A.A)/3)))

        # Base case unequal duration aggregation:
        A_longer = Snapshot(2, 7, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        aggregated_base = Compressor.aggregate(A, A_longer)
        self.assertTrue(aggregated_base.equals(Snapshot(1,7,0.8,
                                                np.array([[0, 1, 0, 1],
                                                          [1, 0, 1, 0],
                                                          [0, 1, 0, 0],
                                                          [1, 0, 0, 0]])
                                                )))
        self.assertTrue(aggregated_base.equals(Snapshot(1,7,0.8, (A.A + 5*A_longer.A)/6)))

    def test_time_ordering(self):
        # SETUP
        A = Snapshot(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        same_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]])
        B = Snapshot(2, 3, 0.8,
                  same_array)
        C = Snapshot(3, 4, 0.8,
                  same_array)
        D = Snapshot(4, 5, 0.8,
                  same_array)
        temporal_net = TemporalNetwork([A, B, C, D])

        # ACT
        compressed_net, _ = Compressor.compress(temporal_net, iterations=2, how='optimal')
        should_be_network = TemporalNetwork([Snapshot(1, 2, 0.8, A.A),
                                             Snapshot(2, 5, 0.8, same_array)])
        should_not_be = TemporalNetwork([Snapshot(1, 2, 0.8, (A.A+B.A)/2),
                                        Snapshot(3, 4, 0.8, same_array)])

        # ASSERT
        self.assertTrue(should_be_network.equals(compressed_net))
        self.assertFalse(should_not_be.equals(compressed_net))

    def test_total_aggregate(self):
        # Test that compressing at level 1, length snapshots iterations, gives the total sum normalized
        # SETUP
        some_array = np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]])
        other_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]])
        third_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 2],
                            [0, 2, 0, 0],
                            [1, 2, 0, 0]])
        A = Snapshot(1, 2, 0.8,
                  some_array)
        B = Snapshot(2, 3, 0.8,
                  other_array)
        C = Snapshot(3, 4, 0.8,
                  third_array)
        D = Snapshot(4, 5, 0.8,
                  other_array)
        E = Snapshot(5, 6, 0.8,
                  some_array)
        F = Snapshot(6, 7, 0.8,
                  some_array)
        temporal_net = TemporalNetwork([A, B, C, D, E, F])

        # ACT
        # optimal
        compressed_net, _ = Compressor.compress(temporal_net, iterations=5, how='optimal')
        # random
        even_net = Compressor.compress(temporal_net, iterations=5, how='even')

        # ASSERT
        self.assertTrue(np.array_equal(compressed_net.snapshots[0].A, (some_array+other_array+third_array+other_array
                                                                    +some_array+some_array)/6))
        self.assertTrue(np.array_equal(even_net.snapshots[0].A, (some_array + other_array + third_array + other_array
                                                                    + some_array + some_array) / 6))
