import numpy as np
import tensorflow as tf

from layers.similarity import manhattan_distance, cosine_distance, euclidean_distance


class TestDistance(tf.test.TestCase):
    def testManhattanDistance(self):
        with self.cached_session() as test_session:
            x1 = np.array([[1.0, 2.0, 3.0]])
            x2 = np.array([[3.0, 2.0, 1.0]])

            actual_output = test_session.run(manhattan_distance(x1, x2))
            correct_output = [4.0]

            self.assertEqual(actual_output, correct_output)

    def testManhattanDistanceSame(self):
        with self.cached_session() as test_session:
            x1 = np.array([[1.0, 1.0]])
            x2 = np.array([[1.0, 1.0]])

            actual_output = test_session.run(manhattan_distance(x1, x2))
            correct_output = [0.0]

            self.assertEqual(actual_output, correct_output)

    def testManhattanDistance2D(self):
        with self.cached_session() as test_session:
            x1 = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
            x2 = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])

            actual_output = test_session.run(manhattan_distance(x1, x2))
            correct_output = [[3.0], [3.0]]

            self.assertAllEqual(actual_output, correct_output)

    def testCosineDistanceOpposite(self):
        with self.cached_session() as test_session:
            x1 = np.array([[-1.0, -1.0]])
            x2 = np.array([[1.0, 1.0]])

            actual_output = test_session.run(cosine_distance(x1, x2))
            correct_output = [[-1.0]]

            self.assertAllClose(actual_output, correct_output)

    def testCosineDistanceSame(self):
        with self.cached_session() as test_session:
            x1 = np.array([[1.0, 1.0]])
            x2 = np.array([[1.0, 1.0]])

            actual_output = list(test_session.run(cosine_distance(x1, x2)))
            correct_output = [[1.0]]

            self.assertAllClose(actual_output, correct_output)

    def testCosineDistanceOrthogonal(self):
        with self.cached_session() as test_session:
            x1 = np.array([[-1.0, 1.0]])
            x2 = np.array([[1.0, 1.0]])

            actual_output = test_session.run(cosine_distance(x1, x2))
            correct_output = [0.0]

            self.assertEqual(actual_output, correct_output)

    def testCosineDistance2D(self):
        with self.cached_session() as test_session:
            x1 = np.array([[0.0, 1.0], [2.0, 3.0]])
            x2 = np.array([[0.0, 1.0], [-2.0, -3.0]])

            actual_output = test_session.run(cosine_distance(x1, x2))
            correct_output = [[1.0], [-1.0]]

            self.assertAllClose(actual_output, correct_output)

    def testEuclideanDistance(self):
        with self.cached_session() as test_session:
            x1 = np.array([[3.0, 1.0]])
            x2 = np.array([[1.0, 1.0]])

            actual_output = test_session.run(euclidean_distance(x1, x2))
            correct_output = [2.0]

            self.assertEqual(actual_output, correct_output)


if __name__ == "__main__":
    tf.test.main()
