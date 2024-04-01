import numpy as np
import tensorflow as tf

from models.rnn import manhattan_similarity


class TestSimilarity(tf.test.TestCase):
    def testManhattanSimilaritySame(self):
        with self.cached_session() as test_session:
            x1 = np.array([[1.0, 1.0]])
            x2 = np.array([[1.0, 1.0]])
            siamese_lstm_model = manhattan_similarity(x1, x2)

            actual_output = test_session.run(siamese_lstm_model)
            correct_output = [1.0]
            self.assertEqual(actual_output, correct_output)

    def testSimilarity2D(self):
        with self.cached_session() as test_session:
            x1 = np.array([[1.0, 1.0], [1.0, 1.0]])
            x2 = np.array([[1.0, 1.0], [1.0, 1.0]])
            siamese_lstm_model = manhattan_similarity(x1, x2)

            actual_output = test_session.run(siamese_lstm_model)
            correct_output = [[1.0], [1.0]]
            self.assertAllEqual(actual_output, correct_output)
