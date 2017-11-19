import unittest

from my_answers import *

class TestMyAnswers(unittest.TestCase):

    def test_window_transform_series(self):
        odd_nums = np.array([1, 3, 5, 7, 9, 11, 13])
        X,y = window_transform_series(odd_nums,2)

        assert (type(X).__name__ == 'ndarray')
        assert (type(y).__name__ == 'ndarray')
        assert (X.shape == (5, 2))
        assert (y.shape in [(5, 1), (5,)])


if __name__ == '__main__':
    unittest.main()
