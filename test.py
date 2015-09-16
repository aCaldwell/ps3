import unittest
from ps3 import *

class TestPS3Methods(unittest.TestCase):

    def test_best_match_3x3(self):
        a = np.array([[9, 5, 7, 5, 1],
                      [8, 6, 4, 4, 5],
                      [9, 4, 3, 0, 7]])

        patch = np.array([[7, 5, 1],
                          [4, 4, 5],
                          [3, 0, 7]])

        expected_x = 2

        best_x = find_best_match(patch, a)
        self.assertEqual(best_x, expected_x)



    def test_best_match_3x3(self):
        a = np.array([[0,1,2,3,4,5,6,7,8,9],
                      [1,2,3,4,5,6,7,8,9,0],
                      [2,3,4,5,6,7,8,9,0,1],
                      [3,4,5,6,7,8,9,0,1,2],
                      [4,5,6,7,8,9,0,1,2,3]])

        patch = np.array([[3,4,5,6,7],
                          [4,5,6,7,8],
                          [5,6,7,8,9],
                          [6,7,8,9,0],
                          [7,8,9,0,1]])

        expected_x = 3

        best_x = find_best_match(patch, a)
        self.assertEqual(best_x, expected_x)


    def test_rolling_window_3x3(self):
        a = np.array([[0,1,2,3,4],
                      [1,2,3,4,5],
                      [2,3,4,5,6]])

        expected_strides = [[[0,1,2],
                             [1,2,3],
                             [2,3,4]],
                            [[1,2,3],
                             [2,3,4],
                             [3,4,5]],
                            [[2,3,4],
                             [3,4,5],
                             [4,5,6]]]

        strides = rolling_window(a,3)
        self.assertEqual(strides.tolist(), expected_strides)


    def test_rolling_window_5x5(self):
       a = np.array([[0,1,2,3,4,5,6,7,8,9],
                     [1,2,3,4,5,6,7,8,9,0],
                     [2,3,4,5,6,7,8,9,0,1],
                     [3,4,5,6,7,8,9,0,1,2],
                     [4,5,6,7,8,9,0,1,2,3]])

       expected_strides = [[[0,1,2,3,4],
                            [1,2,3,4,5],
                            [2,3,4,5,6],
                            [3,4,5,6,7],
                            [4,5,6,7,8]],
                           [[1,2,3,4,5],
                            [2,3,4,5,6],
                            [3,4,5,6,7],
                            [4,5,6,7,8],
                            [5,6,7,8,9]],
                           [[2,3,4,5,6],
                            [3,4,5,6,7],
                            [4,5,6,7,8],
                            [5,6,7,8,9],
                            [6,7,8,9,0]],
                           [[3,4,5,6,7],
                            [4,5,6,7,8],
                            [5,6,7,8,9],
                            [6,7,8,9,0],
                            [7,8,9,0,1]],
                           [[4,5,6,7,8],
                            [5,6,7,8,9],
                            [6,7,8,9,0],
                            [7,8,9,0,1],
                            [8,9,0,1,2]],
                           [[5,6,7,8,9],
                            [6,7,8,9,0],
                            [7,8,9,0,1],
                            [8,9,0,1,2],
                            [9,0,1,2,3]]]

       strides = rolling_window(a,5)
       self.assertEqual(strides.tolist(), expected_strides)


if __name__ == '__main__':
       unittest.main()
