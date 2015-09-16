import unittest
from ps3 import *

class TestPS3Methods(unittest.TestCase):

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
