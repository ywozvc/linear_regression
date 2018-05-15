import unittest
import linear_regression


filename = "testdata.csv"
split = 0.6

class linearRegressionTestCase(unittest.TestCase):


    def test_load_csv(self):
        self.assertIsInstance(linear_regression.load_csv(filename),list)

    def test_mean(self):
        values = [x for x in range(1,10)]
        self.assertAlmostEqual(linear_regression.mean(values), 5)

    def test_train_test_split(self):

        self.assertIsInstance(linear_regression.train_test_split(linear_regression.load_csv(filename), split), list, list)
        

if __name__ == '__main__':
    unittest.main()
unittest.main()


