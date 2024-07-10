import unittest

from server.train import main


class TestTrain(unittest.TestCase):
    def test_main(self):
        main()

        