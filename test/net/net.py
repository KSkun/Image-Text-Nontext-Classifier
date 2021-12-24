import unittest

from src.net.net import ClassifierNet


class TestClassifierNet(unittest.TestCase):
    def test_predict(self):
        text_img_file = '../../TextDis benchmark/text/text_000005.jpg'
        nontext_img_file = '../../TextDis benchmark/nonText/nonText_000005.jpg'

        net = ClassifierNet()
        text_result = net.predict(text_img_file)
        nontext_result = net.predict(nontext_img_file)

        self.assertTrue(text_result and not nontext_result)


if __name__ == '__main__':
    unittest.main()
