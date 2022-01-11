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

    def test_predict_many(self):
        text_img_file = '../../TextDis benchmark/text/text_%06d.jpg'
        imgs = []
        for i in range(1, 11):
            imgs.append(text_img_file % i)

        net = ClassifierNet()
        text_result = net.predict_many(imgs)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
