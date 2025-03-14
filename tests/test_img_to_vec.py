import unittest
from PIL import Image
import numpy

from img2vec_pytorch.img_to_vec import Img2Vec

class TestImg2Vec(unittest.TestCase):
    def test_default(self):
        img2vec = Img2Vec().download_model()
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(512, vec.size)
        img.close()

    def test_resnet101(self):
        img2vec = Img2Vec(model='resnet101').download_model()
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(2048, vec.size)
        img.close()

    def test_resnet152(self):
        img2vec = Img2Vec(model='resnet152').download_model()
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(2048, vec.size)
        img.close()

    def test_alexnet(self):
        img2vec = Img2Vec(model='alexnet').download_model()
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(4096, vec.size)
        img.close()

    def test_vgg(self):
        img2vec = Img2Vec(model='vgg').download_model()
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(4096, vec.size)
        img.close()

    def test_densenet(self):
        img2vec = Img2Vec(model='densenet').download_model()
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1024, vec.size)
        img.close()

    def test_efficientnet_b0(self):
        img2vec = Img2Vec(model='efficientnet_b0').download_model()
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1280, vec.size)
        img.close()

    def test_efficientnet_b1(self):
        img2vec = Img2Vec(model='efficientnet_b1').download_model()
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1280, vec.size)
        img.close()

    def test_efficientnet_b2(self):
        img2vec = Img2Vec(model='efficientnet_b2').download_model()
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1408, vec.size)
        img.close()

    def test_efficientnet_b3(self):
        img2vec = Img2Vec(model='efficientnet_b3').download_model()
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1536, vec.size)
        img.close()

    def test_efficientnet_b4(self):
        img2vec = Img2Vec(model='efficientnet_b4').download_model()
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1792, vec.size)
        img.close()

    def test_efficientnet_b5(self):
        img2vec = Img2Vec(model='efficientnet_b5').download_model()
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(2048, vec.size)
        img.close()

    def test_efficientnet_b6(self):
        img2vec = Img2Vec(model='efficientnet_b6').download_model()
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(2304, vec.size)
        img.close()

    def test_efficientnet_b7(self):
        img2vec = Img2Vec(model='efficientnet_b7').download_model()
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(2560, vec.size)
        img.close()

if __name__ == "__main__":
    unittest.main()