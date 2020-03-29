import unittest
import tempfile
import os
import imageio
import numpy as np
import evaluation.main
import torch
from torchvision.transforms import ToTensor
import collections

from tests.datatests import create_image


def create_random_imagelists(folder, input_size):
    lists = os.path.join(folder, 'imagelist', 'list01')
    os.makedirs(lists)
    with open(os.path.join(lists, 'test.txt'), 'w') as testfile, open(os.path.join(lists, 'train.txt'),
                                                                      'w') as trainfile:
        for idx in range(10):
            classfoldername = os.path.join(folder, 'class_{}'.format(idx))
            os.mkdir(classfoldername)
            for kdx in range(10):
                imagename = os.path.join(classfoldername, "img_{}.png".format(kdx))
                rndimg = create_image(input_size, idx)
                imageio.imsave(imagename, rndimg)
                testfile.write("{};{}\n".format(imagename, idx))
                if kdx <= 4:
                    trainfile.write("{};{}\n".format(imagename, idx))
    return lists


class Fixed_prediction_approach(object):
    def train(self, dataloader=None, log_dir=None):
        return self, {}

    def __call__(self, imgs, log_dir=None):
        tensor = torch.zeros(imgs.shape[0], 10)
        for idx in range(imgs.shape[0]):
            img = imgs[idx, ...].detach().numpy()
            img[img < 0.001] = 0
            prediction = np.sum(img)
            tensor[idx, int(prediction * 85)] = 1
        return tensor

    @property
    def train_transforms(self):
        return [ToTensor()]

    @property
    def inference_transforms(self):
        return [ToTensor()]


class MainTest(unittest.TestCase):
    def test_evaluate_model(self):
        input_size = 40
        approach = Fixed_prediction_approach()
        classdict = collections.defaultdict(lambda: [])
        with tempfile.TemporaryDirectory() as folder, tempfile.NamedTemporaryFile(mode='w+t', suffix='.txt') as fp:
            for idx in range(10):
                classfoldername = os.path.join(folder, 'class_{}'.format(idx))
                os.mkdir(classfoldername)
                for kdx in range(10):
                    imagename = os.path.join(classfoldername, "img_{}.png".format(kdx))
                    rndimg = create_image(input_size, idx)
                    classdict[idx].append(rndimg)
                    imageio.imsave(imagename, rndimg)
                    fp.write("{};{}\n".format(imagename, idx))
            fp.flush()
            fp.seek(0)
            for batchsize in [1, 4, 10, 50, 200]:
                dataloader = evaluation.main.get_dataloader(fp.name.replace('.txt', ''), '', [ToTensor()],
                                                            10, 0,
                                                            False)
                predictions, labels = evaluation.main.evaluate_model(dataloader, approach, 'cpu')
                results = evaluation.main.calculate_kpis(predictions, labels)
                self.assertAlmostEqual(results['accuracy'], 1)

    def test_evaluate(self):
        input_size = 40
        approach = Fixed_prediction_approach()
        with tempfile.TemporaryDirectory() as folder:
            lists = create_random_imagelists(folder, input_size)
            for batchsize in [1, 4, 10, 50, 200]:
                results, _ = evaluation.main.evaluate(lists, approach, 400,
                                                               batchsize, 0, None, 'test', '', 'cpu')
                self.assertAlmostEqual(results['accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
