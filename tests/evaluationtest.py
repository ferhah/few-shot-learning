import unittest
import tempfile
import os
import imageio
import numpy as np
import evaluation.main
import torch
from torchvision.transforms import ToTensor
import collections


def create_image(input_size, classidx):
    img = np.zeros((input_size, input_size))
    x = np.random.choice(input_size, classidx, replace=False)
    y = np.random.choice(input_size, classidx, replace=True)
    img[x, y] = 1
    assert np.sum(img) == classidx
    return img.astype(np.uint8)


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
    def test_main(self):
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


if __name__ == '__main__':
    unittest.main()
