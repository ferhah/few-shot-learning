import unittest
import torchvision
from torchvision import transforms
from data import ImagelistData
import torch.utils.data
import torch.testing
import torch
import tempfile
import os
import numpy as np
import collections
import scipy.misc
import imageio


def create_image(input_size, classidx):
    img = np.zeros((input_size, input_size))
    x = np.random.choice(input_size, classidx, replace=False)
    y = np.random.choice(input_size, classidx, replace=True)
    img[x, y] = 1
    assert np.sum(img) == classidx
    return img.astype(np.uint8)


def create_random_image(input_size, _):
    rndimg = np.random.randint(low=0, high=254, size=(input_size, input_size)).astype(np.uint8)
    return rndimg


def create_random_imagelist(folder, fp, input_size, create_image_fkt=None):
    if not create_image_fkt:
        create_image_fkt=create_random_image
    for idx in range(10):
        classfoldername = os.path.join(folder, 'class_{}'.format(idx))
        os.mkdir(classfoldername)
        for kdx in range(10):
            imagename = os.path.join(classfoldername, "img_{}.png".format(kdx))
            rndimg = create_image_fkt(input_size, idx)
            imageio.imsave(imagename, rndimg)
            fp.write("{};{}\n".format(imagename, idx))
    fp.flush()
    fp.seek(0)
    

class DataloaderTest(unittest.TestCase):
    def test_imagelist(self):
        input_size=224
        # Create Files
        with tempfile.TemporaryDirectory() as folder, tempfile.NamedTemporaryFile(mode='w+t') as fp:
            create_random_imagelist(folder, fp, input_size)

            # Create dataloader
            dataset_folder = torchvision.datasets.ImageFolder(folder,
                transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ]))
            dataset_list = ImagelistData(root='',
                                         imagelistname=fp.name,
                                         transform=transforms.Compose([
                                             transforms.Resize(input_size),
                                             transforms.CenterCrop(input_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])
                                         ]))
            dataloader_folder = torch.utils.data.DataLoader(dataset_folder, batch_size=1,
                                                               shuffle=True, num_workers=3)
            dataloader_list = torch.utils.data.DataLoader(dataset_list, batch_size=1,
                                                               shuffle=True, num_workers=3)

            # Compare dataloader
            folder_images = collections.defaultdict(lambda: [])
            for img, label in dataloader_folder:
                folder_images[label.item()].append(img)

            imagelist_images = collections.defaultdict(lambda: [])
            for img, label in dataloader_list:
                imagelist_images[label.item()].append(img)

            for classname in folder_images:
                reference_image = folder_images[classname][0]
                for classname_list in imagelist_images:
                    for img in imagelist_images[classname_list]:
                        if torch.all(torch.eq(reference_image, img)):
                            break
                    else:
                        continue
                    break

                self.assertEqual(len(folder_images[classname]), len(imagelist_images[classname_list]))
                for f_img in folder_images[classname]:
                    for l_img in imagelist_images[classname_list]:
                        if torch.all(torch.eq(f_img, l_img)):
                            break
                    else:
                        self.assertTrue(False, "Image not found")

if __name__ == '__main__':
    unittest.main()
