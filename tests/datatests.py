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
import imageio
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.data import CombinationMetaDataset
from torchmeta.transforms import ClassSplitter, Categorical
import data

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
        for kdx in range(30):
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



    def test_meta_imagelist(self):
        input_size=40
        num_ways=10
        num_shots=4
        num_shots_test=4
        batch_size=1
        num_workers=0
        self.helper_meta_imagelist(batch_size, input_size, num_shots, num_shots_test, num_ways, num_workers)

        for batch_size in [2, 10]:
            self.helper_meta_imagelist(batch_size, input_size, num_shots, num_shots_test, num_ways, num_workers)
        batch_size = 1
        for num_shots in [8, 10]:
            self.helper_meta_imagelist(batch_size, input_size, num_shots, num_shots_test, num_ways, num_workers)
        num_shots = 4
        for num_shots_test in [8, 10]:
            self.helper_meta_imagelist(batch_size, input_size, num_shots, num_shots_test, num_ways, num_workers)
        num_shots_test = 4
        for num_ways in [4]:
            self.helper_meta_imagelist(batch_size, input_size, num_shots, num_shots_test, num_ways, num_workers)
        for num_workers in [1, 2]:
            self.helper_meta_imagelist(batch_size, input_size, num_shots, num_shots_test, num_ways, num_workers)


    def helper_meta_imagelist(self, batch_size, input_size, num_shots, num_shots_test, num_ways, num_workers):
        with tempfile.TemporaryDirectory() as folder, tempfile.NamedTemporaryFile(mode='w+t') as fp:
            create_random_imagelist(folder, fp, input_size, create_image)
            dataset = data.ImagelistMetaDataset(imagelistname=fp.name,
                                                root='',
                                                transform=transforms.Compose([
                                                    transforms.Resize(input_size),
                                                    transforms.ToTensor()
                                                ]))
            meta_dataset = CombinationMetaDataset(dataset, num_classes_per_task=num_ways,
                                                  target_transform=Categorical(num_ways),
                                                  dataset_transform=ClassSplitter(shuffle=True,
                                                                                  num_train_per_class=num_shots,
                                                                                  num_test_per_class=num_shots_test))
            meta_dataloader = BatchMetaDataLoader(meta_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers,
                                                  pin_memory=True)

            for batch in meta_dataloader:
                batch_data, batch_label = batch['train']
                for img_train, label_train, img_test, label_test in zip(*batch['train'], *batch['test']):
                    classmap = {}
                    for idx in range(img_train.shape[0]):
                        npimg = img_train[idx, ...].detach().numpy()
                        npimg[npimg < 0.001] = 0
                        imagesum = int(np.sum(npimg) * 255)
                        if imagesum not in classmap:
                            classmap[imagesum] = int(label_train[idx])
                        self.assertEqual(classmap[imagesum], int(label_train[idx]))

                    for idx in range(img_test.shape[0]):
                        npimg = img_test[idx, ...].detach().numpy()
                        npimg[npimg < 0.001] = 0
                        imagesum = int(np.sum(npimg) * 255)
                        self.assertEqual(classmap[imagesum], int(label_test[idx]), "Error on {}".format(idx))


if __name__ == '__main__':
    unittest.main()

