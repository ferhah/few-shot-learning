import unittest
import train
import tempfile
import evaluation.maml_pretrained
import os
import tests.evaluationtest
import tests.datatests
import data
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.data import CombinationMetaDataset
from torchmeta.transforms import ClassSplitter, Categorical
from torchvision import transforms
from maml.datasets import Benchmark
import torch.nn.functional as F
from maml.model import ModelConvMiniImagenet
import logging
from collections import namedtuple
import pickle
import torch


class ArgWrapper():
    def __init__(self):
        self.folder = 'data/miniimagenet'
        self.dataset = 'miniimagenet'
        self.output_folder = 'tests_output/'
        self.num_ways = 10
        self.num_shots = 4
        self.num_shots_test = 4
        self.hidden_size = 64
        self.batch_size = 5
        self.num_steps = 1
        self.num_epochs = 1
        self.num_batches = 1
        self.step_size = 0.1
        self.first_order = False
        self.meta_lr = 0.001
        self.num_workers = 0
        self.verbose = True
        self.use_cuda = False
        self.dbg_save_path = None


class MAMLEvaluationTest(unittest.TestCase):
    def test_basic_functionality(self):
        input_size = 40
        # Train
        args = ArgWrapper()
        with tempfile.TemporaryDirectory() as outfolder:
            args.output_folder = outfolder
            output_folder = train.main(args)
            maml_approach = evaluation.maml_pretrained.MAML(os.path.join(args.output_folder, 'model.th'))
            imagelistfolder = tests.evaluationtest.create_random_imagelists(outfolder, input_size)
            for batchsize in [1, 4, 10, 50, 200]:
                results, _ = evaluation.main.evaluate(imagelistfolder, maml_approach, 400,
                                                               batchsize, 0, None, 'test', '', 'cpu')

    def test_custom_db(self):
        pil_logger = logging.getLogger('PIL')
        pil_logger.setLevel(logging.INFO)
        input_size = 40
        num_ways = 10
        num_shots = 4
        num_shots_test = 4
        batch_size = 1
        num_workers = 0
        with tempfile.TemporaryDirectory() as folder, tempfile.NamedTemporaryFile(mode='w+t') as fp:
            tests.datatests.create_random_imagelist(folder, fp, input_size)
            dataset = data.ImagelistMetaDataset(imagelistname=fp.name,
                                                root='',
                                                transform=transforms.Compose([
                                                    transforms.Resize(84),
                                                    transforms.ToTensor()
                                                ]))
            meta_dataset = CombinationMetaDataset(dataset, num_classes_per_task=num_ways,
                                                  target_transform=Categorical(num_ways),
                                                  dataset_transform=ClassSplitter(shuffle=True,
                                                                                  num_train_per_class=num_shots,
                                                                                  num_test_per_class=num_shots_test))
            args = ArgWrapper()
            args.output_folder = folder
            args.dataset = None
            benchmark = Benchmark(meta_train_dataset=meta_dataset,
                                  meta_val_dataset=meta_dataset,
                                  meta_test_dataset=meta_dataset,
                                  model=ModelConvMiniImagenet(args.num_ways, hidden_size=args.hidden_size),
                                  loss_function=F.cross_entropy)
            train.main(args, benchmark)

    def test_inference(self):
        '''
        'train_inputs': train_inputs,
         'train_targets': train_targets,
         'test_inputs': test_inputs,
         'test_targets': test_targets,
         'test_logits': test_logits,
         'adaptation_results': adaptation_results})
        '''
        pil_logger = logging.getLogger('PIL')
        pil_logger.setLevel(logging.INFO)
        input_size = 40
        args = ArgWrapper()
        with tempfile.TemporaryDirectory() as outfolder:
            args.output_folder = outfolder
            args.dbg_save_path = os.path.join(outfolder, "debug_infos_{}_{}")
            output_folder = train.main(args)
            #os.system("ls {}".format(outfolder))
            with open(args.dbg_save_path.format(0, 0), 'rb') as infile:
                groundtruth = pickle.load(infile)
            maml_approach = evaluation.maml_pretrained.MAML(os.path.join(output_folder, 'model.th'))

            # Check model parameters
            new_model, _ = evaluation.main.finetune_model([[groundtruth['train_inputs'], groundtruth['train_targets']]],
                                                           maml_approach, None)
            for k in groundtruth['params']:
                self.assertTrue(torch.allclose(groundtruth['params'][k], new_model.parameters[k]))

            # To ensure that both parameters contains the same keys.
            for k in new_model.parameters:
                self.assertTrue(torch.allclose(groundtruth['params'][k], new_model.parameters[k]))

            # Check logits
            logits = new_model(groundtruth['test_inputs'])
            self.assertTrue(torch.allclose(groundtruth['test_logits'], logits))

            predictions, _ = evaluation.main.evaluate_model([[groundtruth['test_inputs'], groundtruth['test_targets']]],
                                                            new_model, 'cpu')
            # TODO: more efficient/pythonic comparison
            for pred, gt_logits in zip(predictions, groundtruth['test_logits']):
                for p, gt in zip(pred, gt_logits):
                    self.assertAlmostEqual(p, gt)

            predictions, _ = evaluation.main.evaluate_model([[groundtruth['test_inputs'][0:1],
                                                              groundtruth['test_targets'][0:1]]],
                                                            new_model, 'cpu')
            # TODO: more efficient/pythonic comparison
            print(predictions)
            print(groundtruth['test_logits'].detach().numpy())
            for pred, gt_logits in zip(predictions, groundtruth['test_logits'].detach().numpy()):
                for p, gt in zip(pred, gt_logits):
                    self.assertAlmostEqual(p, gt, places=4)


if __name__ == '__main__':
    unittest.main()
