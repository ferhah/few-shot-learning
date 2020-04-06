import unittest
import train
import tempfile
import evaluation.maml_pretrained
import os
import tests.evaluationtest

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


if __name__ == '__main__':
    unittest.main()
