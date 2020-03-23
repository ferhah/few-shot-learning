import unittest
import train
import tempfile
import evaluation.maml_pretrained
import os
import tests.evaluationtest

class ArgWrapper():
    folder='data/miniimagenet'
    dataset='miniimagenet'
    output_folder='tests_output/'
    num_ways=10
    num_shots=4
    num_shots_test=4
    hidden_size=64
    batch_size=5
    num_steps=1
    num_epochs=1
    num_batches=1
    step_size=0.1
    first_order=False
    meta_lr=0.001
    num_workers=0
    verbose=True
    use_cuda=False

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
                self.assertAlmostEqual(results['accuracy'], 1)

if __name__ == '__main__':
    unittest.main()
