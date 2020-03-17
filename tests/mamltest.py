import unittest
import train
import tempfile

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
        # Train
        args = ArgWrapper()
        with tempfile.TemporaryDirectory() as outfolder:
            args.output_folder = outfolder
            train.main(args)


if __name__ == '__main__':
    unittest.main()
