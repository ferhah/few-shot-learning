import os
import random
import argparse

def create_evaluation_dataset(root, indir_training, indir_test, outdir,
                              nrclasses_train, nrclasses_val,
                              nr_tests, nrclasses_test, nrsamples,
                              seed_classes, seed_testclasses, seed_samples):
    outdir = os.path.join(outdir, 'db_%d_%d_%d_seed_%d' % (nrclasses_train, nrclasses_val,
                                                           nrsamples, seed_classes))
    classes = os.listdir(indir_training)
    random.seed(seed_classes)
    random.shuffle(classes)
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(os.path.join(outdir, 'meta_training.txt')):
        for classname in classes[:nrclasses_train]:
            with open(os.path.join(outdir, 'meta_training.txt'), 'w') as outfile:
                for filename in os.listdir(os.path.join(indir_training, classname)):
                    outfile.write("%s;%s\n" % (os.path.relpath(os.path.join(indir_training, classname, filename), root),
                                               classname))

        for classname in classes[nrclasses_train:nrclasses_train + nrclasses_val]:
            with open(os.path.join(outdir, 'meta_validation.txt'), 'w') as outfile:
                for filename in os.listdir(os.path.join(indir_training, classname)):
                    outfile.write("%s;%s\n" % (os.path.relpath(os.path.join(indir_training, classname, filename), root),
                                               classname))

    testclassnames = classes[nrclasses_train + nrclasses_val:]
    random.seed(seed_testclasses)
    testclasssubsets = [random.sample(testclassnames, k=nrclasses_test) for _ in range(nr_tests)]
    random.seed(seed_samples)
    for idx, classsubset in enumerate(testclasssubsets):
        outfoldertestbase = os.path.join(outdir, 'test', 'seed_%d_%d_%d' % (seed_testclasses, seed_samples, idx))
        os.makedirs(outfoldertestbase, exist_ok=False)
        with open(os.path.join(outfoldertestbase, 'train.txt'), 'w') as trainfile, open(
                os.path.join(outfoldertestbase, 'val.txt'), 'w') as valfile, open(
            os.path.join(outfoldertestbase, 'test.txt'), 'w') as testfile:
            for classname in classsubset:
                imagenames = os.listdir(os.path.join(indir_training, classname))
                random.shuffle(imagenames)
                for imagename in imagenames[:nrsamples]:
                    trainfile.write("%s;%s\n" % (
                        os.path.relpath(os.path.join(indir_training, classname, imagename), root), classname))

                for imagename in imagenames[nrsamples:]:
                    valfile.write("%s;%s\n" % (
                        os.path.relpath(os.path.join(indir_training, classname, imagename), root), classname))

                for imagename in os.listdir(os.path.join(indir_test, classname)):
                    testfile.write("%s;%s\n" % (
                        os.path.relpath(os.path.join(indir_test, classname, imagename), root), classname))

    return outdir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir', type=str)
    parser.add_argument('trainingdir', type=str)
    parser.add_argument('testdir', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('--nrclasses_train', default=40, type=int)
    parser.add_argument('--nrclasses_val', default=40, type=int)
    parser.add_argument('--nr_tests', default=1, type=int)
    parser.add_argument('--nrclasses_test', default=10, type=int)
    parser.add_argument('--nrsamples', default=4, type=int)
    parser.add_argument('--seed_classes', default=157, type=int, nargs='*')
    parser.add_argument('--seed_testclasses', default=257, type=int, nargs='*')
    parser.add_argument('--seed_samples', default=357, type=int, nargs='*')
    parser.add_argument('--nr_seed_classes', default=None, type=int)
    parser.add_argument('--nr_seed_testclasses', default=None, type=int)
    parser.add_argument('--nr_seed_samples', default=None, type=int)
    config = parser.parse_args()

    if isinstance(config.seed_classes, int):
        config.seed_classes = [config.seed_classes]
    if isinstance(config.seed_testclasses, int):
        config.seed_testclasses = [config.seed_testclasses]
    if isinstance(config.seed_samples, int):
        config.seed_samples = [config.seed_samples]

    if config.nr_seed_classes:
        config.seed_classes = [config.seed_classes[0] + idx for idx in range(config.nr_seed_classes)]
    if config.nr_seed_testclasses:
        config.seed_testclasses = [config.seed_testclasses[0] + idx for idx in range(config.nr_seed_testclasses)]
    if config.nr_seed_samples:
        config.seed_samples = [config.seed_samples[0] + idx for idx in range(config.nr_seed_samples)]

    outdir = None
    for seed_classes in config.seed_classes:
        for seed_testclasses in config.seed_testclasses:
            for seed_samples in config.seed_samples:
                outdir_tmp = create_evaluation_dataset(config.rootdir,
                                                      config.trainingdir,
                                                      config.testdir,
                                                      config.outdir,
                                                      config.nrclasses_train,
                                                      config.nrclasses_val,
                                                      config.nr_tests,
                                                      config.nrclasses_test,
                                                      config.nrsamples,
                                                      seed_classes,
                                                      seed_testclasses,
                                                      seed_samples)
                if outdir:
                    assert outdir == outdir_tmp
                else:
                    outdir = outdir_tmp