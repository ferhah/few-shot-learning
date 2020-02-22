import os
import torch
import torchvision
from data import ImagelistData
import numpy as np

def calculate_kpis(predictions, labels):
    assert len(predictions) == len(labels)
    results = {}
    preds = np.argmax(predictions, 1)
    accuracy = np.sum(preds == labels)
    results['correct'] = float(accuracy)
    results['nr'] = float(len(predictions))
    accuracy = float(accuracy) / float(len(predictions))
    results['accuracy'] = accuracy
    return results


def evaluation(testdir, outfilename, approaches, device, train_batch_size, val_batch_size, num_workers, datadir=None):
    # List of all dirs, containing a small training dataset (/train or /train.txt) and a  test dir (/test or /test.txt).
    evaluationsdirs = [os.path.join(testdir, x) for x in os.listdir(testdir)]
    os.makedirs(os.path.dirname(outfilename), exist_ok=True)

    for eidx, evaluationdir in enumerate(evaluationsdirs):
        for aidx, approach in enumerate(approaches):
            print(evaluationdir, approach)
            # Load Training dataset
            if os.path.exists(os.path.join(evaluationdir, 'train')):
                dataset_train = torchvision.datasets.ImageFolder(os.path.join(evaluationdir, 'train'),
                                                                 torchvision.transforms.Compose(
                                                                     approach.train_transforms))
            else:
                dataset_train = ImagelistData(root=datadir, imagelistname=os.path.join(evaluationdir, 'train.txt'),
                                              transform=torchvision.transforms.Compose(approach.train_transforms))

            dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=train_batch_size,
                                                           shuffle=True, num_workers=num_workers)

            # Train model
            # Call a specific training function
            model = approach.train(dataloader_train, "%s_run_%d_%d" % (outfilename, eidx, aidx))
            print("Start Evaluation")
            # Load Evaluation dataset
            # Standard pytroch loader
            if os.path.exists(os.path.join(evaluationdir, 'test')):
                dataset_test = torchvision.datasets.ImageFolder(os.path.join(evaluationdir, 'test'),
                                                                 torchvision.transforms.Compose(
                                                                     approach.inference_transforms))
            else:
                dataset_test = ImagelistData(root=datadir, imagelistname=os.path.join(evaluationdir, 'test.txt'),
                                              transform=torchvision.transforms.Compose(approach.inference_transforms))

            dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=val_batch_size,
                                                          shuffle=False, num_workers=num_workers)

            predictions = []
            labels = []
            for data, label in dataloader_test:
                # Get prediction form model
                # Call a specific prediction function (standard forward pass)
                data = data.to(device)
                predictions.extend(model(data).cpu().detach().tolist())
                labels.extend(label)

            # Calculate KPIs
            # Compare predictions with ground truth
            results = calculate_kpis(predictions, labels)
            print(eidx, len(evaluationsdirs), evaluationdir, aidx, approach, results)
            with open(outfilename + '.csv', "a") as outfile:
                outfile.write("%s, %d, %s, %d, %s\n" % (evaluationdir, eidx, approach, aidx, results))

