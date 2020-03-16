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

def get_dataloader(path, datadir, transforms, batch_size, num_workers, shuffle):
    if os.path.exists(path):
        dataset = torchvision.datasets.ImageFolder(path, torchvision.transforms.Compose(transforms))
    else:
        dataset = ImagelistData(root=datadir, imagelistname=path + '.txt',
                                transform=torchvision.transforms.Compose(transforms))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
    return dataloader


def finetune_model(dataloader, approach, log_dir):
    return approach.train(dataloader, log_dir)


def evaluate_model(dataloader, model, device):
    predictions = []
    labels = []
    for data, label in dataloader:
        # Get prediction form model
        # Call a specific prediction function (standard forward pass)
        data = data.to(device)
        predictions.extend(model(data).cpu().detach().tolist())
        labels.extend(label)
    return predictions, labels


def evaluate(evaluationdir, approach, train_batch_size, val_batch_size, num_workers, log_dir, test_name, datadir,
             device):
    dataloader_train = get_dataloader(os.path.join(evaluationdir, 'train'), datadir, approach.train_transforms,
                                      train_batch_size, num_workers, True)
    dataloader_test= get_dataloader(os.path.join(evaluationdir, test_name), datadir, approach.inference_transforms,
                                    val_batch_size, num_workers, False)
    model, train_kpis = finetune_model(dataloader_train, approach, log_dir)
    predictions, labels = evaluate_model(dataloader_test, model, device)
    # Calculate KPIs
    results = calculate_kpis(predictions, labels)
    return results, train_kpis


def evaluation(testdir, outfilename, approaches, device, train_batch_size, val_batch_size, num_workers, datadir=None):
    # List of all dirs, containing a small training dataset (/train or /train.txt) and a  test dir (/test or /test.txt).
    evaluationsdirs = [os.path.join(testdir, x) for x in os.listdir(testdir)]
    os.makedirs(os.path.dirname(outfilename), exist_ok=True)

    for eidx, evaluationdir in enumerate(evaluationsdirs):
        for aidx, approach in enumerate(approaches):
            print(evaluationdir, approach)
            results, train_kpis = evaluate(evaluationdir, approach, train_batch_size, val_batch_size, num_workers,
                               "%s_run_%d_%d" % (outfilename, eidx, aidx), test_name, datadir, device)
            print(eidx, len(evaluationsdirs), evaluationdir, aidx, approach, results)
            with open(outfilename + '.csv', "a") as outfile:
                outfile.write("%s, %d, %s, %d, %s, %s\n" % (evaluationdir, eidx, approach, aidx, results,
                                                            train_kpis))

