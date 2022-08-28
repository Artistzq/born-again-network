import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
import torchvision
from art.estimators.classification import PyTorchClassifier
from art.metrics.metrics import clever_u
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

datasets = {
    "CIFAR10": torchvision.datasets.CIFAR10, 
    "CIFAR100": torchvision.datasets.CIFAR100
}
classes = {
    "CIFAR10":('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    "CIFAR100":('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
}


def _acc(model, data_loader):
    model.eval()
    model = model.to(device)
    error = 0
    total = 0
    for X, y in data_loader:
        with torch.no_grad():
            X = X.to(device)
            pred = model(X)
            y = y.to(device)
        diff = y - torch.argmax(pred, axis=-1)
        error += diff.count_nonzero().item()
        total += y.numel()
    return 1 - (error / total)


def _ece(model, data_loader, num_bins=10):
    model.eval()
    model = model.to(device)
    true_labels = []
    pred_labels = []
    confidences = []
    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        true_labels.append(y)
        with torch.no_grad():
            pred = model(X)
        pred_labels.append(torch.argmax(pred, axis=-1))
        confidences.append(pred)
    true_labels = torch.cat(true_labels).cpu().detach().numpy()
    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    confidences = torch.cat(confidences).cpu()
    # resnet输出没有softmax层，因此需要经过softmax处理
    confidences = torch.nn.functional.softmax(confidences, dim=-1).detach().numpy()
    confidences = np.max(confidences, axis=-1)
    return __compute_calibration(true_labels, pred_labels, confidences, num_bins)


def __compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)
    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)
    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    return ece


def _nfr(model1, model2, data_loader):
    model1.eval()
    model2.eval()
    model1 = model1.to(device)
    model2 = model2.to(device)
    vals = []
    total = 0
    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred1 = model1(X)
            pred2 = model2(X)
        pred_h1 = torch.argmax(pred1, axis=-1)
        pred_h2 = torch.argmax(pred2, axis=-1)
        vals.append(__nfr_compute(pred_h1, pred_h2, y).item())
        total += y.numel()
    return 100. * sum(vals) / total


def __nfr_compute(pred_h1: torch.Tensor, pred_h2: torch.Tensor, truth: torch.Tensor):
    correct_h1 = pred_h1.eq(truth)
    wrong_h2 = ~pred_h2.eq(truth)
    negative_flip = wrong_h2.masked_select(correct_h1)
    val = negative_flip.count_nonzero()
    return val                          
    # val = negative_flip.count_nonzero() / truth.size(0)
    # return 100. * val.item()


def _robustness(model: torch.nn.Module, dataset: torch.utils.data.Dataset, num_classes):
    model.eval()
    single_sample = dataset[0][0]
    input_shape = single_sample.size
    # num_classes = num_classes
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1.e-6, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=input_shape,
        nb_classes=num_classes,
        optimizer=optimizer
    )
    scores = []
    for i in range(len(dataset)):
        score = clever_u(classifier, dataset[i][0].numpy(), 8, 8, 5, norm=2, pool_factor=3)
        if i % 500 == 0:
            print(i, 'images rb tested:', score)
        scores.append(score)
    return sum(scores) / len(scores)



class Metric():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        print("==> Evaluating on {}".format(device))
        self.testset, self.testloader = self.get_set_loader()
        
    def get_set_loader(self):
        dataset = self.dataset
        # Data
        assert dataset is not None, "specify the dataset"
        print('==> Preparing Testing data..')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets[dataset](
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        return testset, testloader

    def acc(self, model):
        return _acc(model, self.testloader)

    def ece(self, model, num_bins=10):
        return _ece(model, self.testloader, num_bins)

    def nfr(self, model1, model2):
        return _nfr(model1, model2, self.testloader)

    def clever(self, model):
        return _robustness(model, self.testset, len(classes[self.dataset]))
