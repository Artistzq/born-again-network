import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from metrics import Metric
from models import *
from utils import get_model

from torchvision import transforms
import torchvision


parser = argparse.ArgumentParser(description='Bans Evaluating')
parser.add_argument('--dataset', default="CIFAR10", type=str)
parser.add_argument('--path', type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = args.dataset
evaluate_path = args.path

# Data
print('==> Preparing Testing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
datasets = {
    "CIFAR10": torchvision.datasets.CIFAR10, 
    "CIFAR100": torchvision.datasets.CIFAR100
}
testset = datasets[dataset](
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

metric = Metric(testset, testloader)
model_type = evaluate_path.split("_")[0]

net = get_model(model_type)
origin = get_model(model_type)

model_names = os.listdir("./results/{}".format(evaluate_path))
model_names = [item for item in model_names if "ep" in item]
model_names.sort()
model_names = model_names[-1: ] + sorted(model_names[: -1], key=lambda x: int(x.split("_")[0][3:]))

logs = []
records = []
for name in model_names:
    path = os.path.join("./results/{}".format(evaluate_path), name)
    net.load_state_dict(torch.load(path)["net"], strict=False)
    net.to(device)
    acc = metric.acc(net)
    class_acc = metric.class_acc(net, 10)
    s = " ".join(["{:.2f}".format(item) for item in class_acc])
    ece_10 = metric.ece(net)
    ece_15 = metric.ece(net, num_bins=15)
    
    record = {
        "name": name,
        "acc": acc,
        "ece_10": ece_10,
        "ece_15": ece_15,
        "class_acc": class_acc
    }
    if "ban" in name:
        nfr = metric.nfr(net, origin)
        record["nfr"] = nfr
        log = "{:<20s} acc:{:.4f} ece_10:{:.6f} ece_15:{:.6f} nfr:{:.4f} class_acc:{}".format(name, acc, ece_10, ece_15, nfr, s)
    else:
        origin.load_state_dict(torch.load(path)["net"])
        origin.to(device)
        log = "{:<20s} acc:{:.4f} ece_10:{:.6f} ece_15:{:.6f} class_acc:{}".format(name, acc, ece_10, ece_15, s)
    print(log)
    records.append(record)

json.dump(records, open("./results/{}/results.json".format(evaluate_path), "w"), indent=4)
json.dump(records, open("./results/{}_results.json".format(evaluate_path), "w"), indent=4)
# with open("./results/{}/results.txt".format(evaluate_path), "w") as f:
#     f.writelines(logs)
