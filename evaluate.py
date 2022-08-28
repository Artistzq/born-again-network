import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from metrics import Metric
from models import *


parser = argparse.ArgumentParser(description='Bans Evaluating')
parser.add_argument('--dataset', default="CIFAR10", type=str)
parser.add_argument('--path', type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
evaluate_path = args.path
metric = Metric(args.dataset)

if evaluate_path == "resnet18":
    net = ResNet18()
    origin = ResNet18()
elif evaluate_path == "vgg19":
    net = VGG("VGG19")
    origin = VGG("VGG19")
elif evaluate_path == "vgg11":
    net = VGG("VGG11")
    origin = VGG("VGG11")
elif evaluate_path == "lenet":
    net = LeNet()
    origin = LeNet()

model_names = os.listdir("./results/{}".format(evaluate_path))
model_names = [item for item in model_names if "ep" in item]
model_names.sort()
model_names = model_names[-1: ] + sorted(model_names[: -1], key=lambda x: int(x.split("_")[0][3:]))
print(model_names)

logs = []
for name in model_names:
    path = os.path.join("./results/{}".format(evaluate_path), name)
    net.load_state_dict(torch.load(path)["net"], strict=False)
    net.to(device)
    acc = metric.acc(net)
    ece_10 = metric.ece(net)
    ece_15 = metric.ece(net, num_bins=15)
    if "ban" in name:
        nfr = metric.nfr(net, origin)
        log = "{:<20s} acc:{:.4f} ece_10:{:.6f} ece_15:{:.6f} nfr:{:.4f}".format(name, acc, ece_10, ece_15, nfr)
    else:
        origin.load_state_dict(torch.load(path)["net"])
        origin.to(device)
        log = "{:<20s} acc:{:.4f} ece_10:{:.6f} ece_15:{:.6f}".format(name, acc, ece_10, ece_15)
    print(log)
    logs.append(log+"\n")
    
with open("./results/{}/results.txt".format(evaluate_path), "w") as f:
    f.writelines(logs)
