import argparse
import torch
from metrics import Metric
from models import *
import os

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

model_names = os.listdir("./results/{}".format(evaluate_path))
model_names = [item for item in model_names if "ep" in item]
model_names.sort()
model_names = model_names[-1: ] + model_names[:-1]

logs = []
for name in model_names:
    path = os.path.join("./results/{}".format(evaluate_path), name)
    net.load_state_dict(torch.load(path)["net"])
    net.to(device)
    acc = metric.acc(net)
    ece = metric.ece(net)
    if "ban" in name:
        nfr = metric.nfr(net, origin)
        log = "{:<20s} acc:{:.4f} ece:{:.4f} nfr:{:.4f}".format(name, acc, ece, nfr)
    else:
        origin.load_state_dict(torch.load(path)["net"])
        origin.to(device)
        log = "{:<20s} acc:{:.4f} ece:{:.4f}".format(name, acc, ece)
    print(log)
    logs.append(log+"\n")
    
with open("./results/{}/results.txt".format(evaluate_path), "w") as f:
    f.writelines(logs)
