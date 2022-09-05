# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class BANLoss:
    def __init__(self, criterion, teacher=None):
        self.teacher = teacher
        self.criterion = criterion
    
    def __call__(self, outputs, targets, inputs):
        if self.teacher is not None:
            teacher_outputs = self.teacher(inputs).detach()
            loss = self.kd_loss(outputs, targets, teacher_outputs)
        else:
            loss = self.criterion(outputs, targets)
        return loss

    def kd_loss(self, outputs, labels, teacher_outputs, alpha=0.8, T=4):
        KD_loss = (1. - alpha) * F.cross_entropy(outputs, labels) + \
            nn.KLDivLoss()(
                F.log_softmax(outputs/T, dim=1), 
                F.softmax(teacher_outputs/T, dim=1)
            ) * alpha * T * T
        return KD_loss