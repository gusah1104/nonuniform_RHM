from torch import nn
import torch.optim as optim


def loss_func(o, y):
    # device=o.device
    # o=o.float()
    # y = y.to(device).float()
    loss = nn.functional.cross_entropy(o, y.long(), reduction="mean")
    return loss

def regularize(loss, f, l):
    # device=loss.device
    
    for p in f.parameters():
        # p=p.to(device)
        loss += l * p.pow(2).mean()

def measure_accuracy(out, targets, correct, total):
    # print(f"out={out.size()}, targets={targets}")
    # device=out.device
    # correct = correct.to(device)
    # targets = targets.to(device)
    _, predicted = out.max(1)
    correct += predicted.eq(targets).sum().item()
    total += targets.size(0)
    return correct, total


def opt_algo(net,lr=0.1,width=256,epochs=200):
    lr=lr;alpha=1;width=256;momentum=0.9;epochs=epochs   
    optimizer = optim.SGD(net.parameters(), lr *width, momentum) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * 0.8)
    return optimizer, scheduler
