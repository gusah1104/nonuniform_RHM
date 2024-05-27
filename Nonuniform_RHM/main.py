import torch
from cnn import CNN
from fcn import FCN
from lcn import LocallyHierarchicalNet
from patchwise_perceptron import patchwise_perceptron
import copy
from optimizer import loss_func, regularize, opt_algo, measure_accuracy
import numpy as np
import torch.nn.functional as F
import argparse
from Dataset import sample_RHM, get_whole_rule
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(trainloader, net0,lr=0.1,width=256,num_epochs=200):
    criterion = loss_func
    epochs=num_epochs
    net = copy.deepcopy(net0)
    optimizer, scheduler = opt_algo(net,lr=lr,width=width,epochs=num_epochs)
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())
            train_loss += loss.detach().item()
            regularize(loss, net, 5e-4)
            loss.backward()
            optimizer.step()
            correct, total = measure_accuracy(outputs, targets, correct, total)
        if train_loss/(batch_idx+1) < 0.02:
            break 
        if epoch %5  == 0:
            print(
                f"[Train epoch {epoch+1} / {epochs}]"
                f"[tr.Loss: {train_loss *1 / (batch_idx + 1):.03f}]"
                f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]",
                flush=True
            )
        if epoch==70 and train_loss/(batch_idx+1)>0.02:
            break
    return net

def test(testloader, net):
    criterion = loss_func
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha=1.0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())
            test_loss += loss.item()
            correct, total = measure_accuracy( outputs, targets, correct, total)

        print(
            f"[TEST][te.Loss: {test_loss * alpha / (batch_idx + 1):.03f}]"  #0 was batch_idx
            f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]\n",
            flush=True                
            )
    return 100.0 * correct / total


#Count the total overlap of training set and test set
def count_disjoint_paths(paths1, paths2):
    paths2_set = set(map(tuple, paths2))
    num_disjoint_paths = sum(tuple(path1) not in paths2_set for path1 in paths1)
    return num_disjoint_paths


#divice test set into how many paths were seen in training set
def divide_test_index(train_path, test_path, s, L):
    train_length = len(train_path)
    train_path = np.array(train_path)

    y = np.zeros((train_length, s**(L-1), 1+L))
    y[:, :, 0] = train_path[:, 0][:, np.newaxis]
    y[:, :, 1] = train_path[:, 1][:, np.newaxis]
    for l in range(2, L+1):
        choice_so_far = int(1+np.sum([s**j for j in range(l-1)]))
        repeats = s**(L-l)
        y[:, :, l] = np.repeat(train_path[:, choice_so_far:choice_so_far+s**(l-1)], repeats, axis=1)

    train_positions = [set(tuple(y[:, i, :][j]) for j in range(train_length)) for i in range(s**(L-1))]

    test_length = len(test_path)
    test_path = np.array(test_path)

    z = np.zeros((test_length, s**(L-1), 1+L))
    z[:, :, 0] = test_path[:, 0][:, np.newaxis]
    z[:, :, 1] = test_path[:, 1][:, np.newaxis]
    for l in range(2, L+1):
        choice_so_far = int(1+np.sum([s**j for j in range(l-1)]))
        repeats = s**(L-l)
        z[:, :, l] = np.repeat(test_path[:, choice_so_far:choice_so_far+s**(l-1)], repeats, axis=1)

    new_position_sets = [[j for j in range(test_length) if tuple(z[:, i, :][j]) not in train_positions[i]] for i in range(s**(L-1))]

    # Initialize sets to store indices for each category(how many paths are seen in training set)
    indices_sets = [[] for _ in range(s**(L-1) + 1)]

    # Iterate through each index and count how many sets it belongs to
    for i in range(test_length):
        count = sum(i in new_set for new_set in new_position_sets)
        indices_sets[count].append(i)
    return indices_sets
            
        

# parser = argparse.ArgumentParser()
# parser.add_argument('--v', type=int)
# parser.add_argument('--n_c', type=int)
# parser.add_argument('--m',  type=int)
# parser.add_argument('--s',  type=int)
# parser.add_argument('--L', type=int)
# parser.add_argument('--lr', default=0.3)
# args = parser.parse_args()



v=10;num_classes=10; m=10;num_layers=2;s=2
# v=args.v; num_classes=args.n_c; m=args.m; s=args.s; num_layers=args.L
print(f"v={v}, m={m}, num_layers={num_layers}, s={s}")
ptr_list=np.logspace(1.5,4,3)
alpha_list=[1]
trial_number=1

result=torch.zeros(trial_number,len(alpha_list),len(ptr_list),s**(num_layers-1)+2)
size=torch.zeros(trial_number,len(alpha_list),len(ptr_list),s**(num_layers-1)+1)
summed_result=torch.zeros(len(alpha_list),len(ptr_list),s**(num_layers-1)+2)
summed_size=torch.zeros(len(alpha_list),len(ptr_list),s**(num_layers-1)+1)
for trial in range(trial_number):
    for alpha_count,alpha in enumerate(alpha_list):
        print(f"alpha={alpha}\n\n\n\n") 
        rule_layers=get_whole_rule(num_classes,v,m,num_layers,s)
        num_sample = int(10**4+10**4)
        total_dataset,total_path = sample_RHM(num_sample,alpha, rule_layers, m,v,num_classes,num_layers,s)
        input_dim = total_dataset[0][0].shape[-1]
        ch = total_dataset[0][0].shape[-2]
        print(f"input_dim={input_dim}, ch={ch}")
        output_dim=num_classes
        h=100;lr=0.03;num_epochs=200
        print(f"h={h}, lr={lr}")
        for ptr_count,ptr in enumerate(ptr_list):  
            total_idx=np.random.choice(len(total_dataset),int(ptr)+10**4,replace=False)
            train_idx=total_idx[:int(ptr)]
            test_idx=total_idx[int(ptr):]
            train_path=[total_path[i] for i in train_idx]
            test_path=[total_path[i] for i in test_idx]
            trainset=torch.utils.data.Subset(total_dataset,train_idx)
            testset=torch.utils.data.Subset(total_dataset,test_idx)  
            
            # net0=patchwise_perceptron(input_channels=ch**(s),h=num_classes,out_dim=output_dim,num_layers=num_layers,bias=False,s=s).to(device)
            # net0= FCN(num_layers=num_layers,input_channels=s**num_layers*ch,h=h,out_dim=output_dim,bias=False).to(device)
            # net0=CNN(num_layers=num_layers,input_channels=ch,h=h,out_dim=output_dim,bias=False,patch_size=s).to(device)
            net0=LocallyHierarchicalNet(input_channels=ch,h=num_classes,out_dim=output_dim,num_layers=num_layers,bias=False,s=s).to(device)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=0)
            testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
            
            net=train(trainloader, net0,lr=lr,width=h,num_epochs=num_epochs)
            # net.visualize_weights()
            result[trial][alpha_count][ptr_count][s**(num_layers-1)+1]=test(testloader, net)
            
            indicies_sets=divide_test_index(train_path,test_path,s,L=num_layers)
            print(f"overlap of train and test set={(len(testset)-count_disjoint_paths(test_path,train_path))/len(testset)}")
            for i in range(s**(num_layers-1)+1):
                print(f"path percentage={1-i/s**(num_layers-1):.2f}, len={len(indicies_sets[i])}")
                if len(indicies_sets[i])==0:
                    continue
                ith_testset=torch.utils.data.Subset(testset,indicies_sets[i])
                testloader = torch.utils.data.DataLoader(ith_testset, batch_size=256, shuffle=False, num_workers=0)
                result[trial][alpha_count][ptr_count][i]=test(testloader, net)
                size[trial][alpha_count][ptr_count][i]=len(indicies_sets[i])  
            
    summed_result=torch.sum(result,axis=0)
    summed_size=torch.sum(size,axis=0)
print(summed_result/trial_number)
print(f"ptr_list={torch.tensor(ptr_list)}")
print(summed_size/trial_number)













