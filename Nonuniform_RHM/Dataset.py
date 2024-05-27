import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F



def onehot(y, num_classes):
    y = F.one_hot(y.long(), num_classes).float()
    y = y.permute(1, 0)
    return y

def digits_in_base(num, base):
    num = np.asarray(num)
    digits_list = []
    while np.any(num > 0):
        digits_list.insert(0, num % base)
        num //= base
    digits_array = np.vstack(digits_list)
    return digits_array

def get_rule_class(n_c,v,m,s=2):
    rule = np.arange(v ** s)
    np.random.shuffle(rule)
    rule = rule[:m * n_c]
    rule = rule.reshape(n_c, m)
    return rule

def get_rule(v,m,s=2):
    rule = np.arange(v ** s)
    np.random.shuffle(rule)
    rule = rule[:m * v]
    rule = rule.reshape(v, m)
    return rule

def get_whole_rule(n_c,v,m,L,s):
    rule_layers = []
    rule_layers.append(get_rule_class(n_c,v, m,s))
    for _ in range(L-1):
        rule_layers.append(get_rule(v, m,s))
    return rule_layers



#It generates (input,output) as well as s^(L-1) paths
def sample_RHM(length,temperature, rule_layers, m,v,n_c,L,s):
    dataset = []
    paths=[]
    uniform__class_p= np.ones(n_c) / n_c
    uniform_synonym_p= np.ones(m) / m
    zipf_synonym_p=np.ones(m)
    for i in range(m):
        zipf_synonym_p[i]=(i+1)**(-1-temperature)
    zipf_synonym_p=zipf_synonym_p/np.sum(zipf_synonym_p)
    choice_of_synonyms=1+np.sum([s**i for i in range(L)])
    a = np.zeros((choice_of_synonyms, length), dtype=int)
    
    #Uniform class choice
    idx = np.arange(length)
    np.random.shuffle(idx)
    counts = np.floor(length * np.array(uniform__class_p))
    counts[-1] = length - np.sum(counts[:-1])
    current_idx = 0
    for j in range(n_c):
        next_idx = int(current_idx + counts[j])
        a[0, idx[current_idx:next_idx]] = j
        current_idx = next_idx
        
    #Zipf synonym choice   
    for i in range(1,choice_of_synonyms):
        idx = np.arange(length)
        np.random.shuffle(idx)
        counts = np.floor(length * np.array(zipf_synonym_p))
        counts[-1] = length - np.sum(counts[:-1])
        current_idx = 0
        for j in range(m):
            next_idx = int(current_idx + counts[j])
            a[i, idx[current_idx:next_idx]] = j
            current_idx = next_idx
    #Uniform synonym choice        
    for i in []:
        idx = np.arange(length)
        np.random.shuffle(idx)
        counts = np.floor(length * np.array(uniform_synonym_p))
        counts[-1] = length - np.sum(counts[:-1])
        current_idx = 0
        for j in range(m):
            next_idx = int(current_idx + counts[j])
            a[i, idx[current_idx:next_idx]] = j
            current_idx = next_idx
    a = a.transpose(1, 0)

    #Generate inputs using choices of synonyms from rules
    upper_layer=digits_in_base(rule_layers[0][a[:, 0], a[:, 1]],v)
    for l in range(1,L):
        choices_so_far=int(1+np.sum([s**i for i in range(l)]))
        bottom_layer=np.concatenate([digits_in_base(rule_layers[l][upper_layer[i], a[:, choices_so_far+i]],v) for i in range(s**l)],axis=0)
        upper_layer=bottom_layer
    bottom_layer=torch.tensor(bottom_layer)
    for i in range(len(a)):
        dataset.append((onehot(bottom_layer[:, i], v), a[i, 0]))
        paths.append(a[i,:])
    idx = np.arange(length)
    np.random.shuffle(idx)
    dataset = [dataset[i] for i in idx]
    paths = [paths[i] for i in idx]
    return dataset, paths    

