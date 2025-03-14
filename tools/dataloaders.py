import torch
import numpy as np


def get_data_for_mia_base(model, dataloaders, dataset_sizes, device):
    preds = []
    true_label = []
    model.to(device)
    model.eval()
    for phase in ['train', 'test']:
        for data, _ in dataloaders[phase]:
            with torch.no_grad():
                inputs = data.to(device)
                output = torch.softmax(model(inputs), dim=1)
                preds.append(output)
        if phase == "train":
            true_label.append(torch.ones(dataset_sizes[phase], dtype=torch.long, device=device))
        else:
            true_label.append(torch.zeros(dataset_sizes[phase], dtype=torch.long, device=device))
    preds = torch.cat(preds, dim=0)
    true_label = torch.cat(true_label, dim=0)
    return preds, true_label


def get_data_for_attack_eval(model, dataloader, device):
    label = []
    pred = []
    soft_pred = []
    model.to(device)
    model.eval()
    for batch_idx, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            inputs, labels = data.to(device), target.to(device)
            output = torch.softmax(model(inputs), dim=1)
            preds = torch.argmax(output, dim=1)
            soft_preds = output[:, 1]
        for cla in labels.cpu().detach().numpy():
            label.append(cla)
        for out in preds.cpu().detach().numpy():
            pred.append(out)
        for s_out in soft_preds.cpu().detach().numpy():
            soft_pred.append(s_out)    
    return (np.array(label), np.array(pred), np.array(soft_pred))


def train_test_acc(model, dataloaders, dataset_sizes, device):
    model.to(device)
    model.eval()
    acc = []
    for phase in ['train', 'test']:
        running_corrects = 0
        for batch_idx, (data, target) in enumerate( dataloaders[phase]):
            with torch.no_grad():
                inputs, labels = data.to(device), target.to(device)
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        accuracy = running_corrects.double() / dataset_sizes[phase]
        acc.append(accuracy.cpu().detach().numpy())
    return np.array(acc)

def get_preds(model, dataloaders, device):
    train_preds = []
    test_preds = []
    model.to(device)
    model.eval()
    for phase in ['train', 'test']:
        for inputs, _ in dataloaders[phase]:
            with torch.no_grad():
                inputs = inputs.to(device)
                output = torch.softmax(model(inputs), dim=1)
            if phase == "train":
                train_preds.append(output)
            else:
                test_preds.append(output)
        if phase == "train":        
            train_preds = torch.cat(train_preds, dim=0)
        else:
            test_preds = torch.cat(test_preds, dim=0)
    return train_preds, test_preds

def kl_divergence(P, Q, eps=1e-10):
    P = P + eps
    Q = Q + eps
    kl = torch.sum(P * (torch.log(P) - torch.log(Q)), dim=1)
    kl = torch.clamp(kl, min=0)
    return kl


def js_divergence(P, Q, eps=1e-10):
    M = 0.5 * (P + Q)
    kl_PM = kl_divergence(P, M, eps)
    kl_QM = kl_divergence(Q, M, eps)
    js = 0.5 * (kl_PM + kl_QM)
    return js


def cosine(P, Q, eps=1e-10):
    dot = torch.sum(P * Q, dim=1)
    norm_p = torch.norm(P, dim=1)
    norm_q = torch.norm(Q, dim=1)
    return dot / (norm_p * norm_q + eps)
    

def hellinger_distance(P, Q, eps=1e-10):
    sqrtP = torch.sqrt(P + eps)
    sqrtQ = torch.sqrt(Q + eps)
    return torch.sqrt(torch.sum((sqrtP - sqrtQ) ** 2, dim=1)) / (2 ** 0.5)
    
    
def bhattacharyya(P, Q, eps=1e-10):
    bc = torch.sum(torch.sqrt(P * Q + eps), dim=1)
    return -torch.log(bc + eps)
    
    
def pearson(P, Q, eps=1e-10):
    P_mean = P.mean(dim=1, keepdim=True)
    Q_mean = Q.mean(dim=1, keepdim=True)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    cov = torch.sum(P_centered * Q_centered, dim=1)
    P_std = torch.sqrt(torch.sum(P_centered ** 2, dim=1) + eps)
    Q_std = torch.sqrt(torch.sum(Q_centered ** 2, dim=1) + eps)
    return cov / (P_std * Q_std)
    
    
def get_similarity(preds, q_preds, metric):
    if metric == "KL_divergence":
        similarity = kl_divergence(preds, q_preds, eps=1e-10)
        
    elif metric == "JS_divergence":
        similarity = js_divergence(preds, q_preds, eps=1e-10)
        
    elif metric == "Cross_Entropy":
        similarity = -torch.sum(preds * torch.log(q_preds + 1e-10), dim=1)
    
    elif metric == "Cosine":
        similarity = cosine(preds, q_preds, eps=1e-10)
        
    elif metric == "Euclidean":
        similarity = torch.sqrt(torch.sum((preds - q_preds) ** 2, dim=1))
    
    elif metric == "Hellinger":
        similarity = hellinger_distance(preds, q_preds, eps=1e-10)
    
    elif metric == "Bhattacharyya":
        similarity = bhattacharyya(preds, q_preds, eps=1e-10)
    
    elif metric == "Pearson":
        similarity = pearson(preds, q_preds, eps=1e-10)
    similarity_np = similarity.cpu().detach().numpy()

    return  similarity_np
