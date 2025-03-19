import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
import shutil
import datetime
import torch.optim as optim
import copy
from torch.optim import lr_scheduler
from quantization import *
from tools import *
from utils import MyConfig
import timm
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping, QConfig
import faulthandler
faulthandler.enable()


def get_args_parser():
    parser = argparse.ArgumentParser(description="RepQ-ViT", add_help=False)
    parser.add_argument("--model", default="resnet50", type=str,
                        choices=['resnet34', 'resnet50', 'vgg16', 'vgg19', 'wide_resnet50_2', 'mobilenetv2_100'])
    parser.add_argument('--dataset', default="CIFAR100", type=str,
                        choices=['CIFAR10', 'CIFAR100', 'CINIC10', 'GTSRB', 'ImageNet100'])
    parser.add_argument('--attack_method_attention', default="rollout", type=str,
                        choices=['rollout', 'last_attention', 'EncoderMI'])
    parser.add_argument('--metric', default="KL_divergence", type=str, 
                        choices=['KL_divergence', 'JS_divergence', 'Cross_Entropy', 'Cosine', 
                                 'Euclidean', 'Hellinger', 'Bhattacharyya', 'Pearson'])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=10, type=int)
    return parser


def attack_eval(path, attack_model, test_loader_attack, device, method):
    labels, preds, soft_preds = get_data_for_attack_eval(attack_model, test_loader_attack, device)
    
    total = labels.size
    correct = np.sum(labels == preds)
    accuracy = correct / total if total > 0 else 0
    
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 1
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    np.save(path + f"/res_accuracy_{method}.npy", accuracy)
    np.save(path + f"/res_precision_{method}.npy", precision)
    np.save(path + f"/res_recall_{method}.npy", recall)
    np.save(path + f"/res_label_attack_{method}.npy", labels)
    np.save(path + f"/res_soft_pred_attack_{method}.npy", soft_preds)


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def quantization_similarity(args, path, w_bits, a_bits, target_model, distill_target_model, shadow_model, distill_shadow_model, 
                            dataloaders_target, dataloaders_shadow, distill_calib_data):
    
    print(f'Performing quantization W{w_bits}A{a_bits} ................................................................................')
    wq_params = {'n_bits': w_bits, 'channel_wise': True}
    aq_params = {'n_bits': a_bits, 'channel_wise': False}

    dt_model = copy.deepcopy(distill_target_model)
    q_distill_target_model = quant_model(dt_model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_distill_target_model.to(device)
    q_distill_target_model.eval()
    set_quant_state(q_distill_target_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_distill_target_model(distill_calib_data)

    ds_model = copy.deepcopy(distill_shadow_model)
    q_distill_shadow_model = quant_model(ds_model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_distill_shadow_model.to(device)
    q_distill_shadow_model.eval()
    set_quant_state(q_distill_shadow_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_distill_shadow_model(distill_calib_data)

    target_train_preds, target_test_preds = get_preds(target_model, dataloaders_target, args.device)
    shadow_train_preds, shadow_test_preds = get_preds(shadow_model, dataloaders_shadow, args.device)
    
    print(f'Computing similarity W{w_bits}A{a_bits} ..........................................................................')

    distill_target_train_preds, distill_target_test_preds = get_preds(q_distill_target_model, dataloaders_target, args.device)
    distill_target_train_sim = get_similarity(target_train_preds, distill_target_train_preds, args.metric)
    distill_target_test_sim = get_similarity(target_test_preds, distill_target_test_preds, args.metric)
    np.save(f"{path}/res_w{w_bits}a{a_bits}_distill_target_train_sim.npy", distill_target_train_sim)
    np.save(f"{path}/res_w{w_bits}a{a_bits}_distill_target_test_sim.npy", distill_target_test_sim)

    distill_shadow_train_preds, distill_shadow_test_preds = get_preds(q_distill_shadow_model, dataloaders_shadow, args.device)
    distill_shadow_train_sim = get_similarity(shadow_train_preds, distill_shadow_train_preds, args.metric)
    distill_shadow_test_sim = get_similarity(shadow_test_preds, distill_shadow_test_preds, args.metric)
    np.save(f"{path}/res_w{w_bits}a{a_bits}_distill_shadow_train_sim.npy", distill_shadow_train_sim)
    np.save(f"{path}/res_w{w_bits}a{a_bits}_distill_shadow_test_sim.npy", distill_shadow_test_sim)
    
    return (torch.from_numpy(distill_target_train_sim).to(args.device), 
            torch.from_numpy(distill_target_test_sim).to(args.device), 
            torch.from_numpy(distill_shadow_train_sim).to(args.device), 
            torch.from_numpy(distill_shadow_test_sim).to(args.device))


def main():
    print(args)
    seed(args.seed)
    
    now = str(datetime.datetime.now())[:19]
    now = now.replace(":","_")
    now = now.replace("-","_")
    now = now.replace(" ","_")
    
    save_folder = f'./saved_models/{args.model}_{args.dataset}_{str(now)}/'
    os.makedirs(save_folder, exist_ok=True)
    
    config_dict = {'CIFAR10': "config/CIFAR10/",
                'CIFAR100': "config/CIFAR100/",
                'CINIC10': "config/CINIC10/",
                'GTSRB': "config/GTSRB/",
                'ImageNet100': "config/ImageNet100/"}
    
    config = MyConfig.MyConfig(path=config_dict[args.dataset])
    
    src_dir = config.path.config_path
    path = config.path.result_path + args.dataset + "/" + args.model + "_" + str(now)
    os.makedirs(path)
    dst_dir = path+ "/config.yaml"
    shutil.copy(src_dir,dst_dir)
    
    device = torch.device(args.device)
    
    print(f'Performing experiments for {args.model} on {args.dataset} ....................................................................................')
    
    print('Building dataloader ...........................................................................................................................')
    dataloaders_target, dataset_sizes_target, dataloaders_shadow, dataset_sizes_shadow, dataloader_distill = build_dataset(args, config)
    
    criterion = nn.CrossEntropyLoss()
    
    print('Building target model .........................................................................................................................')
    target_model = timm.create_model(args.model, pretrained=False, num_classes=config.general.num_classes)
    # target_model.load_state_dict(torch.load("./saved_models/wide_resnet50_2_CINIC10_2025_03_11_10_46_49/target_model.pth", weights_only=True))
    # target_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg19_bn", pretrained=True, num_classes=config.general.num_classes)
    # target_model.load_state_dict(torch.load("./saved_models/resnet50_ImageNet100_2025_03_10_16_54_39/target_model.pth", weights_only=True))
    optimizer = optim.SGD(target_model.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum, weight_decay=config.learning.weight_decay)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_factor, gamma=config.learning.decrease_lr_every)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.learning.epochs)
    target_model = train_model(target_model, criterion, optimizer, exp_lr_scheduler, dataloaders_target, dataset_sizes_target, num_epochs=config.learning.epochs)
    save_path = os.path.join(save_folder, 'target_model.pth')
    torch.save(target_model.state_dict(), save_path)
    target_train_test_accuracy = train_test_acc(target_model, dataloaders_target, dataset_sizes_target, device)
    np.save(path + "/res_target_train_test_accuracy"+".npy", target_train_test_accuracy)
    
    distill_target_model = distill(args, config, target_model, dataloader_distill)
    # distill_target_model = timm.create_model(args.model, pretrained=True, num_classes=config.general.num_classes)
    # distill_target_model.load_state_dict(torch.load("./saved_models/vgg16_CINIC10_2025_03_09_18_02_43/distill_target_model.pth", weights_only=True))
    save_path = os.path.join(save_folder, 'distill_target_model.pth')
    torch.save(distill_target_model.state_dict(), save_path)
    
    print('Building shadow model .........................................................................................................................')
    shadow_model = timm.create_model(args.model, pretrained=False, num_classes=config.general.num_classes)
    # shadow_model.load_state_dict(torch.load("./saved_models/wide_resnet50_2_CINIC10_2025_03_11_10_46_49/shadow_model.pth", weights_only=True))
    # shadow_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg19_bn", pretrained=True, num_classes=config.general.num_classes)
    # shadow_model.load_state_dict(torch.load("./saved_models/resnet50_ImageNet100_2025_03_10_16_54_39/shadow_model.pth", weights_only=True))
    optimizer = optim.SGD(shadow_model.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum, weight_decay=config.learning.weight_decay)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_factor, gamma=config.learning.decrease_lr_every)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.learning.epochs)
    shadow_model = train_model(shadow_model, criterion, optimizer, exp_lr_scheduler, dataloaders_shadow, dataset_sizes_shadow, num_epochs=config.learning.epochs)
    save_path = os.path.join(save_folder, 'shadow_model.pth')
    torch.save(shadow_model.state_dict(), save_path)
    shadow_train_test_accuracy = train_test_acc(shadow_model, dataloaders_shadow, dataset_sizes_shadow, device)
    np.save(path + "/res_shadow_train_test_accuracy"+".npy", shadow_train_test_accuracy)
    
    distill_shadow_model = distill(args, config, shadow_model, dataloader_distill)
    # distill_shadow_model = timm.create_model(args.model, pretrained=True, num_classes=config.general.num_classes)
    # distill_shadow_model.load_state_dict(torch.load("./saved_models/vgg16_CINIC10_2025_03_09_18_02_43/distill_shadow_model.pth", weights_only=True))
    save_path = os.path.join(save_folder, 'distill_shadow_model.pth')
    torch.save(distill_shadow_model.state_dict(), save_path)
    
    for data, _ in dataloader_distill:
        distill_calib_data = data.to(device)
        break
    distill_calib_data.to(device)
    
    quantization_configs = [(16, 16), (8, 16), (16, 8), (8, 8), (6, 8), 
                            (8, 6), (6, 6), (4, 6), (6, 4), (4, 4)]

    target_train_sim_list = []
    target_test_sim_list = []
    shadow_train_sim_list = []
    shadow_test_sim_list = []
    
    for w_bits, a_bits in quantization_configs:
        target_train_sim, target_test_sim, shadow_train_sim, shadow_test_sim = 
        quantize_and_evaluate(args, path, w_bits, a_bits, target_model, distill_target_model, shadow_model, distill_shadow_model, 
                            dataloaders_target, dataloaders_shadow, distill_calib_data)
        
        target_train_sim_list.append(target_train_sim)
        target_test_sim_list.append(target_test_sim)
        shadow_train_sim_list.append(shadow_train_sim)
        shadow_test_sim_list.append(shadow_test_sim)
        
    seq_target_train_sim = torch.stack(target_train_sim_list, dim=-1)
    seq_target_test_sim = torch.stack(target_test_sim_list, dim=-1)
    seq_shadow_train_sim = torch.stack(shadow_train_sim_list, dim=-1)
    seq_shadow_test_sim = torch.stack(shadow_test_sim_list, dim=-1)
    
    train_data = torch.cat([seq_shadow_train_sim, seq_shadow_test_sim])
    train_label = torch.cat([torch.ones(seq_shadow_train_sim.shape[0], dtype=torch.long), torch.zeros(seq_shadow_test_sim.shape[0], dtype=torch.long)]).to(args.device)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.attack_learning.train_batch_size, shuffle=True)
    
    test_data = torch.cat([seq_target_train_sim, seq_target_test_sim])
    test_label = torch.cat([torch.ones(seq_target_train_sim.shape[0], dtype=torch.long), torch.zeros(seq_target_test_sim.shape[0], dtype=torch.long)]).to(args.device)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.attack_learning.test_batch_size, shuffle=False)
    
    dataloaders_attack = {"train": train_loader, "test": test_loader}
    dataset_sizes_attack = {"train": len(train_dataset), "test": len(test_dataset)}
    
    print('Performing membership inference attack (QS_MIA) .................................................................................................')
    attack_model = AttackModel(len(quantization_configs)).to(args.device)
    optimizer = optim.SGD(attack_model.parameters(), lr=config.attack_learning.learning_rate, momentum=config.attack_learning.momentum, weight_decay=config.attack_learning.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.attack_learning.decrease_lr_factor, gamma=config.attack_learning.decrease_lr_every)
    criterion = nn.CrossEntropyLoss()
    attack_model = train_model(attack_model, criterion, optimizer, exp_lr_scheduler, dataloaders_attack, dataset_sizes_attack, num_epochs=config.attack_learning.epochs)
    attack_eval(path, attack_model, test_loader, args.device, method='QS_MIA')
    
    print('Performing membership inference attack (Base) .................................................................................................')
    attack_model, test_loader_attack = mia_base(args, config, target_model, dataloaders_target, dataset_sizes_target, shadow_model, dataloaders_shadow, dataset_sizes_shadow)
    attack_eval(path, attack_model, test_loader_attack, args.device, method="MIA_base")
    