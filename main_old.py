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
    
    np.save(path + f"/res_accuracy_{method}_classifier.npy", accuracy)
    np.save(path + f"/res_precision_{method}_classifier.npy", precision)
    np.save(path + f"/res_recall_{method}_classifier.npy", recall)
    np.save(path + f"/res_label_attack_{method}_classifier.npy", labels)
    np.save(path + f"/res_soft_pred_attack_{method}_classifier.npy", soft_preds)


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    target_model = timm.create_model(args.model, pretrained=True, num_classes=config.general.num_classes)
    target_model.load_state_dict(torch.load("./saved_models/vgg16_CIFAR10_2025_03_17_12_06_33/target_model.pth", weights_only=True))
    # target_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg19_bn", pretrained=True, num_classes=config.general.num_classes)
    # target_model.load_state_dict(torch.load("./saved_models/resnet50_ImageNet100_2025_03_10_16_54_39/target_model.pth", weights_only=True))
    # optimizer = optim.SGD(target_model.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum, weight_decay=config.learning.weight_decay)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_factor, gamma=config.learning.decrease_lr_every)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.learning.epochs)
    # target_model = train_model(target_model, criterion, optimizer, exp_lr_scheduler, dataloaders_target, dataset_sizes_target, num_epochs=config.learning.epochs)
    # save_path = os.path.join(save_folder, 'target_model.pth')
    # torch.save(target_model.state_dict(), save_path)
    target_train_test_accuracy = train_test_acc(target_model, dataloaders_target, dataset_sizes_target, device)
    np.save(path + "/res_target_train_test_accuracy"+".npy", target_train_test_accuracy)
    
    # distill_target_model = distill(args, config, target_model, dataloader_distill)
    distill_target_model = timm.create_model(args.model, pretrained=True, num_classes=config.general.num_classes)
    distill_target_model.load_state_dict(torch.load("./saved_models/vgg16_CIFAR10_2025_03_17_12_06_33/distill_target_model.pth", weights_only=True))
    # save_path = os.path.join(save_folder, 'distill_target_model.pth')
    # torch.save(distill_target_model.state_dict(), save_path)
    
    print('Building shadow model .........................................................................................................................')
    shadow_model = timm.create_model(args.model, pretrained=True, num_classes=config.general.num_classes)
    # shadow_model.load_state_dict(torch.load("./saved_models/wide_resnet50_2_CINIC10_2025_03_11_10_46_49/shadow_model.pth", weights_only=True))
    # shadow_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg19_bn", pretrained=True, num_classes=config.general.num_classes)
    # shadow_model.load_state_dict(torch.load("./saved_models/resnet50_ImageNet100_2025_03_10_16_54_39/shadow_model.pth", weights_only=True))
    optimizer = optim.SGD(shadow_model.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum, weight_decay=config.learning.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_factor, gamma=config.learning.decrease_lr_every)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.learning.epochs)
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
    
    # device_cpu = torch.device('cpu')

    # example_inputs = (next(iter(dataloaders_shadow["train"]))[0])
    
    for data, _ in dataloader_distill:
        distill_calib_data = data.to(device)
        break
    distill_calib_data.to(device)
    
    # for data, _ in dataloaders_shadow["test"]:
    #     calib_data = data.to(device)
    #     break
    # calib_data.to(device)
    
    print('Performing quantization W16A16 ................................................................................................................')
    
    wq_params = {'n_bits': 16, 'channel_wise': True}
    aq_params = {'n_bits': 16, 'channel_wise': False}
    
    # w16a16_tmodel = copy.deepcopy(target_model)
    # w16a16_target_model = quant_model(w16a16_tmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    # w16a16_target_model.to(device)
    # w16a16_target_model.eval()
    
    # set_quant_state(w16a16_target_model, input_quant=True, weight_quant=True)
    # with torch.no_grad():
    #     _ = w16a16_target_model(calib_data)
    # # save_path = os.path.join(save_folder, 'w16a16_target_model.pth')
    # # torch.save(w16a16_target_model, save_path)
    # w16a16_target_train_test_accuracy = train_test_acc(w16a16_target_model, dataloaders_target, dataset_sizes_target, device)
    # np.save(path + "/res_w16a16_target_train_test_accuracy"+".npy", w16a16_target_train_test_accuracy)
    
    
    w16a16_dtmodel = copy.deepcopy(distill_target_model)
    w16a16_distill_target_model = quant_model(w16a16_dtmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    w16a16_distill_target_model.to(device)
    w16a16_distill_target_model.eval()
    
    set_quant_state(w16a16_distill_target_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = w16a16_distill_target_model(distill_calib_data)
        
        
    # w16a16_smodel = copy.deepcopy(shadow_model)
    # w16a16_shadow_model = quant_model(w16a16_smodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    # w16a16_shadow_model.to(device)
    # w16a16_shadow_model.eval()
    
    # set_quant_state(w16a16_shadow_model, input_quant=True, weight_quant=True)
    # with torch.no_grad():
    #     _ = w16a16_shadow_model(calib_data)
    # # save_path = os.path.join(save_folder, 'w16a16_shadow_model.pth')
    # # torch.save(w16a16_shadow_model, save_path)
    # w16a16_shadow_train_test_accuracy = train_test_acc(w16a16_shadow_model, dataloaders_shadow, dataset_sizes_shadow, device)
    # np.save(path + "/res_w16a16_shadow_train_test_accuracy"+".npy", w16a16_shadow_train_test_accuracy)
    
    
    w16a16_dsmodel = copy.deepcopy(distill_shadow_model)
    w16a16_distill_shadow_model = quant_model(w16a16_dsmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    w16a16_distill_shadow_model.to(device)
    w16a16_distill_shadow_model.eval()
    
    set_quant_state(w16a16_distill_shadow_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = w16a16_distill_shadow_model(distill_calib_data)
    
    print('Performing quantization W8A8 ..................................................................................................................')
    
    wq_params = {'n_bits': 8, 'channel_wise': True}
    aq_params = {'n_bits': 8, 'channel_wise': False}
    
    # w8a8_tmodel = copy.deepcopy(target_model)
    # w8a8_target_model = quant_model(w8a8_tmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    # w8a8_target_model.to(device)
    # w8a8_target_model.eval()
    
    # set_quant_state(w8a8_target_model, input_quant=True, weight_quant=True)
    # with torch.no_grad():
    #     _ = w8a8_target_model(calib_data)
    # # save_path = os.path.join(save_folder, 'w8a8_target_model.pth')
    # # torch.save(w8a8_target_model, save_path)
    # w8a8_target_train_test_accuracy = train_test_acc(w8a8_target_model, dataloaders_target, dataset_sizes_target, device)
    # np.save(path + "/res_w8a8_target_train_test_accuracy"+".npy", w8a8_target_train_test_accuracy)
    
    
    w8a8_dtmodel = copy.deepcopy(distill_target_model)
    w8a8_distill_target_model = quant_model(w8a8_dtmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    w8a8_distill_target_model.to(device)
    w8a8_distill_target_model.eval()
    
    set_quant_state(w8a8_distill_target_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = w8a8_distill_target_model(distill_calib_data)
        
        
    # w8a8_smodel = copy.deepcopy(shadow_model)
    # w8a8_shadow_model = quant_model(w8a8_smodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    # w8a8_shadow_model.to(device)
    # w8a8_shadow_model.eval()
    
    # set_quant_state(w8a8_shadow_model, input_quant=True, weight_quant=True)
    # with torch.no_grad():
    #     _ = w8a8_shadow_model(calib_data)
    # # save_path = os.path.join(save_folder, 'w8a8_shadow_model.pth')
    # # torch.save(w8a8_shadow_model, save_path)
    # w8a8_shadow_train_test_accuracy = train_test_acc(w8a8_shadow_model, dataloaders_shadow, dataset_sizes_shadow, device)
    # np.save(path + "/res_w8a8_shadow_train_test_accuracy"+".npy", w8a8_shadow_train_test_accuracy)
    
    
    w8a8_dsmodel = copy.deepcopy(distill_shadow_model)
    w8a8_distill_shadow_model = quant_model(w8a8_dsmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    w8a8_distill_shadow_model.to(device)
    w8a8_distill_shadow_model.eval()
    
    set_quant_state(w8a8_distill_shadow_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = w8a8_distill_shadow_model(distill_calib_data)
    
    print('Performing quantization W6A6 ..................................................................................................................')
    
    wq_params = {'n_bits': 6, 'channel_wise': True}
    aq_params = {'n_bits': 6, 'channel_wise': False}
    
    # w6a6_tmodel = copy.deepcopy(target_model)
    # w6a6_target_model = quant_model(w6a6_tmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    # w6a6_target_model.to(device)
    # w6a6_target_model.eval()
    
    # set_quant_state(w6a6_target_model, input_quant=True, weight_quant=True)
    # with torch.no_grad():
    #     _ = w6a6_target_model(calib_data)
    # # save_path = os.path.join(save_folder, 'w6a6_target_model.pth')
    # # torch.save(w6a6_target_model, save_path)
    # w6a6_target_train_test_accuracy = train_test_acc(w6a6_target_model, dataloaders_target, dataset_sizes_target, device)
    # np.save(path + "/res_w6a6_target_train_test_accuracy"+".npy", w6a6_target_train_test_accuracy)
    
    
    w6a6_dtmodel = copy.deepcopy(distill_target_model)
    w6a6_distill_target_model = quant_model(w6a6_dtmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    w6a6_distill_target_model.to(device)
    w6a6_distill_target_model.eval()
    
    set_quant_state(w6a6_distill_target_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = w6a6_distill_target_model(distill_calib_data)
        
        
    # w6a6_smodel = copy.deepcopy(shadow_model)
    # w6a6_shadow_model = quant_model(w6a6_smodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    # w6a6_shadow_model.to(device)
    # w6a6_shadow_model.eval()
    
    # set_quant_state(w6a6_shadow_model, input_quant=True, weight_quant=True)
    # with torch.no_grad():
    #     _ = w6a6_shadow_model(calib_data)
    # # save_path = os.path.join(save_folder, 'w6a6_shadow_model.pth')
    # # torch.save(w6a6_shadow_model, save_path)
    # w6a6_shadow_train_test_accuracy = train_test_acc(w6a6_shadow_model, dataloaders_shadow, dataset_sizes_shadow, device)
    # np.save(path + "/res_w6a6_shadow_train_test_accuracy"+".npy", w6a6_shadow_train_test_accuracy)
    
    
    w6a6_dsmodel = copy.deepcopy(distill_shadow_model)
    w6a6_distill_shadow_model = quant_model(w6a6_dsmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    w6a6_distill_shadow_model.to(device)
    w6a6_distill_shadow_model.eval()
    
    set_quant_state(w6a6_distill_shadow_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = w6a6_distill_shadow_model(distill_calib_data)
        
    print('Performing quantization W4A4 ..................................................................................................................')
    
    wq_params = {'n_bits': 4, 'channel_wise': True}
    aq_params = {'n_bits': 4, 'channel_wise': False}
    
    # w4a4_tmodel = copy.deepcopy(target_model)
    # w4a4_target_model = quant_model(w4a4_tmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    # w4a4_target_model.to(device)
    # w4a4_target_model.eval()
    
    # set_quant_state(w4a4_target_model, input_quant=True, weight_quant=True)
    # with torch.no_grad():
    #     _ = w4a4_target_model(calib_data)
    # # save_path = os.path.join(save_folder, 'w4a4_target_model.pth')
    # # torch.save(w4a4_target_model, save_path)
    # w4a4_target_train_test_accuracy = train_test_acc(w4a4_target_model, dataloaders_target, dataset_sizes_target, device)
    # np.save(path + "/res_w4a4_target_train_test_accuracy"+".npy", w4a4_target_train_test_accuracy)
    
    
    w4a4_dtmodel = copy.deepcopy(distill_target_model)
    w4a4_distill_target_model = quant_model(w4a4_dtmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    w4a4_distill_target_model.to(device)
    w4a4_distill_target_model.eval()
    
    set_quant_state(w4a4_distill_target_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = w4a4_distill_target_model(distill_calib_data)
        
        
    # w4a4_smodel = copy.deepcopy(shadow_model)
    # w4a4_shadow_model = quant_model(w4a4_smodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    # w4a4_shadow_model.to(device)
    # w4a4_shadow_model.eval()
    
    # set_quant_state(w4a4_shadow_model, input_quant=True, weight_quant=True)
    # with torch.no_grad():
    #     _ = w4a4_shadow_model(calib_data)
    # # save_path = os.path.join(save_folder, 'w4a4_shadow_model.pth')
    # # torch.save(w4a4_shadow_model, save_path)
    # w4a4_shadow_train_test_accuracy = train_test_acc(w4a4_shadow_model, dataloaders_shadow, dataset_sizes_shadow, device)
    # np.save(path + "/res_w4a4_shadow_train_test_accuracy"+".npy", w4a4_shadow_train_test_accuracy)
    
    
    w4a4_dsmodel = copy.deepcopy(distill_shadow_model)
    w4a4_distill_shadow_model = quant_model(w4a4_dsmodel, input_quant_params=aq_params, weight_quant_params=wq_params)
    w4a4_distill_shadow_model.to(device)
    w4a4_distill_shadow_model.eval()
    
    set_quant_state(w4a4_distill_shadow_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = w4a4_distill_shadow_model(distill_calib_data)
    
    print('Computing similarity W16A16 ..................................................................................................................')
    target_train_preds, target_test_preds = get_preds(target_model, dataloaders_target, device)
    
    # w16a16_target_train_preds, w16a16_target_test_preds = get_preds(w16a16_target_model, dataloaders_target, device)
    w16a16_distill_target_train_preds, w16a16_distill_target_test_preds = get_preds(w16a16_distill_target_model, dataloaders_target, device)
    
    # w16a16_target_train_sim = get_similarity(target_train_preds, w16a16_target_train_preds, args.metric)
    # w16a16_target_test_sim = get_similarity(target_test_preds, w16a16_target_test_preds, args.metric)
    # np.save(path + "/res_w16a16_target_train_sim"+".npy", w16a16_target_train_sim)
    # np.save(path + "/res_w16a16_target_test_sim"+".npy", w16a16_target_test_sim)
    
    w16a16_distill_target_train_sim = get_similarity(target_train_preds, w16a16_distill_target_train_preds, args.metric)
    w16a16_distill_target_test_sim = get_similarity(target_test_preds, w16a16_distill_target_test_preds, args.metric)
    np.save(path + "/res_w16a16_distill_target_train_sim"+".npy", w16a16_distill_target_train_sim)
    np.save(path + "/res_w16a16_distill_target_test_sim"+".npy", w16a16_distill_target_test_sim)
    
    shadow_train_preds, shadow_test_preds = get_preds(shadow_model, dataloaders_shadow, device)
    
    # w16a16_shadow_train_preds, w16a16_shadow_test_preds = get_preds(w16a16_shadow_model, dataloaders_shadow, device)
    w16a16_distill_shadow_train_preds, w16a16_distill_shadow_test_preds = get_preds(w16a16_distill_shadow_model, dataloaders_shadow, device)
    
    # w16a16_shadow_train_sim = get_similarity(shadow_train_preds, w16a16_shadow_train_preds, args.metric)
    # w16a16_shadow_test_sim = get_similarity(shadow_test_preds, w16a16_shadow_test_preds, args.metric)
    # np.save(path + "/res_w16a16_shadow_train_sim"+".npy", w16a16_shadow_train_sim)
    # np.save(path + "/res_w16a16_shadow_test_sim"+".npy", w16a16_shadow_test_sim)
    
    w16a16_distill_shadow_train_sim = get_similarity(target_train_preds, w16a16_distill_shadow_train_preds, args.metric)
    w16a16_distill_shadow_test_sim = get_similarity(target_test_preds, w16a16_distill_shadow_test_preds, args.metric)
    np.save(path + "/res_w16a16_distill_shadow_train_sim"+".npy", w16a16_distill_shadow_train_sim)
    np.save(path + "/res_w16a16_distill_shadow_test_sim"+".npy", w16a16_distill_shadow_test_sim)
    
    
    print('Computing similarity W8A8 ....................................................................................................................')
    # w8a8_target_train_preds, w8a8_target_test_preds = get_preds(w8a8_target_model, dataloaders_target, device)
    w8a8_distill_target_train_preds, w8a8_distill_target_test_preds = get_preds(w8a8_distill_target_model, dataloaders_target, device)
    
    # w8a8_target_train_sim = get_similarity(target_train_preds, w8a8_target_train_preds, args.metric)
    # w8a8_target_test_sim = get_similarity(target_test_preds, w8a8_target_test_preds, args.metric)
    # np.save(path + "/res_w8a8_target_train_sim"+".npy", w8a8_target_train_sim)
    # np.save(path + "/res_w8a8_target_test_sim"+".npy", w8a8_target_test_sim)
    
    w8a8_distill_target_train_sim = get_similarity(target_train_preds, w8a8_distill_target_train_preds, args.metric)
    w8a8_distill_target_test_sim = get_similarity(target_test_preds, w8a8_distill_target_test_preds, args.metric)
    np.save(path + "/res_w8a8_distill_target_train_sim"+".npy", w8a8_distill_target_train_sim)
    np.save(path + "/res_w8a8_distill_target_test_sim"+".npy", w8a8_distill_target_test_sim)
    
    # w8a8_shadow_train_preds, w8a8_shadow_test_preds = get_preds(w8a8_shadow_model, dataloaders_shadow, device)
    w8a8_distill_shadow_train_preds, w8a8_distill_shadow_test_preds = get_preds(w8a8_distill_shadow_model, dataloaders_shadow, device)
    
    # w8a8_shadow_train_sim = get_similarity(shadow_train_preds, w8a8_shadow_train_preds, args.metric)
    # w8a8_shadow_test_sim = get_similarity(shadow_test_preds, w8a8_shadow_test_preds, args.metric)
    # np.save(path + "/res_w8a8_shadow_train_sim"+".npy", w8a8_shadow_train_sim)
    # np.save(path + "/res_w8a8_shadow_test_sim"+".npy", w8a8_shadow_test_sim)
    
    w8a8_distill_shadow_train_sim = get_similarity(target_train_preds, w8a8_distill_shadow_train_preds, args.metric)
    w8a8_distill_shadow_test_sim = get_similarity(target_test_preds, w8a8_distill_shadow_test_preds, args.metric)
    np.save(path + "/res_w8a8_distill_shadow_train_sim"+".npy", w8a8_distill_shadow_train_sim)
    np.save(path + "/res_w8a8_distill_shadow_test_sim"+".npy", w8a8_distill_shadow_test_sim)
    
    print('Computing similarity W6A6 ....................................................................................................................')
    # w6a6_target_train_preds, w6a6_target_test_preds = get_preds(w6a6_target_model, dataloaders_target, device)
    w6a6_distill_target_train_preds, w6a6_distill_target_test_preds = get_preds(w6a6_distill_target_model, dataloaders_target, device)
    
    # w6a6_target_train_sim = get_similarity(target_train_preds, w6a6_target_train_preds, args.metric)
    # w6a6_target_test_sim = get_similarity(target_test_preds, w6a6_target_test_preds, args.metric)
    # np.save(path + "/res_w6a6_target_train_sim"+".npy", w6a6_target_train_sim)
    # np.save(path + "/res_w6a6_target_test_sim"+".npy", w6a6_target_test_sim)
    
    w6a6_distill_target_train_sim = get_similarity(target_train_preds, w6a6_distill_target_train_preds, args.metric)
    w6a6_distill_target_test_sim = get_similarity(target_test_preds, w6a6_distill_target_test_preds, args.metric)
    np.save(path + "/res_w6a6_distill_target_train_sim"+".npy", w6a6_distill_target_train_sim)
    np.save(path + "/res_w6a6_distill_target_test_sim"+".npy", w6a6_distill_target_test_sim)
    
    # w6a6_shadow_train_preds, w6a6_shadow_test_preds = get_preds(w6a6_shadow_model, dataloaders_shadow, device)
    w6a6_distill_shadow_train_preds, w6a6_distill_shadow_test_preds = get_preds(w6a6_distill_shadow_model, dataloaders_shadow, device)
    
    # w6a6_shadow_train_sim = get_similarity(shadow_train_preds, w6a6_shadow_train_preds, args.metric)
    # w6a6_shadow_test_sim = get_similarity(shadow_test_preds, w6a6_shadow_test_preds, args.metric)
    # np.save(path + "/res_w6a6_shadow_train_sim"+".npy", w6a6_shadow_train_sim)
    # np.save(path + "/res_w6a6_shadow_test_sim"+".npy", w6a6_shadow_test_sim)
    
    w6a6_distill_shadow_train_sim = get_similarity(target_train_preds, w6a6_distill_shadow_train_preds, args.metric)
    w6a6_distill_shadow_test_sim = get_similarity(target_test_preds, w6a6_distill_shadow_test_preds, args.metric)
    np.save(path + "/res_w6a6_distill_shadow_train_sim"+".npy", w6a6_distill_shadow_train_sim)
    np.save(path + "/res_w6a6_distill_shadow_test_sim"+".npy", w6a6_distill_shadow_test_sim)
    
    print('Computing similarity W4A4 ....................................................................................................................')
    # w4a4_target_train_preds, w4a4_target_test_preds = get_preds(w4a4_target_model, dataloaders_target, device)
    w4a4_distill_target_train_preds, w4a4_distill_target_test_preds = get_preds(w4a4_distill_target_model, dataloaders_target, device)
    
    # w4a4_target_train_sim = get_similarity(target_train_preds, w4a4_target_train_preds, args.metric)
    # w4a4_target_test_sim = get_similarity(target_test_preds, w4a4_target_test_preds, args.metric)
    # np.save(path + "/res_w4a4_target_train_sim"+".npy", w4a4_target_train_sim)
    # np.save(path + "/res_w4a4_target_test_sim"+".npy", w4a4_target_test_sim)
    
    w4a4_distill_target_train_sim = get_similarity(target_train_preds, w4a4_distill_target_train_preds, args.metric)
    w4a4_distill_target_test_sim = get_similarity(target_test_preds, w4a4_distill_target_test_preds, args.metric)
    np.save(path + "/res_w4a4_distill_target_train_sim"+".npy", w4a4_distill_target_train_sim)
    np.save(path + "/res_w4a4_distill_target_test_sim"+".npy", w4a4_distill_target_test_sim)
    
    # w4a4_shadow_train_preds, w4a4_shadow_test_preds = get_preds(w4a4_shadow_model, dataloaders_shadow, device)
    w4a4_distill_shadow_train_preds, w4a4_distill_shadow_test_preds = get_preds(w4a4_distill_shadow_model, dataloaders_shadow, device)
    
    # w4a4_shadow_train_sim = get_similarity(shadow_train_preds, w4a4_shadow_train_preds, args.metric)
    # w4a4_shadow_test_sim = get_similarity(shadow_test_preds, w4a4_shadow_test_preds, args.metric)
    # np.save(path + "/res_w4a4_shadow_train_sim"+".npy", w4a4_shadow_train_sim)
    # np.save(path + "/res_w4a4_shadow_test_sim"+".npy", w4a4_shadow_test_sim)
    
    w4a4_distill_shadow_train_sim = get_similarity(target_train_preds, w4a4_distill_shadow_train_preds, args.metric)
    w4a4_distill_shadow_test_sim = get_similarity(target_test_preds, w4a4_distill_shadow_test_preds, args.metric)
    np.save(path + "/res_w4a4_distill_shadow_train_sim"+".npy", w4a4_distill_shadow_train_sim)
    np.save(path + "/res_w4a4_distill_shadow_test_sim"+".npy", w4a4_distill_shadow_test_sim)
    
    
    print('Performing membership inference attack (Base) .................................................................................................')
    attack_model, test_loader_attack = mia_base(args, config, target_model, dataloaders_target, dataset_sizes_target, shadow_model, dataloaders_shadow, dataset_sizes_shadow)
    attack_eval(path, attack_model, test_loader_attack, args.device, method="mia_base")
    
    # attack_model, test_loader_attack = mia_base(args, config, distill_model, dataloaders_target, dataset_sizes_target, shadow_model, dataloaders_shadow, dataset_sizes_shadow)
    # attack_eval(path, attack_model, test_loader_attack, args.device, method="mia_base_black")
    
    # attack_model, test_loader = mia_attention_classifier(args, config, q_target_model, dataloaders_target, q_shadow_model, dataloaders_shadow)
    # attack_eval(path, attack_model, test_loader, args.device, method=args.attack_method_attention + "_quantized")
    
    # mia_attention_threshold(args, config, path, q_target_model, dataloaders_target, q_shadow_model, dataloaders_shadow, q="quantized")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('RepQ-ViT', parents=[get_args_parser()])
    args = parser.parse_args()
    main()