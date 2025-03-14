import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc
def find_best_threshold(threshold_data, threshold_label):
    sorted_data, indices = torch.sort(threshold_data)
    sorted_labels = threshold_label[indices]
    best_accuracy = 0.0
    best_threshold = 0.0
    for i in range(len(sorted_data) - 1):
        if sorted_data[i] == sorted_data[i+1]:
            continue
        threshold = (sorted_data[i] + sorted_data[i+1]) / 2.0
        predicted_labels = (threshold_data >= threshold).to(torch.int32)
        accuracy = (predicted_labels == threshold_label).to(torch.float32).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    # print("阈值：", best_threshold)
    # print("准确率：", best_accuracy)
    return best_threshold

def calculate_accuracy(data, labels, threshold):
    preds = (data >= threshold).to(torch.int32)
    
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    
    total = labels.size
    correct = np.sum(labels == preds)
    accuracy = correct / total if total > 0 else 0
    
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 1
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    print("test accuracy: ", accuracy)
    print("test precision: ", precision)
    print("test recall: ", recall)

target_train_test_accuracy = np.load("res_target_train_test_accuracy.npy")
print("target_train_test_accuracy\n", target_train_test_accuracy)
shadow_train_test_accuracy = np.load("res_shadow_train_test_accuracy.npy")
print("shadow_train_test_accuracy\n", shadow_train_test_accuracy)
print('*'*100)

w16a16_target_train_test_accuracy = np.load("res_w16a16_target_train_test_accuracy.npy")
print("w16a16_target_train_test_accuracy\n", w16a16_target_train_test_accuracy)
w16a16_shadow_train_test_accuracy = np.load("res_w16a16_shadow_train_test_accuracy.npy")
print("w16a16_shadow_train_test_accuracy\n", w16a16_shadow_train_test_accuracy)
print('*'*100)

w8a8_target_train_test_accuracy = np.load("res_w8a8_target_train_test_accuracy.npy")
print("w8a8_target_train_test_accuracy\n", w8a8_target_train_test_accuracy)
w8a8_shadow_train_test_accuracy = np.load("res_w8a8_shadow_train_test_accuracy.npy")
print("w8a8_shadow_train_test_accuracy\n", w8a8_shadow_train_test_accuracy)
print('*'*100)

w6a6_target_train_test_accuracy = np.load("res_w6a6_target_train_test_accuracy.npy")
print("w6a6_target_train_test_accuracy\n", w6a6_target_train_test_accuracy)
w6a6_shadow_train_test_accuracy = np.load("res_w6a6_shadow_train_test_accuracy.npy")
print("w6a6_shadow_train_test_accuracy\n", w6a6_shadow_train_test_accuracy)
print('*'*100)

w4a4_target_train_test_accuracy = np.load("res_w4a4_target_train_test_accuracy.npy")
print("w4a4_target_train_test_accuracy\n", w4a4_target_train_test_accuracy)
w4a4_shadow_train_test_accuracy = np.load("res_w4a4_shadow_train_test_accuracy.npy")
print("w4a4_shadow_train_test_accuracy\n", w4a4_shadow_train_test_accuracy)
print('*'*100)

# w2a2_target_train_test_accuracy = np.load("res_w2a2_target_train_test_accuracy.npy")
# print("w2a2_target_train_test_accuracy\n", w2a2_target_train_test_accuracy)
# w2a2_shadow_train_test_accuracy = np.load("res_w2a2_shadow_train_test_accuracy.npy")
# print("w2a2_shadow_train_test_accuracy\n", w2a2_shadow_train_test_accuracy)
# print('*'*100)

# W16A16-----------------------------------------------------------------------------------------------------------------------------------
similarities_member = np.load("res_w16a16_shadow_train_sim.npy")
similarities_nonmember = np.load("res_w16a16_shadow_test_sim.npy")
print('W16A16-MIA训练集分布', '*'*50)
print("Member: min =", similarities_member.min(), "max =", similarities_member.max(), "std =", similarities_member.std())
print("Non-member: min =", similarities_nonmember.min(), "max =", similarities_nonmember.max(), "std =", similarities_nonmember.std())

# 1. 使用 seaborn 绘制直方图（带核密度估计）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_member, color='blue', kde=True, alpha=0.3, bins=30)
plt.xlabel("Similarity (Member)")
plt.ylabel("Density")
plt.title("Member Distribution")
plt.savefig('w16a16_shadow_train.png')
plt.close()

# 2. 使用 seaborn 绘制单独的核密度估计曲线（kdeplot）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_nonmember, color='orange', kde=True, alpha=0.3, bins=30) 
plt.xlabel("Similarity (Non-member)")
plt.ylabel("Density")
plt.title("Non-member Distribution")
plt.savefig('w16a16_shadow_test.png')
plt.close()

similarities_member =torch.from_numpy(similarities_member)
similarities_nonmember = torch.from_numpy(similarities_nonmember)
threshold_data = torch.cat([similarities_member, similarities_nonmember]).to('cuda')
threshold_label = torch.cat([torch.ones(similarities_member.shape[0], dtype=torch.long), torch.zeros(similarities_nonmember.shape[0], dtype=torch.long)]).to('cuda')
best_threshold = find_best_threshold(threshold_data, threshold_label)

print('W16A16-MIA测试集分布', '*'*50)
similarities_member = np.load("res_w16a16_target_train_sim.npy")
similarities_nonmember = np.load("res_w16a16_target_test_sim.npy")

    
print("Member: min =", similarities_member.min(), "max =", similarities_member.max(), "std =", similarities_member.std())
print("Non-member: min =", similarities_nonmember.min(), "max =", similarities_nonmember.max(), "std =", similarities_nonmember.std())

# 1. 使用 seaborn 绘制直方图（带核密度估计）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_member, color='blue', kde=True, alpha=0.3, bins=30)
plt.xlabel("Similarity (Member)")
plt.ylabel("Density")
plt.title("Member Distribution")
plt.savefig('w16a16_target_train.png')
plt.close()

# 2. 使用 seaborn 绘制单独的核密度估计曲线（kdeplot）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_nonmember, color='orange', kde=True, alpha=0.3, bins=30) 
plt.xlabel("Similarity (Non-member)")
plt.ylabel("Density")
plt.title("Non-member Distribution")
plt.savefig('w16a16_target_test.png')
plt.close()

similarities_member =torch.from_numpy(similarities_member)
similarities_nonmember = torch.from_numpy(similarities_nonmember)
threshold_data = torch.cat([similarities_member, similarities_nonmember]).to('cuda')
threshold_label = torch.cat([torch.ones(similarities_member.shape[0], dtype=torch.long), torch.zeros(similarities_nonmember.shape[0], dtype=torch.long)]).to('cuda')
# find_best_threshold(threshold_data, threshold_label)
calculate_accuracy(threshold_data, threshold_label, best_threshold)

preds = threshold_data.cpu().detach().numpy()
labels = threshold_label.cpu().detach().numpy()

fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)
print("AUC\n", roc_auc)

target_fpr = 1e-3
if target_fpr in fpr:
    # 找到 fpr_value 在 fpr 数组中的索引
    index = np.where(fpr == target_fpr)[0][0]  # 找到第一个匹配的索引
    # 根据索引获取对应的 tpr 值
    corresponding_tpr = tpr[index]
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
else:
    print(f"FPR = {target_fpr} 不存在于 FPR 数组中，插值法结果为：")
    corresponding_tpr = np.interp(target_fpr, fpr, tpr)
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
    
fpr[fpr == 0] = 1e-10
tpr[tpr == 0] = 1e-10
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.xscale('log')  # 将x轴（FPR）设置为对数刻度
plt.xlim([1e-3, 1.0])
plt.yscale('log')
plt.ylim([1e-3, 1.0])
plt.xlabel('False Positive Rate (log scale)')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Logarithmic Scale')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('w16a16_ROC_log.png')
plt.close()

print('*'*100)


# W8A8-----------------------------------------------------------------------------------------------------------------------------------
similarities_member = np.load("res_w8a8_shadow_train_sim.npy")
similarities_nonmember = np.load("res_w8a8_shadow_test_sim.npy")
print('W8A8-MIA训练集分布', '*'*50)
print("Member: min =", similarities_member.min(), "max =", similarities_member.max(), "std =", similarities_member.std())
print("Non-member: min =", similarities_nonmember.min(), "max =", similarities_nonmember.max(), "std =", similarities_nonmember.std())

# 1. 使用 seaborn 绘制直方图（带核密度估计）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_member, color='blue', kde=True, alpha=0.3, bins=30)
plt.xlabel("Similarity (Member)")
plt.ylabel("Density")
plt.title("Member Distribution")
plt.savefig('w8a8_shadow_train.png')
plt.close()

# 2. 使用 seaborn 绘制单独的核密度估计曲线（kdeplot）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_nonmember, color='orange', kde=True, alpha=0.3, bins=30) 
plt.xlabel("Similarity (Non-member)")
plt.ylabel("Density")
plt.title("Non-member Distribution")
plt.savefig('w8a8_shadow_test.png')
plt.close()

similarities_member =torch.from_numpy(similarities_member)
similarities_nonmember = torch.from_numpy(similarities_nonmember)
threshold_data = torch.cat([similarities_member, similarities_nonmember]).to('cuda')
threshold_label = torch.cat([torch.ones(similarities_member.shape[0], dtype=torch.long), torch.zeros(similarities_nonmember.shape[0], dtype=torch.long)]).to('cuda')
best_threshold = find_best_threshold(threshold_data, threshold_label)


similarities_member = np.load("res_w8a8_target_train_sim.npy")
similarities_nonmember = np.load("res_w8a8_target_test_sim.npy")
print('W8A8-MIA测试集分布', '*'*50)
print("Member: min =", similarities_member.min(), "max =", similarities_member.max(), "std =", similarities_member.std())
print("Non-member: min =", similarities_nonmember.min(), "max =", similarities_nonmember.max(), "std =", similarities_nonmember.std())

# 1. 使用 seaborn 绘制直方图（带核密度估计）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_member, color='blue', kde=True, alpha=0.3, bins=30)
plt.xlabel("Similarity (Member)")
plt.ylabel("Density")
plt.title("Member Distribution")
plt.savefig('w8a8_target_train.png')
plt.close()

# 2. 使用 seaborn 绘制单独的核密度估计曲线（kdeplot）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_nonmember, color='orange', kde=True, alpha=0.3, bins=30) 
plt.xlabel("Similarity (Non-member)")
plt.ylabel("Density")
plt.title("Non-member Distribution")
plt.savefig('w8a8_target_test.png')
plt.close()

similarities_member =torch.from_numpy(similarities_member)
similarities_nonmember = torch.from_numpy(similarities_nonmember)
threshold_data = torch.cat([similarities_member, similarities_nonmember]).to('cuda')
threshold_label = torch.cat([torch.ones(similarities_member.shape[0], dtype=torch.long), torch.zeros(similarities_nonmember.shape[0], dtype=torch.long)]).to('cuda')
# find_best_threshold(threshold_data, threshold_label)
calculate_accuracy(threshold_data, threshold_label, best_threshold)

preds = threshold_data.cpu().detach().numpy()
labels = threshold_label.cpu().detach().numpy()

fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)
print("AUC\n", roc_auc)

target_fpr = 1e-3
if target_fpr in fpr:
    # 找到 fpr_value 在 fpr 数组中的索引
    index = np.where(fpr == target_fpr)[0][0]  # 找到第一个匹配的索引
    # 根据索引获取对应的 tpr 值
    corresponding_tpr = tpr[index]
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
else:
    print(f"FPR = {target_fpr} 不存在于 FPR 数组中，插值法结果为：")
    corresponding_tpr = np.interp(target_fpr, fpr, tpr)
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
    
fpr[fpr == 0] = 1e-10
tpr[tpr == 0] = 1e-10
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.xscale('log')  # 将x轴（FPR）设置为对数刻度
plt.xlim([1e-3, 1.0])
plt.yscale('log')
plt.ylim([1e-3, 1.0])
plt.xlabel('False Positive Rate (log scale)')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Logarithmic Scale')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('w8a8_ROC_log.png')
plt.close()

print('*'*100)


# W6A6-----------------------------------------------------------------------------------------------------------------------------------
similarities_member = np.load("res_w6a6_shadow_train_sim.npy")
similarities_nonmember = np.load("res_w6a6_shadow_test_sim.npy")
print('W6A6-MIA训练集分布', '*'*50)
print("Member: min =", similarities_member.min(), "max =", similarities_member.max(), "std =", similarities_member.std())
print("Non-member: min =", similarities_nonmember.min(), "max =", similarities_nonmember.max(), "std =", similarities_nonmember.std())

# 1. 使用 seaborn 绘制直方图（带核密度估计）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_member, color='blue', kde=True, alpha=0.3, bins=30)
plt.xlabel("Similarity (Member)")
plt.ylabel("Density")
plt.title("Member Distribution")
plt.savefig('w6a6_shadow_train.png')
plt.close()

# 2. 使用 seaborn 绘制单独的核密度估计曲线（kdeplot）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_nonmember, color='orange', kde=True, alpha=0.3, bins=30) 
plt.xlabel("Similarity (Non-member)")
plt.ylabel("Density")
plt.title("Non-member Distribution")
plt.savefig('w6a6_shadow_test.png')
plt.close()

similarities_member =torch.from_numpy(similarities_member)
similarities_nonmember = torch.from_numpy(similarities_nonmember)
threshold_data = torch.cat([similarities_member, similarities_nonmember]).to('cuda')
threshold_label = torch.cat([torch.ones(similarities_member.shape[0], dtype=torch.long), torch.zeros(similarities_nonmember.shape[0], dtype=torch.long)]).to('cuda')
best_threshold = find_best_threshold(threshold_data, threshold_label)


similarities_member = np.load("res_w6a6_target_train_sim.npy")
similarities_nonmember = np.load("res_w6a6_target_test_sim.npy")
print('W6A6-MIA测试集分布', '*'*50)
print("Member: min =", similarities_member.min(), "max =", similarities_member.max(), "std =", similarities_member.std())
print("Non-member: min =", similarities_nonmember.min(), "max =", similarities_nonmember.max(), "std =", similarities_nonmember.std())

# 1. 使用 seaborn 绘制直方图（带核密度估计）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_member, color='blue', kde=True, alpha=0.3, bins=30)
plt.xlabel("Similarity (Member)")
plt.ylabel("Density")
plt.title("Member Distribution")
plt.savefig('w6a6_target_train.png')
plt.close()

# 2. 使用 seaborn 绘制单独的核密度估计曲线（kdeplot）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_nonmember, color='orange', kde=True, alpha=0.3, bins=30) 
plt.xlabel("Similarity (Non-member)")
plt.ylabel("Density")
plt.title("Non-member Distribution")
plt.savefig('w6a6_target_test.png')
plt.close()

similarities_member =torch.from_numpy(similarities_member)
similarities_nonmember = torch.from_numpy(similarities_nonmember)
threshold_data = torch.cat([similarities_member, similarities_nonmember]).to('cuda')
threshold_label = torch.cat([torch.ones(similarities_member.shape[0], dtype=torch.long), torch.zeros(similarities_nonmember.shape[0], dtype=torch.long)]).to('cuda')
# find_best_threshold(threshold_data, threshold_label)
calculate_accuracy(threshold_data, threshold_label, best_threshold)

preds = threshold_data.cpu().detach().numpy()
labels = threshold_label.cpu().detach().numpy()

fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)
print("AUC\n", roc_auc)

target_fpr = 1e-3
if target_fpr in fpr:
    # 找到 fpr_value 在 fpr 数组中的索引
    index = np.where(fpr == target_fpr)[0][0]  # 找到第一个匹配的索引
    # 根据索引获取对应的 tpr 值
    corresponding_tpr = tpr[index]
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
else:
    print(f"FPR = {target_fpr} 不存在于 FPR 数组中，插值法结果为：")
    corresponding_tpr = np.interp(target_fpr, fpr, tpr)
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
    
fpr[fpr == 0] = 1e-10
tpr[tpr == 0] = 1e-10
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.xscale('log')  # 将x轴（FPR）设置为对数刻度
plt.xlim([1e-3, 1.0])
plt.yscale('log')
plt.ylim([1e-3, 1.0])
plt.xlabel('False Positive Rate (log scale)')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Logarithmic Scale')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('w6a6_ROC_log.png')
plt.close()

print('*'*100)


# W4A4-----------------------------------------------------------------------------------------------------------------------------------
similarities_member = np.load("res_w4a4_shadow_train_sim.npy")
similarities_nonmember = np.load("res_w4a4_shadow_test_sim.npy")
print('W4A4-MIA训练集分布', '*'*50)
print("Member: min =", similarities_member.min(), "max =", similarities_member.max(), "std =", similarities_member.std())
print("Non-member: min =", similarities_nonmember.min(), "max =", similarities_nonmember.max(), "std =", similarities_nonmember.std())

# 1. 使用 seaborn 绘制直方图（带核密度估计）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_member, color='blue', kde=True, alpha=0.3, bins=30)
plt.xlabel("Similarity (Member)")
plt.ylabel("Density")
plt.title("Member Distribution")
plt.savefig('w4a4_shadow_train.png')
plt.close()

# 2. 使用 seaborn 绘制单独的核密度估计曲线（kdeplot）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_nonmember, color='orange', kde=True, alpha=0.3, bins=30) 
plt.xlabel("Similarity (Non-member)")
plt.ylabel("Density")
plt.title("Non-member Distribution")
plt.savefig('w4a4_shadow_test.png')
plt.close()

similarities_member =torch.from_numpy(similarities_member)
similarities_nonmember = torch.from_numpy(similarities_nonmember)
threshold_data = torch.cat([similarities_member, similarities_nonmember]).to('cuda')
threshold_label = torch.cat([torch.ones(similarities_member.shape[0], dtype=torch.long), torch.zeros(similarities_nonmember.shape[0], dtype=torch.long)]).to('cuda')
best_threshold = find_best_threshold(threshold_data, threshold_label)


similarities_member = np.load("res_w4a4_target_train_sim.npy")
similarities_nonmember = np.load("res_w4a4_target_test_sim.npy")
print('W4A4-MIA测试集分布', '*'*50)
print("Member: min =", similarities_member.min(), "max =", similarities_member.max(), "std =", similarities_member.std())
print("Non-member: min =", similarities_nonmember.min(), "max =", similarities_nonmember.max(), "std =", similarities_nonmember.std())

# 1. 使用 seaborn 绘制直方图（带核密度估计）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_member, color='blue', kde=True, alpha=0.3, bins=30)
plt.xlabel("Similarity (Member)")
plt.ylabel("Density")
plt.title("Member Distribution")
plt.savefig('w4a4_target_train.png')
plt.close()

# 2. 使用 seaborn 绘制单独的核密度估计曲线（kdeplot）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_nonmember, color='orange', kde=True, alpha=0.3, bins=30) 
plt.xlabel("Similarity (Non-member)")
plt.ylabel("Density")
plt.title("Non-member Distribution")
plt.savefig('w4a4_target_test.png')
plt.close()

similarities_member =torch.from_numpy(similarities_member)
similarities_nonmember = torch.from_numpy(similarities_nonmember)
threshold_data = torch.cat([similarities_member, similarities_nonmember]).to('cuda')
threshold_label = torch.cat([torch.ones(similarities_member.shape[0], dtype=torch.long), torch.zeros(similarities_nonmember.shape[0], dtype=torch.long)]).to('cuda')
# find_best_threshold(threshold_data, threshold_label)
calculate_accuracy(threshold_data, threshold_label, best_threshold)

preds = threshold_data.cpu().detach().numpy()
labels = threshold_label.cpu().detach().numpy()

fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)
print("AUC\n", roc_auc)

target_fpr = 1e-3
if target_fpr in fpr:
    # 找到 fpr_value 在 fpr 数组中的索引
    index = np.where(fpr == target_fpr)[0][0]  # 找到第一个匹配的索引
    # 根据索引获取对应的 tpr 值
    corresponding_tpr = tpr[index]
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
else:
    print(f"FPR = {target_fpr} 不存在于 FPR 数组中，插值法结果为：")
    corresponding_tpr = np.interp(target_fpr, fpr, tpr)
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
    
fpr[fpr == 0] = 1e-10
tpr[tpr == 0] = 1e-10
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.xscale('log')  # 将x轴（FPR）设置为对数刻度
plt.xlim([1e-3, 1.0])
plt.yscale('log')
plt.ylim([1e-3, 1.0])
plt.xlabel('False Positive Rate (log scale)')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Logarithmic Scale')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('w4a4_ROC_log.png')
plt.close()

print('*'*100)

accuracy_mia_base_classifier = np.load("res_accuracy_mia_base_classifier.npy")
print("accuracy_mia_base_classifier\n", accuracy_mia_base_classifier)

precision_mia_base_classifier = np.load("res_precision_mia_base_classifier.npy")
print("precision_mia_base_classifier\n", precision_mia_base_classifier)

recall_mia_base_classifier = np.load("res_recall_mia_base_classifier.npy")
print("recall_mia_base_classifier\n", recall_mia_base_classifier)

labels = np.load("res_label_attack_mia_base_classifier.npy")
preds = np.load("res_soft_pred_attack_mia_base_classifier.npy")

fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)
print("AUC\n", roc_auc)

target_fpr = 1e-3
if target_fpr in fpr:
    # 找到 fpr_value 在 fpr 数组中的索引
    index = np.where(fpr == target_fpr)[0][0]  # 找到第一个匹配的索引
    # 根据索引获取对应的 tpr 值
    corresponding_tpr = tpr[index]
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
else:
    print(f"FPR = {target_fpr} 不存在于 FPR 数组中，插值法结果为：")
    corresponding_tpr = np.interp(target_fpr, fpr, tpr)
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
    
fpr[fpr == 0] = 1e-10
tpr[tpr == 0] = 1e-10
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.xscale('log')  # 将x轴（FPR）设置为对数刻度
plt.xlim([1e-3, 1.0])
plt.yscale('log')
plt.ylim([1e-3, 1.0])
plt.xlabel('False Positive Rate (log scale)')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Logarithmic Scale')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('mia_base_ROC_log.png')
plt.close()
