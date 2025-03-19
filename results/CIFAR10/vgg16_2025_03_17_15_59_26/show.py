import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc


def find_best_threshold(threshold_data, threshold_label):
    sorted_data, _ = torch.sort(threshold_data)
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
    return best_threshold


def calculate_accuracy(data, labels, threshold):
    preds = (data >= threshold).astype(np.int32)
    
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


def load_and_print_stats(member_file, nonmember_file, bit_width):
    similarities_member = np.load(member_file)
    similarities_nonmember = np.load(nonmember_file)
    
    if 'shadow' in member_file:
        print(f'{bit_width}-MIA train set distribution', '*'*50)
    elif 'target' in member_file:
        print(f'{bit_width}-MIA test set distribution', '*'*50)
    elif 'distill' in member_file:
        print(f'{bit_width}-MIA test set distribution (distill)', '*'*50)
        
    print("Member: min =", similarities_member.min(), "max =", similarities_member.max(), "std =", similarities_member.std())
    print("Non-member: min =", similarities_nonmember.min(), "max =", similarities_nonmember.max(), "std =", similarities_nonmember.std())

    return similarities_member, similarities_nonmember


def plot_distribution(similarities, filename, xlabel):
    plt.figure(figsize=(6, 4))
    if 'nonmember' in filename:
        sns.histplot(similarities, color='orange', kde=True, alpha=0.3, bins=30)
    elif 'member' in filename:
        sns.histplot(similarities, color='blue', kde=True, alpha=0.3, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(f"{xlabel} Distribution")
    plt.savefig(filename)
    plt.close()
    

def evaluate_threshold(similarities_member, similarities_nonmember):
    similarities_member = torch.from_numpy(similarities_member)
    similarities_nonmember = torch.from_numpy(similarities_nonmember)
    
    threshold_data = torch.cat([similarities_member, similarities_nonmember]).to('cuda')
    threshold_label = torch.cat([
        torch.ones(similarities_member.shape[0], dtype=torch.long), 
        torch.zeros(similarities_nonmember.shape[0], dtype=torch.long)
    ]).to('cuda')

    best_threshold = find_best_threshold(threshold_data, threshold_label)
    
    return threshold_data.cpu().detach().numpy(), threshold_label.cpu().detach().numpy(), best_threshold.cpu().detach().numpy()


def plot_roc_curve(preds, labels, filename):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    print("AUC\n", roc_auc)
    
    target_fpr = 1e-3
    corresponding_tpr = np.interp(target_fpr, fpr, tpr) if target_fpr not in fpr else tpr[np.where(fpr == target_fpr)[0][0]]
    print(f'TPR at FPR = {target_fpr}: {corresponding_tpr}')
    
    fpr[fpr == 0] = 1e-10
    tpr[tpr == 0] = 1e-10

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.xscale('log')
    plt.xlim([1e-3, 1.0])
    plt.yscale('log')
    plt.ylim([1e-3, 1.0])
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Logarithmic Scale')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def show_results(bit_width):
    # distill-shadow
    similarities_member_distill_shadow, similarities_nonmember_distill_shadow = load_and_print_stats(
        f"res_{bit_width}_distill_shadow_train_sim.npy", 
        f"res_{bit_width}_distill_shadow_test_sim.npy", 
        f"{bit_width}"
    )

    plot_distribution(similarities_member_distill_shadow, f"{bit_width}_distill_shadow_member.png", "Similarity (Member)")
    plot_distribution(similarities_nonmember_distill_shadow, f"{bit_width}_distill_shadow_nonmember.png", "Similarity (Non-member)")

    _, _, best_threshold = evaluate_threshold(similarities_member_distill_shadow, similarities_nonmember_distill_shadow)
    
    # distill-target
    similarities_member_distill_target, similarities_nonmember_distill_target = load_and_print_stats(
        f"res_{bit_width}_distill_target_train_sim.npy", 
        f"res_{bit_width}_distill_target_test_sim.npy", 
        f"{bit_width}"
    )

    plot_distribution(similarities_member_distill_target, f"{bit_width}_distill_target_member.png", "Similarity (Member)")
    plot_distribution(similarities_nonmember_distill_target, f"{bit_width}_distill_target_nonmember.png", "Similarity (Non-member)")

    preds_distill_target, labels_distill_target, _ = evaluate_threshold(similarities_member_distill_target, similarities_nonmember_distill_target)
    calculate_accuracy(preds_distill_target, labels_distill_target, best_threshold)
    plot_roc_curve(preds_distill_target, labels_distill_target, f"{bit_width}_distill_ROC_log.png")
    
    
target_train_test_accuracy = np.load("res_target_train_test_accuracy.npy")
print("target_train_test_accuracy\n", target_train_test_accuracy)
shadow_train_test_accuracy = np.load("res_shadow_train_test_accuracy.npy")
print("shadow_train_test_accuracy\n", shadow_train_test_accuracy)
print('*'*100)

show_results("w16a16")
print('*' * 100)
show_results("w8a16")
print('*' * 100)
show_results("w16a8")
print('*' * 100)
show_results("w8a8")
print('*' * 100)
show_results("w6a8")
print('*' * 100)
show_results("w8a6")
print('*' * 100)
show_results("w6a6")
print('*' * 100)
show_results("w4a6")
print('*' * 100)
show_results("w6a4")
print('*' * 100)
show_results("w4a4")
print('*' * 100)


accuracy_QS_MIA = np.load("res_accuracy_QS_MIA.npy")
print("accuracy_QS_MIA\n", accuracy_QS_MIA)

precision_QS_MIA = np.load("res_precision_QS_MIA.npy")
print("precision_QS_MIA\n", precision_QS_MIA)

recall_QS_MIA = np.load("res_recall_QS_MIA.npy")
print("recall_QS_MIA\n", recall_QS_MIA)

labels = np.load("res_label_attack_QS_MIA.npy")
preds = np.load("res_soft_pred_attack_QS_MIA.npy")

plot_roc_curve(preds, labels, 'QS_MIA_ROC_log.png')


accuracy_MIA_base = np.load("res_accuracy_MIA_base.npy")
print("accuracy_MIA_base\n", accuracy_MIA_base)

precision_MIA_base = np.load("res_precision_MIA_base.npy")
print("precision_MIA_base\n", precision_MIA_base)

recall_MIA_base = np.load("res_recall_MIA_base.npy")
print("recall_MIA_base\n", recall_MIA_base)

labels = np.load("res_label_attack_MIA_base.npy")
preds = np.load("res_soft_pred_attack_MIA_base.npy")

plot_roc_curve(preds, labels, 'mia_base_ROC_log.png')
