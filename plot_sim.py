import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据：你可以将你的相似度张量转为 NumPy 数组
# similarities_member = your_member_tensor.numpy()
# similarities_nonmember = your_nonmember_tensor.numpy()

# 这里只是随机生成一些示例数据
np.random.seed(42)
similarities_member = np.random.normal(loc=0.97, scale=0.005, size=1000)
similarities_nonmember = np.random.normal(loc=0.99, scale=0.005, size=1000)

# 1. 使用 seaborn 绘制直方图（带核密度估计）
plt.figure(figsize=(6, 4))
sns.histplot(similarities_member, color='blue', kde=True, label='Member', alpha=0.3, bins=30)
sns.histplot(similarities_nonmember, color='orange', kde=True, label='Non-member', alpha=0.3, bins=30)
plt.xlabel("Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.title("(a) Histogram + KDE of Similarities")
plt.savefig('Histogram + KDE of Similarities.png')

# 2. 使用 seaborn 绘制单独的核密度估计曲线（kdeplot）
plt.figure(figsize=(6, 4))
sns.kdeplot(similarities_member, color='blue', label='Member', fill=True)
sns.kdeplot(similarities_nonmember, color='orange', label='Non-member', fill=True)
plt.xlabel("Similarity")
plt.ylabel("Density")
plt.legend()
plt.title("(b) KDE Plot of Similarities")
plt.savefig('KDE Plot of Similarities.png')
