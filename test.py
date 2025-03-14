import torch

# 生成第一个概率向量 P：100 维，假设第 0 个位置为 1，其余为 0
P = torch.zeros(100)
P[0] = 1.0

# 生成第二个概率向量 Q：每个元素都是 0.01
Q = torch.full((100,), 0.01)

# 为防止 log(0) 问题，加上一个很小的 eps，但这里 P 中只有1和0，所以可以忽略
eps = 1e-10
P = P + eps
Q = Q + eps

# 计算 KL 散度
kl = torch.sum(Q * (torch.log(Q) - torch.log(P)))
print("KL divergence:", kl.item())