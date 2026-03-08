import math
import numpy as np 
import matplotlib.pyplot as plt
from numpy import pi , exp
import torch
import torch.nn as nn
# INVERSE PROBLEM OF OSCILLATOR
# GIVEN THE OBSERVATION POINTS , EXPLORE THE UNKNOWN SYSTEM PARAMETERS

def exact_solution(d, w0, t):
    # 定义欠阻尼谐振子问题的解析解
    assert d < w0  # 确保阻尼系数d小于自然频率w0
    w = np.sqrt(w0**2 - d**2)  # 计算欠阻尼频率
    phi = np.arctan(-d/w)  # 计算初始相位
    A = 1/(2*np.cos(phi))  # 计算振幅
    cos = torch.cos(phi + w * t)  # 计算余弦项
    exp = torch.exp(-d * t)  # 计算指数衰减项
    u = exp * 2 * A * cos  # 计算解
    return u

class FCN(nn.Module):
  # 定义一个全连接神经网络（FCN）类
  def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
    super().__init__()
    activation = nn.Tanh  # 使用双曲正切作为激活函数
    # 第一层全连接层，从输入层到隐藏层
    self.fcs = nn.Sequential(*[ nn.Linear(N_INPUT, N_HIDDEN), activation()])
    # 中间隐藏层，可能有多层
    self.fch = nn.Sequential(*[ nn.Sequential(*[ nn.Linear(N_HIDDEN, N_HIDDEN),activation()]) for _ in range(N_LAYERS-1)])
    # 最后一层全连接层，从隐藏层到输出层
    self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
  def forward(self, x):
      # 定义网络的前向传播过程
      x = self.fcs(x)  # 通过第一层全连接层
      x = self.fch(x)  # 通过中间隐藏层
      x = self.fce(x)  # 通过最后一层全连接层
      return x

# 首先，创建一些带噪声的观测数据
torch.manual_seed(123)  # 设置随机种子，以确保结果的可重复性

d, w0 = 2, 20  # 定义两个变量 d 和 w0
print(f"True value of mu: {2*d}")  # 打印变量 mu 的真实值，这里 mu 被设置为 2*d

t_obs = torch.rand(30).view(-1, 1)  # 生成40个随机观测时间点，值在0到1之间
u_obs = exact_solution(d, w0, t_obs) + 0.04*torch.randn_like(t_obs)  # 生成观测数据。这里首先计算每个时间点的精确解，然后添加高斯噪声
t_test = torch.linspace(0, 1, 30).view(-1, 1)  # 创建100个均匀分布的测试时间点，用于绘制精确解
u_exact = exact_solution(d, w0, t_test)  # 计算测试时间点上的精确解


# ====
torch.manual_seed(123)  # 设置随机种子以确保结果可重复

# 定义一个神经网络用于训练
pinn = FCN(1, 1, 64, 3)  # 初始化一个具有1个输入和1个输出，32个神经元，3个隐藏层的全连接网络

# 定义整个域上的训练点，用于计算物理损失
t_physics = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)  # 创建一个从0到1的30个均匀分布的点，形状为(30, 1)，需要计算梯度
lambda1 = 5e4

# 训练PINN
d, w0 = 2, 20  # 定义物理参数
_, k = 2 * d, w0 ** 2  # 计算另一个物理参数 k
t_test = torch.linspace(0, 1, 100).view(-1, 1)  # 创建300个测试点
u_exact = exact_solution(d, w0, t_test)  # 计算精确解

# ！！！！！！！！
# 将 mu 视为可学习参数，并添加到优化器，这个是重点
# ！！！！！！！！
mu = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))  # 初始化 mu 参数
optimizer = torch.optim.Adam([
 {'params': pinn.parameters(), 'lr': 1e-3}, {'params': [mu],'lr': 1e-2}])  # 创建一个优化器，包含网络参数和 mu

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

# for record the mu during the optimization
mus = []
best_mu = None
best_loss = float('inf')

 
for i in range(20001):
  optimizer.zero_grad()  # 清除旧的梯度
  # 计算物理损失
  u_physics = pinn(t_physics)  # 网络在物理点的输出
  dudt = torch.autograd.grad(u_physics, t_physics, torch.ones_like(u_physics), create_graph=True)[0]  # 输出的一阶导数
  d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]  # 输出的二阶导数
  loss_phy = torch.mean((d2udt2 + mu * dudt + k * u_physics) ** 2)  # 计算物理损失
  # 计算数据损失
  u_obs_pred = pinn(t_obs)
  loss_data = torch.mean((u_obs_pred - u_obs) ** 2)  # 计算观测数据的损失
 
  # 反向传播总损失，更新优化器
  total_loss = loss_phy + lambda1 * loss_data  
  total_loss.backward()
  optimizer.step()
  # scheduler.step()
  if i % 5000 == 0:
    u = pinn(t_test).detach()
    plt.figure(figsize=(6,2.5))
    plt.scatter(t_obs[:,0] , u_obs[:,0],label='noisy observation',alpha= 0.6)
    plt.plot(t_test[:,0], u[:,0], label='PINN Solution' , color='tab:green')
    plt.title(f'training step{i}')
    plt.show()
    print("="*80)
    print(f'the iteration times {i}')
    print(f"physics loss : {loss_phy.item():.6f}")
    print(f"data loss :  {loss_data.item():.6f}")
 
    print(f"total loss : {total_loss.item():.2f}")
    print("="*80)
  
  mus.append(mu.item())
  if total_loss < best_loss:
    best_loss = total_loss.item()
    best_mu = mu.item()
 
# ============ FINAL RESULTS ============
print("="*80)
print(f"\nFINAL RESULTS:")
print(f"  Estimated μ: {best_mu:.6f}")
print(f"  True μ:      {2*d:.6f}")
print(f"  Error:       {abs(best_mu - 2*d):.6f}")
print(f"  Error %:     {100*abs(best_mu - 2*d)/(2*d):.2f}%")
print("="*80)
print(f"  physics loss : {loss_phy.item():.6f}")
print(f"  data loss :      {loss_data.item():.6f}")
print(f"  total loss :     {total_loss.item():.2f}")
print("="*80)
# Final plot
plt.figure(figsize=(12, 5))

# Loss evolution
plt.subplot(1, 2, 1)
plt.semilogy(mus, label="μ evolution", color='green', linewidth=1.5)
plt.axhline(2*d, color='grey', linestyle='--', label="True value", linewidth=2)
plt.xlabel("Iteration", fontsize=11)
plt.ylabel("μ", fontsize=11)
plt.title("Parameter Convergence", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Final solution
plt.subplot(1, 2, 2)
with torch.no_grad():
    u_final = pinn(t_test).detach()
u_exact_final = exact_solution(d, w0, t_test)
plt.plot(t_test[:, 0], u_final[:, 0], label="PINN final", 
        color='green', linewidth=2)
plt.plot(t_test[:, 0], u_exact_final[:, 0], label="Exact", 
        color='grey', linestyle='--', linewidth=2, alpha=0.7)
plt.scatter(t_obs[:, 0], u_obs[:, 0], s=20, alpha=0.5, label="Noisy obs", color='blue')
plt.xlabel("t", fontsize=11)
plt.ylabel("u(t)", fontsize=11)
plt.title("Final Solution", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# # Parameter evolution
# plt.subplot(1, 3, 3)
# plt.plot(mus, label="Estimated μ", color='green', linewidth=2)
# plt.hlines(2*d, 0, len(mus), label="True μ", color='grey', 
          # linestyle='--', linewidth=2)
# plt.fill_between(range(len(mus)), 2*d-0.5, 2*d+0.5, alpha=0.1, color='grey')
# plt.xlabel("Iteration", fontsize=11)
# plt.ylabel("μ", fontsize=11)
# plt.title("Parameter Recovery", fontsize=12)
# plt.legend(fontsize=10)
# plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()