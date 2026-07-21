# 当前复现状态

状态：`Iter[D_i O_all]` 主方法已实现，工程审计通过。

已完成：

- 纯 U(4) staircase 解析层；
- χ=2 TT-SVD 与未截断 residual 的正确分离；
- 全线路 forward polar 优化与 r=0.6 式 (6)；
- N=12 三类论文目标；
- TFIM、XXZ 等附加多体基态目标；
- 幅度/平方保真度和基态能量指标；
- 自包含 Hamiltonian CSR 产物、ordered U(4)、statevector 与 SHA-256；
- 逐函数入口审计、独立 oracle、跨实现和论文尺度回归。

审计中特别修复：Schur 负实轴附近错误吸附、非默认 `rank_rtol` 破坏 polar 最优性、
MPS 极端值溢出被误归一化、BAS CLI 默认误成 3×4、基态能量未导出，以及只改 manifest
并重算 checksum 仍能假通过的问题。

边界说明：

- 论文未公开 Figure 4 random-MPS seed，本项目默认 seed=2023，仅作为可复现实例；
- 精确简并基态没有唯一目标向量，本项目要求显式 symmetry/tie-break，否则拒绝；
- T=100/1000 的正式长跑应在 Ubuntu 主机完成；
- 论文方法复现范围仍不含另外五种协议；现已另设只读后处理层，将冻结的
  `Iter[D_i O_all]` U(4) 线路精确综合到 FakeWashingtonV2 原生门并执行 Aer
  密度矩阵噪声评估。该扩展不改变或重新训练论文方法。
