# 本地最终验证报告

验证日期：2026-07-19（Windows，仅作为 Ubuntu 正式验证前的独立门槛）。

环境：Python 3.11.5，NumPy 1.26.4，SciPy 1.14.1，complex128。

## 工程结果

- `pytest -q`：130 passed；
- 源码函数入口审计：111/111，`missing=[]`；
- source-tree SHA-256：`cef75614c98b29c52c78cd54624d17eca7f18d915edbdd7b82e806fbb9da5218`；
- test-tree SHA-256：`02e54afd9c46346a5e4d938f20164a5b0adbae3aec33108d370a7b4a9424dd91`；
- coverage.py branch mode：总覆盖率 85%；
- Ruff 0.12.2：all checks passed；
- `compileall`：通过；
- wheel 构建：通过。

覆盖率不是正确性证明，也不要求每个防御性异常分支都被触发。核心科学路径另外具有解析
oracle、随机性质测试和跨实现对照；函数入口审计用于确保没有完全未执行的源码函数。

## 跨实现结果

20 个 N=6 复随机态对比 Ran_2020：

- χ=2 压缩态最大绝对误差：`1.2071e-14`；
- `D_i` residual 最大绝对误差：`3.0010e-14`；
- 单层制备态最大绝对误差：`1.2094e-14`。

50 个 N=4 环境对比 Ben_Dov_2024：

- 环境矩阵最大误差：`0.0`；
- polar 门最大误差：`0.0`。

完整报告位于 `artifacts/cross_project_validation.json`。

## N=12、K=8、T=10 主方法结果

每个目标均执行 3960 次 U(4) 更新，最终 artifact 通过门重放与语义验证。

| 目标 | 幅度 fidelity | 论文 infidelity `1-f` | 平方 fidelity |
|---|---:|---:|---:|
| 3×4 Heisenberg ground state | 0.9716557982 | 0.0283442018 | 0.9441149902 |
| 2×6 BAS | 0.9963144827 | 0.0036855173 | 0.9926425485 |
| random MPS, seed=2023 | 0.8724798693 | 0.1275201307 | 0.7612211224 |

Heisenberg：`E0=-6.6916801935`，制备态能量 `-6.4579472291`，每格点能量误差
`0.0194777470`。

结果目录为 `artifacts/figure4_main_method_t10_local_v2`。这些数值验证主方法的实现趋势，
但 random seed 并非论文公开 seed，因此不声称逐点重建论文作者的隐藏原始数据。
