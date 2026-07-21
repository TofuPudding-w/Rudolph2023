# FakeWashingtonV2 扩展本地验证报告

日期：2026-07-19

## 结论

Rudolph 线路到 FakeWashingtonV2 噪声密度矩阵的完整链路已实现，并通过组件 oracle、
真实 fake backend 小系统端到端测试和全项目逐函数入口审计。该功能是冻结论文结果的
只读后处理，不改变 `Iter[D_i O_all]`。

## 组件测试

在 Qiskit 2.3.0、Qiskit Aer 0.17.2、Qiskit IBM Runtime 0.45.1 环境中：

```text
tests/test_qiskit_adapter.py
tests/test_washington_noise.py
tests/test_native_noise_protocol.py

31 passed
```

覆盖：

- 非对称随机 U(4) 的位序与完整矩阵 oracle；
- 多层线路 NumPy/Qiskit logical/native 三路重放；
- 精确 `rz/sx/x/cx` 综合和全局相位；
- compact Target 校准复制与 snapshot 稳定性；
- identity layout、无 SWAP、无 opaque unitary；
- 空噪声 fidelity/energy；
- 三个强度的解析 depolarizing 通道；
- density trace、Hermiticity、PSD 与宽度保护；
- 真实 FakeWashingtonV2 provider；
- 旧 artifact 核心源码兼容校验、原子化导出和 CLI。

## 真实 FakeWashingtonV2 的 Rudolph N=2 端到端结果

输入：N=2 TFIM，`J=1`、`h_x=0.7`，Rudolph depth=2、sweeps=2。

- 无噪声 squared fidelity：`0.9999999999999989`；
- U(4) 逐块综合 fidelity：`0.9999999999999999`；
- FakeWashington Target 编译 fidelity：`0.9999999999999987`；
- 物理路径：`[0,1]`；
- backend snapshot SHA-256：
  `2a8f710b9c6fce598fb4f52124702d37fac2b3bddface60d9630f4319df1307e`；
- SWAP：`0`；
- opaque unitary：`0`；
- CX：`3`；
- SX：`16`；
- RZ：`23`；
- 噪声目标 fidelity：`0.9328105042030751`；
- 噪声 native retention：`0.9328105042030748`；
- 噪声能量误差/site：`0.06669565542597289`；
- density trace error：`6.67e-16`；
- Hermiticity residual：`2.37e-16`；
- minimum eigenvalue：`0.005786227336893455`。

输出 `SHA256SUMS` 中的 backend snapshot、QPY 和 manifest 均重新计算通过。

## 全项目入口审计

在具有 Qiskit 2.2.3/Aer 0.17.2 的本地兼容测试环境中执行非 paper-scale 完整套件：

```text
157 passed, 1 skipped, 3 deselected
function-call audit: 168 / 168
missing: []
Ruff: All checks passed
```

跳过项仅是该环境未安装 FakeWashingtonV2 provider；同一测试已在上述 Runtime 0.45.1
环境独立通过。三个 deselected 是已有 N=12 paper-scale 回归，核心代码未被本次扩展
修改；正式 Python 3.11 Ubuntu 环境仍须执行无筛选的 `python -m pytest -q` 和 Ruff，
其结果作为跨平台最终验收。

## 已知资源边界

N=12 原始 complex128 密度矩阵约 256 MiB，Aer 内部存储和通道执行会占用更多内存。
本地仅完成真实 N=2 噪声端到端测试；N=12 正式噪声实验应在 Ubuntu 主机串行执行。
