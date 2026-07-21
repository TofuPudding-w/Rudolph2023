# 验证协议与通过标准

## 快速门槛

```bash
python -m compileall -q src tests scripts
python -m ruff check src tests scripts
python -m pytest -q -m 'not slow'
```

## 完整门槛

```bash
python -m pytest -q
```

完整门槛包含 N=12、K=8、T=10 的 Heisenberg 3×4、BAS 2×6 和 random-MPS
seed=2023 回归。必须同时满足：

- 测试零失败；
- `artifacts/function_call_audit.json` 的 `status=passed`、`missing=[]`；
- 每组线路形状 `(8,11,4,4)`；
- 每组总更新数 3960；
- 环境与 residual 恒等式误差小于 `1e-10`；
- Heisenberg 基态能量为 `-6.691680193514956 ± 2e-10`；
- 固定环境下 T=10 的最终幅度保真度与本项目回归值在 `2e-3` 内。

回归值不是论文原始数据文件，也不能取代公式级验证。它用于发现代码、BLAS 或分支约定
变化；允许 `2e-3` 是因为简并 SVD 子空间和秩亏补全的门 gauge 可跨平台不同。

## 跨实现验证

```bash
python scripts/cross_validate_prior_implementations.py \
  --ran-root /path/to/Ran_2020 \
  --bendov-root /path/to/Ben_Dov_2024 \
  --output artifacts/cross_project_validation.json
```

报告 `status` 必须为 `passed`，所有最大误差必须小于 `1e-11`。

## 产物验证含义

`verify-artifact` 会验证当前源码树、内部文件/数组哈希、有限性、归一化、线路结构、门重放、
全部最终 fidelity、更新数恒等式以及可选 Hamiltonian 能量。`SHA256SUMS.json` 没有外部
签名，因此它能发现意外损坏和内部不一致，不能抵御攻击者同时篡改全部文件；文档不把它
描述成密码学签名。
