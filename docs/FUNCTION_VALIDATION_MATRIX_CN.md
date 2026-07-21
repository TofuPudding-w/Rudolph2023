# 逐函数验证矩阵

“调用过”与“正确”是两回事。本项目对每个函数至少要求入口被完整测试套件执行；核心
科学函数还必须具有不共享实现路径的 oracle 或交叉实现证据。

| 模块 | 功能 | 主要独立证据 |
|---|---|---|
| `conventions` | dtype、归一化、相位、比特数 | 错误输入、相位等价、范数边界测试 |
| `linalg` | SVD、rank、polar、Schur 分数幂、式 (6) | 重构；nuclear-norm oracle；切口两侧；端点 r=0/1；非默认 rank 阈值 |
| `mps` | dense/MPS、canonical、χ 截断 | dense round-trip；NumPy vdot；Eckart–Young；未归一输入；溢出/下溢负测 |
| `completion` | 等距列到酉矩阵 | 规定列逐位保持、unitarity、空列补全、整数类型负测 |
| `circuit` | U(4) 执行、层/线路顺序 | 独立 full-Kronecker matrix oracle、任意输入 U(2) 折叠、空线路宽度负测 |
| `analytic` | χ=2 层和 `D_i` | Ran_2020 独立实现；一般随机 row-isometry；全部 N≤6 基态；GHZ/W |
| `optimizer` | 环境、suffix、polar sweep | 16 个局域矩阵元逐一替换并全线路重放；Ben-Dov 独立实现；naive sweep |
| `protocol` | 逐层增长、全局优化、residual | 相邻深度完整向量恒等式；全部旧层顺序；full-matrix 多层 oracle；更新数公式 |
| `targets` | Heisenberg/BAS/random/TFIM/XXZ | 小系统 Kronecker；singlet −3/4；dense/sparse ED；BAS 独立逻辑谓词；RNG 直比 |
| `metrics` | 两种 fidelity、能量、熵 | 手算态、norm 漂移、Bell 熵、非有限 Hamiltonian 负测 |
| `export` | U(4)/state/Hamiltonian/哈希/重放 | 文件和语义哈希；重算全部 final fidelity；重算能量；篡改后重哈希仍拒绝 |
| `cli` | doctor/smoke/五类 target | 每分支端到端；BAS 无参数默认 2×6；基态输出 ΔE/N |
| `qiskit_adapter` | q0-MSB 映射、U(4) 逻辑线路与精确原生综合 | 非对称随机 U(4) 的 full-Kronecker/Qiskit Operator/NumPy 三方 oracle；4 比特 3 层多 seed；累积全局相位；barrier 等价 |
| `washington_noise` | compact FakeWashington Target、无路由编译、密度矩阵噪声 | 校准逐字段复制；snapshot 确定性；无 SWAP/opaque unitary；零噪声能量；三点解析 depolarizing；trace/Hermiticity/PSD；真实 FakeWashingtonV2 smoke |
| `native_noise_protocol` | 冻结 artifact 的只读噪声后处理和可审计导出 | 原 U(4) 重放；两级编译 fidelity fail-closed；旧 artifact core-source 兼容校验；原子化 QPY/manifest/backend snapshot/SHA-256 导出；CLI 端到端 |

完整测试结束时，`artifacts/function_call_audit.json` 记录源码树与测试树 SHA-256，且
`missing=[]`。该文件只证明函数入口覆盖；数学正确性依据上表各类 oracle。

跨项目验证固定使用：20 个 N=6 复随机态对比 Ran_2020 的压缩态、制备态和 residual；
50 个环境对比 Ben-Dov_2024 的环境和 polar。容许误差阈值为 `1e-11`。
