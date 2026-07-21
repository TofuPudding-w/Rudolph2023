# 约定台账

以下约定会直接影响门矩阵、线路顺序和跨论文数值比较。

| 项目 | 本复现约定 |
|---|---|
| 数值类型 | state/MPS/gate 均为 `complex128`，实数诊断为 `float64` |
| 比特序 | q0 为最高有效位；statevector 索引按 `q0...qN-1` 二进制解释 |
| U(4) 局域基 | `00, 01, 10, 11`；矩阵行为输出、列为输入 |
| MPS 轴序 | `(chi_left, physical, chi_right)` |
| 单层定义 | 相邻门 `(0,1),(1,2),...,(N-2,N-1)` 依次物理执行 |
| 线路层序 | `layers[0]` 最先物理执行；新的解析层 prepend 到输入端 |
| 末端 U(2) | `(I2 ⊗ U2) @ U4_last`，不另计单比特门 |
| 目标全局相位 | 最大模振幅中最小索引者固定为正实数 |
| 论文 fidelity | `f = abs(<target|prepared>)` |
| 论文 infidelity | `1 - f`，不是 `1 - f**2` |
| 跨论文 fidelity | 同时导出 `abs(<.|.>)**2` |
| Heisenberg | `S=σ/2`，3×4 row-major OBC，共 17 条边 |
| BAS | 2×6 row-major，66 个唯一图样的等幅叠加 |
| random MPS | OBC 可行键维数上限 64，张量实部独立采样 N(0,1) |
| polar 目标 | 最大化 `Re Tr(F†U)`；数值零空间采用确定性补全 |
| 式 (6) | `U <- U @ (U† @ U_opt)**r`，默认 `r=0.6` |
| 分数幂 | complex-Schur 严格主分支；仅精确负实轴点取 `+π` |
| sweep | 仅 forward，旧后缀和已更新前缀组成 Gauss–Seidel 更新 |

论文没有公布 Figure 4 使用的 random-MPS seed，也没有规定秩亏酉补全、简并 SVD
子空间或分数幂切口的全部数值规范。因此本项目把这些选择写入 manifest，而不把某个
随机样本声称为论文原始样本的逐位复现。
