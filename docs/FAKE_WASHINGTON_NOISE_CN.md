# FakeWashingtonV2 原生编译与噪声评估

## 定位

该功能是论文复现结果的只读后处理层，不属于 Rudolph 等人在论文中提出的
`Iter[D_i O_all]` 优化本身。输入必须是已经由 `rudolph2023 run` 导出并通过完整性
校验的 artifact。噪声命令不会重新优化 U(4)、剪枝、近似综合、改变目标态或执行测量。

处理链为：

1. 从 `result_arrays.npz` 读取物理执行顺序的 U(4)；
2. 按 `project site i -> Qiskit qubit N-1-i` 映射线路；
3. 将每个 U(4) 独立且精确地综合为 `rz/sx/x/cx`；
4. 从 FakeWashingtonV2 复制一条连续物理路径的校准到 N 比特 compact Target；
5. 使用 identity initial layout、`routing_method=none` 和 `approximation_degree=1` 编译；
6. 拒绝任何 SWAP、残留 opaque `unitary`、非 identity final layout 或理想编译保真度失败；
7. 用 Aer density-matrix 方法施加门错误与热弛豫噪声。

不能直接模拟 127 比特 FakeWashingtonV2 的完整密度矩阵。本实现只模拟重编号后的
N 比特 compact backend，同时保存原物理比特路径和逐门校准。

## 指标

- 论文幅度保真度：`|<psi_target|psi_logical>|`；
- 逻辑全局平方保真度：`|<psi_target|psi_logical>|^2`；
- 编译保真度：`|<psi_logical|psi_native>|^2`；
- 噪声目标保真度：`<psi_target|rho_noisy|psi_target>`；
- 噪声线路保持率：`<psi_native|rho_noisy|psi_native>`；
- 基态能量误差/site：`(Tr[H rho_noisy]-E0)/N`。

所有 fidelity 均不做 clipping。论文的幅度 infidelity 与噪声混态平方保真度是不同
定义，导出文件不会把二者混写。

## 噪声边界

- 默认 `gate_error=true`、`thermal_relaxation=true`；
- 因为没有测量，`readout_error=false`；
- `shots=1` 只用于执行保存密度矩阵的无测量线路，保存的是 density-matrix 通道输出，
  不是有限 shots tomography；
- 模型不包含串扰、泄漏、校准漂移；
- 未插入 delay/scheduling，因此不声称完整模拟所有 idle 时段。

## Ubuntu 安装

```bash
conda create -n rudolph2023-py311 python=3.11 -y
conda activate rudolph2023-py311
python -m pip install --upgrade pip
python -m pip install -e '.[dev,plot,noise]'

python - <<'PY'
import qiskit
import qiskit_aer
import qiskit_ibm_runtime
print("qiskit:", qiskit.__version__)
print("qiskit-aer:", qiskit_aer.__version__)
print("qiskit-ibm-runtime:", qiskit_ibm_runtime.__version__)
PY
```

支持并审计的依赖范围是 Qiskit 2.2–2.x、Aer 0.17.x、IBM Runtime
0.45–0.48。正式结果应保留实际版本。

## 小规模真实后端验证

先生成 N=2 TFIM artifact：

```bash
python -m rudolph2023 run \
  --target tfim \
  --num-qubits 2 \
  --coupling 1.0 \
  --field-x 0.7 \
  --depth 2 \
  --sweeps 2 \
  --output artifacts/tfim_n2_noise_source

python -m rudolph2023 native-noise \
  --artifact artifacts/tfim_n2_noise_source \
  --output artifacts/tfim_n2_fake_washington \
  --physical-path 0 1 \
  --optimization-level 0 \
  --seed-transpiler 2026 \
  --seed-simulator 2026 \
  --max-density-matrix-qubits 2

cd artifacts/tfim_n2_fake_washington
sha256sum --check SHA256SUMS
```

应检查：

- `compilation_fidelity` 接近 1；
- `swap_count=0`、`unitary_count=0`；
- trace error 与 Hermiticity residual 接近机器精度；
- noisy target fidelity 是有限的 [0,1] 内数值；
- 后端 snapshot SHA-256、路径、QPY 和输入 artifact hashes 均存在。

## N=12 正式运行

N=12 的一个 complex128 密度矩阵裸数据约 256 MiB；Aer 内部对象、噪声通道和诊断会
需要更多内存。应逐个 artifact 串行运行，不并行保存多个密度矩阵。

首次可省略 `--physical-path`，让程序确定性选取一条连续路径：

```bash
python -m rudolph2023 native-noise \
  --artifact artifacts/tfim_n12_k8_t10 \
  --output artifacts/tfim_n12_k8_t10_washington_auto \
  --optimization-level 0 \
  --seed-transpiler 2026 \
  --seed-simulator 2026 \
  --max-density-matrix-qubits 12
```

从输出的 `native_noise_manifest.json` 读取 `noise_protocol.physical_path`。正式跨方法比较
时，把这条路径显式写回所有方法的命令，并要求
`backend_snapshot_sha256` 完全一致。自动路径是确定性的 smoke 约定，不代表错误率最优路径。

## 输出

- `native_noise_manifest.json`：输入来源、编译诊断、全部指标和环境；
- `fake_washington_compact_snapshot.json`：路径、端序映射和校准；
- `native_circuit.qpy`：可重放原生线路；
- `SHA256SUMS`：上述文件的 SHA-256。

默认不导出原始密度矩阵，以避免 N=12 产生大文件并降低误用风险。
