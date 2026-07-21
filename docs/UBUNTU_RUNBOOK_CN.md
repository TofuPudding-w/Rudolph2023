# Ubuntu 验证与正式运行手册

假设代码位于 `/home/cg/PycharmProjects/Paper_Reproductions/Rudolph_2023`。

## 1. 创建环境

```bash
cd /home/cg/PycharmProjects/Paper_Reproductions/Rudolph_2023

conda create -n rudolph2023-py311 python=3.11 -y
conda activate rudolph2023-py311

python -m pip install --upgrade pip
python -m pip install -e '.[dev,plot,noise]'

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## 2. 完整工程验证

```bash
python --version
python -m rudolph2023 doctor | tee artifacts/ubuntu_doctor.json
python -m compileall -q src tests scripts
python -m ruff check src tests scripts
python -m pytest -q

python - <<'PY'
import json
from pathlib import Path

audit = json.loads(Path('artifacts/function_call_audit.json').read_text())
assert audit['status'] == 'passed'
assert audit['missing'] == []
print('function audit:', audit['called_count'], '/', audit['expected_count'])
print('source tree:', audit['source_tree_sha256'])
print('test tree:', audit['test_tree_sha256'])
PY
```

## 3. 最小端到端重放

```bash
python -m rudolph2023 smoke --output artifacts/ubuntu_smoke
python -m rudolph2023 verify-artifact artifacts/ubuntu_smoke
```

## 4. 多体基态编码示例

```bash
python -m rudolph2023 run \
  --target tfim \
  --num-qubits 12 \
  --coupling 1.0 \
  --field-x 1.0 \
  --depth 8 \
  --sweeps 10 \
  --output artifacts/tfim_n12_k8_t10

python -m rudolph2023 verify-artifact artifacts/tfim_n12_k8_t10
```

输出同时包含 `paper_infidelity`、`squared_state_fidelity`、`prepared_energy` 和
`energy_error_per_site`。

## 5. Figure 4 主方法长跑

先做 T=10：

```bash
python scripts/run_paper_main_method.py \
  --output-root artifacts/figure4_iter_di_oall_t10 \
  --depth 8 \
  --sweeps 10 \
  --random-seed 2023
```

确认三个 artifact 均 verified 后，再启动 T=100/1000：

```bash
python scripts/run_paper_main_method.py \
  --output-root artifacts/figure4_iter_di_oall_t100_t1000 \
  --depth 8 \
  --sweeps 100 1000 \
  --random-seed 2023
```

建议在 `tmux` 中运行，并保留终端日志。T=1000 的总门更新量是 T=10 的 100 倍，运行
时间也近似线性增加。

## 6. FakeWashingtonV2 噪声验证

先对一个小系统执行真实校准后端 smoke，再运行 N=12。详细边界和指标定义见
`FAKE_WASHINGTON_NOISE_CN.md`。

```bash
python -m rudolph2023 run \
  --target tfim \
  --num-qubits 2 \
  --field-x 0.7 \
  --depth 2 \
  --sweeps 2 \
  --output artifacts/tfim_n2_noise_source

python -m rudolph2023 native-noise \
  --artifact artifacts/tfim_n2_noise_source \
  --output artifacts/tfim_n2_fake_washington \
  --physical-path 0 1 \
  --max-density-matrix-qubits 2

(cd artifacts/tfim_n2_fake_washington && sha256sum --check SHA256SUMS)
```

正式跨方法比较必须显式冻结同一 `physical_path`，并核对
`backend_snapshot_sha256`。N=12 密度矩阵裸数据约 256 MiB，建议串行运行。

## 7. 完整性检查

```bash
find artifacts -type f -print0 | sort -z | xargs -0 sha256sum \
  > artifacts/ubuntu_all_files_sha256.txt
sha256sum artifacts/ubuntu_all_files_sha256.txt
```

正式结果回传时至少保留每个 run 的 `manifest.json`、`result_arrays.npz`、
`SHA256SUMS.json`，以及总汇总 JSON。
