[English](compare.md) | 中文

# Compare 工具说明

本文件说明独立于 production mainline 的 compare 工具。

production workflow 仍然保持为纯计算：

- `APEX_CAS`
  - `prepare -> scf -> buildcas -> fcidump -> testcas`
- `APEX_Filter`
  - `load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt -> dmrg-basis -> dmrg -> extrapolate -> report`

compare 工具属于验证侧能力，用于：

- retained artifact 检查
- rerun 验证
- benchmark / regression 分析

## 推荐入口

直接使用：

```bash
python scripts/compare_artifacts.py FILE1 FILE2
```

示例：

```bash
python scripts/compare_artifacts.py ref.FCIDUMP new.FCIDUMP
python scripts/compare_artifacts.py ref_uhf.h5 new_uhf.h5
python scripts/compare_artifacts.py ref_dmrg_basis.npz new_dmrg_basis.npz --format json
```

## 当前支持的产物类型

目前已经支持：

- `FCIDUMP`
  - 一电子矩阵本征值比较
  - 二电子张量 RMS / max 差异
  - `ECORE` 差异
- `*.h5` / `*.hdf5`
  - step3 UHF HDF5
  - step7 DMRG-basis HDF5
  - step8 DMRG HDF5
  - `APEX_CAS testcas` DMRG HDF5
  - 以及 generic HDF5 fallback
- `*.npz`
  - 按 key 的数组比较
- `*.json`
  - 扁平化 scalar path 比较

## Compare 的判断原则

对于矩阵类和张量类结果，compare 会优先使用更能反映本质差异的指标，而不是只看逐元素差异。

例如：

- `1-RDM`
  - elementwise delta
  - eigenspectrum delta
  - trace / `trace(gamma^2)`
  - `basis_rotation_likely`
- basis state
  - active subspace 是否一致
  - 最佳匹配重叠
  - ordering / pairing 差异
- `2PDM`
  - elementwise delta
  - pair-space spectrum
  - 若干 invariant

这样可以更清楚地区分：

- 真实物理差异
- 仅仅是 basis / ordering / pairing 的表示差异

## Python API

如果想在 Python 中直接调用：

```python
from shared.comparison import compare_artifacts

result = compare_artifacts("ref_uhf.h5", "new_uhf.h5")
```

常用底层接口还包括：

- `compare_fcidumps(...)`
- `compare_density_matrices(...)`
- `compare_basis_states(...)`
- `compare_two_particle_density_tensors(...)`
- `compare_matrix_entries(...)`
- `compare_matrix_spectra(...)`

## Fe2S2 专用脚本

仓库中仍保留：

```bash
python scripts/compare_fe2s2_runs.py --current examples/fe2s2 --baseline /path/to/fe2s2
```

这个脚本仍然是 Fe2S2 专用 compare。  
如果需要通用文件级 compare，请优先使用：

`scripts/compare_artifacts.py`
