# APEX_CAS

> 面向过渡金属簇的活性空间自动构建工具包

## 项目概述

APEX_CAS 是面向过渡金属簇的活性空间自动构建工具包，从 APEX（Automated Progressive Electronic structure eXploration）项目中独立提取。支持结构解析、SCF 计算（UKS/UHF + sf-X2C）、UNO/LUO/AVAS 轨道构建、NOON 质量验证、轨道可视化和 FCIDUMP 生成。基于 Chan 组方法论（Li & Chan, JCTC 2017; Li et al., JCP 2019）。

---

## 目录

- [安装](#安装)
- [依赖](#依赖)
- [Quick Example](#quick-example--以-fe₂s₂-为例)
- [当前 CLI 命令](#当前-cli-命令)
- [计算流水线](#计算流水线)
- [Python API 参考](#python-api-参考)
- [复现脚本](#复现脚本)
- [项目结构](#项目结构)
- [来源](#来源)

---

## 安装

```bash
pip install -e .
```

## 依赖

- numpy >= 1.24
- pyyaml >= 6.0
- pyscf >= 2.4（计算类 CAS 必需）
- h5py（状态持久化）

---

## Quick Example — 以 Fe₂S₂ 为例

以下以 `[Fe2S2(SCH3)4]^{2-}` 体系为例，演示从零开始到生成 FCIDUMP 的完整流程。

### Step 1: 准备 inputs

创建案例目录，放入结构文件（`.xyz`）和计算参数文件（`.yaml`）：

```
examples/fe2s2/
├── inputs/
│   ├── fe2s2.xyz                # 分子结构（标准 XYZ 格式）
│   └── fe2s2_settings.yaml      # CAS 计算参数
└── outputs/                     ← 运行后自动生成
```

**`fe2s2_settings.yaml`** 可从模板修改而来：

```bash
cp shared/config/cas_settings_template.yaml examples/fe2s2/inputs/fe2s2_settings.yaml
```

关键配置（以 2017 年论文参数为例，BP86/sto-3g）：

```yaml
charge: -2                       # 体系总电荷
spin: 0.0                        # 高自旋 S 值（此处 singlet S=0）

scf_method: "uks"                # UKS 或 UHF
xc_functional: "BP86"            # DFT 泛函

basis_set_default: "sto-3g"      # 默认基组
# basis_per_element:             # 也可按元素指定不同基组
#   Fe: "def2-TZVP"
#   S:  "def2-TZVP"
#   C:  "def2-SVP"
#   H:  "def2-SVP"

conv_tol: 1.0e-8                 # SCF 收敛阈值
max_cycle: 500                   # SCF 最大迭代次数
localization_method: "pm"        # 轨道局域化方法: "pm" 或 "boys"
```

模板中所有字段均为注释状态，只取消注释需要修改的字段即可。

### Step 2: 运行 compute

```bash
apex-cas compute examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_settings.yaml
```

| 参数 | 必选 | 说明 |
|------|------|------|
| `structure` | 是 | XYZ/PDB 结构文件路径 |
| `--case-dir` | 否 | 案例目录，inputs 和 outputs 均在其下查找 |
| `--cas-settings` | 否 | YAML 配置文件路径（不指定则使用默认参数） |
| `--charge` / `--spin` | 否 | 命令行覆盖 YAML 中的 charge/spin |

运行完成后，`outputs/` 目录下生成：

```
examples/fe2s2/outputs/
├── scf/
│   ├── scf_info.json            # SCF 收敛信息：能量、活性空间大小、方法
│   └── *.chk                    # PySCF checkpoint 文件
├── orbitals/
│   ├── orbital_report.yaml      # 轨道分类报告（含 selected 字段，见 Step 3）
│   ├── noon_plot.png            # NOON 占据数分布图
│   └── cas_data.h5              # HDF5 格式轨道数据
└── fcidump/                     ← Step 4 后生成
```

**`scf_info.json` 示例：**

```json
{
  "energy": -5021.07,
  "E_core": -5000.12,
  "E_act": -20.95,
  "E_vir": 0.00,
  "E_tot": -5021.07,
  "converged": true,
  "n_electrons": 10,
  "n_orbitals": 10,
  "cpt_cas_type": "uno",
  "source_method": "UKS-BP86/UNO"
}
```

### Step 3: 确认/调整活性轨道

1. **查看 NOON 图**：打开 `outputs/orbitals/noon_plot.png`，确认占据数在 0.02–1.98 之间的轨道（active 区域）是否合理

2. **检查轨道报告**：打开 `outputs/orbitals/orbital_report.yaml`，每个轨道包含：

```yaml
- index: 78
  occupation: 1.8234
  auto_label: "active_0"
  chemical_label: "Fe0_3dxy"
  block: "active"
  selected: true              # ← 控制是否纳入活性空间
```

3. **调整（可选）**：
   - 将不需要的轨道改为 `selected: false`
   - 将需要加入的轨道（如某些 virtual）改为 `selected: true`
   - 如果不修改，直接使用自动选择的默认结果即可

### Step 4: 生成 FCIDUMP

```bash
apex-cas fcidump --case-dir examples/fe2s2
```

| 参数 | 必选 | 说明 |
|------|------|------|
| `--case-dir` | 是 | 案例目录（自动查找 outputs/） |
| `--selection` | 否 | 自定义 orbital_report.yaml 路径（默认自动定位） |
| `--reference-fcidump` | 否 | 参考 FCIDUMP 路径，用于自动对比（相对 case-dir 解析） |
| `--target-spin` | 否 | 目标自旋 MS2（默认 0.0） |

**输出：**

```
examples/fe2s2/outputs/fcidump/FCIDUMP
examples/fe2s2/outputs/fcidump/FCIDUMP.ecore <---E_core
```

标准 FCIDUMP 格式文件，可直接用于 Block2、CheMPS2、Dice 等后处理程序。

**与参考数据对比（可选）：**

```bash
apex-cas fcidump --case-dir examples/fe2s2 \
  --reference-fcidump chan_ref/fe2s2
```

---

## 当前 CLI 命令

所有命令通过 `apex-cas` 入口点调用，定义于 `apex_cas/main.py`。

### `apex-cas compute` — 解析结构 → SCF → CAS → 可视化

完整计算流水线：从结构文件出发，依次完成 SCF 计算、活性空间轨道构建和质量验证，并生成可视化输出。

**参数列表：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `structure` | 位置参数 | — | XYZ/PDB 结构文件路径 |
| `--case-dir` | 路径 | 结构文件所在目录 | 案例目录，所有输出保存在此目录下 |
| `--cas-settings` | 路径 | — | CAS 计算 YAML 配置文件（见下方说明） |
| `--charge` | int | 0（或 YAML 中的值） | 总电荷（覆盖 YAML 中的值） |
| `--spin` | float | 0.0（或 YAML 中的值） | 高自旋 S 值（覆盖 YAML 中的值） |
| `--no-cubes` | 标志 | False | 跳过 cube 文件生成 |
| `--cube-grid` | str | 80x80x80 | Cube 文件网格精度 |

**使用示例：**

```bash
# 使用 YAML 配置文件（推荐）
apex-cas compute Fe2S2.xyz --cas-settings fe2s2_settings.yaml

# 仅使用默认参数
apex-cas compute Fe2S2.xyz --charge -2 --spin 0
```

### CAS 配置文件（YAML）

所有计算参数通过 YAML 配置文件管理。模板位于 `shared/config/cas_settings_template.yaml`

**可配置字段：**

| 字段 | 说明 |
|------|------|
| `preset` | 预设方案（`default` / `fast`） |
| `scf_method` | SCF 方法（`uks` / `uhf`） |
| `xc_functional` | DFT 交换相关泛函（如 B3LYP、BP86） |
| `basis_set_default` | 默认基组 |
| `basis_per_element` | 逐元素基组覆盖 |
| `relativistic` | 标量相对论（`none` / `sf-x2c` / `dkh`） |
| `solvation_model` | 溶剂化模型（`none` / `ddcosmo`） |
| `solvation_epsilon` | 介电常数 |
| `conv_tol` | 能量收敛阈值 |
| `max_cycle` | 最大 SCF 迭代次数 |
| `init_guess` | 初始猜测（`atom` / `minao` / `huckel` / `vsap`） |
| `scf_damp` | 密度阻尼因子 |
| `scf_level_shift` | 虚轨道能级偏移 |
| `diis_space` | DIIS 向量数 |
| `localization_method` | 定域化方法（`pm` / `boys`） |

### `apex-cas fcidump` — 加载状态 → 用户 YAML → FCIDUMP

从已保存的计算状态中加载 CAS 数据，读取用户编辑的轨道选择 YAML，生成标准 FCIDUMP 文件。

**参数列表：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--case-dir` | 路径（必填） | — | 包含 `outputs/` 的案例目录 |
| `--target-spin` | float | 0.0 | 目标自旋 |
| `--output` | 路径 | FCIDUMP | 输出文件路径 |
| `--reference-fcidump` | 路径 | — | 参考 FCIDUMP 文件路径，用于对比验证 |
| `--selection` | 路径 | — | 用户编辑的 `orbital_report.yaml` 路径 |

**使用示例：**

```bash
# 生成 FCIDUMP 文件
apex-cas fcidump --case-dir examples/fe2s2/outputs_2019/ --selection orbital_report.yaml

# 生成并对比参考 FCIDUMP
apex-cas fcidump --case-dir examples/fe2s2/outputs_2019/ \
    --selection orbital_report.yaml \
    --reference-fcidump ref/FCIDUMP_fe2s2.dat
```

---

## 计算流水线

APEX_CAS 的核心是一条六步计算流水线，从原始结构文件出发，最终生成可直接用于下游量子化学计算（如 DMRG、FCI 等）的 FCIDUMP 积分文件。

### Step 1: 结构解析（Structure Parsing）

**功能**：解析 XYZ/PDB 文件，自动识别金属中心、桥联原子和端基配体。

**调用函数**：`parse_structure(filepath, charge=0, target_spin=0.0)`（来自 `structure_analyzer.py`）

**输出**：`ClusterInfo` 数据类，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `metals` | `MetalCenter` 列表 | 过渡金属中心（元素、索引、位置、近邻、配位数） |
| `bridging_atoms` | `BridgingAtom` 列表 | 桥联原子（元素、索引、桥接的金属） |
| `terminal_ligands` | `TerminalLigand` 列表 | 端基配体 |
| `all_elements` | list | 所有元素列表 |
| `all_positions` | ndarray | 所有原子坐标 |
| `formula` | str | 分子式 |
| `total_charge` | int | 总电荷 |
| `target_spin` | float | 目标自旋 |

**核心逻辑**：

1. **过渡金属识别**：从预定义的 `TRANSITION_METALS` 集合（3d、4d、5d 过渡金属元素）中匹配元素符号。
2. **桥联原子识别**：检测与 2 个或以上金属中心成键的原子（常见桥联元素：S, O, N, Cl 等），记录其桥接的金属索引。
3. **端基配体识别**：检测仅与单个金属成键的原子或原子团，归类为端基配体。

### Step 2: SCF 计算（SCF Computation）

**功能**：构建 PySCF 分子对象并运行高自旋 SCF 计算（UKS 或 UHF）。

**调用函数**：

- `build_mol_with_basis(cluster_info, settings)`（来自 `CAS_builder_computing.py`） → 返回 PySCF `Mole` 对象，支持逐元素基组设置
- `_run_high_spin_scf(mol, settings)` → 返回收敛的 SCF 对象

**ComputationSettings 参数**（来自 `shared/models.py`，默认值对应 Chan 2019 方法论）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `scf_method` | `"uks"` | SCF 方法：UKS 或 UHF |
| `xc_functional` | `"B3LYP"` | DFT 交换相关泛函（仅 UKS 模式下有效） |
| `basis_set_default` | `"def2-TZVP"` | 默认基组 |
| `basis_set_per_element` | `{Fe: def2-TZVP, S: def2-TZVP, C: def2-SVP, H: def2-SVP, ...}` | 逐元素基组覆盖 |
| `relativistic` | `"sf-x2c"` | 标量相对论方法（`none` / `sf-x2c` / `dkh`） |
| `solvation_model` | `"ddcosmo"` | 溶剂化模型（`none` / `ddcosmo`） |
| `solvation_epsilon` | 4.0 | 介电常数 |
| `conv_tol` | 1e-8 | SCF 收敛阈值 |
| `max_cycle` | 200 | 最大 SCF 迭代次数 |
| `scf_verbose` | 4 | PySCF SCF 输出级别 |

> [!WARNING]
>
> PySCF 2.12 solvent 与 x2c 不兼容，暂时只能将solvent关闭

**预设方案**：

- **`default`**：对应 Chan 2019 方法论，def2-TZVP/def2-SVP 混合基组，sf-X2C 标量相对论，ddcosmo 溶剂化
- **`fast`**：全部使用 def2-SVP 基组，无相对论校正，较松收敛阈值，适用于快速测试

**输出**：`(mol, mf, chkfile_path)` 元组，其中：
- `mol`：PySCF 分子对象
- `mf`：收敛的 SCF 对象
- `chkfile_path`：检查点文件路径

### Step 3: 轨道构建（Orbital Construction）

**功能**：将 SCF 轨道转化为具有化学意义的活性空间轨道。这是整个流水线的核心步骤。

**入口函数**：`build_computed_CAS(cluster_info, settings, cpt_cas_type="uno")`（来自 `CAS_builder_computing.py`）

**返回**：`CAS` 数据类（含 `mo_coeff_full`, `occupations_full`, `orbital_labels_full` 等字段）

根据 `cpt_cas_type` 参数，支持三种轨道构建路径：

#### UNO 路径（默认，`cpt_cas_type="uno"`）

基于 Chan 2017/2019 方法论的 UNO（United Natural Orbital）路径：

1. **自然轨道计算**：`compute_unos(mol, mf)` — 对角化总（alpha + beta）1-RDM：`eig(S·D·S, S)` → 按占据数降序排列的自然轨道。
2. **分块定域化**：`split_localize(mol, mo_coeff_uno, occ_uno, ...)` — 按占据数将轨道分为三个块：
   - **core**（occ > 1.98）：双占据核心轨道
   - **active**（0.02 ≤ occ ≤ 1.98）：部分占据的活性轨道
   - **virtual**（occ < 0.02）：空轨道
   - 对每个块分别进行 Pipek-Mezey 或 Boys 定域化，赋予化学特征标签。
3. **活性选取**：NOON 在 [0.02, 1.98] 范围内的轨道自动选为活性轨道。
4. **电子数计算**：`n_electrons = round(sum(occupations[active_indices]))`

#### LUO 路径（`cpt_cas_type="luo"`）

用于需要非限制性表示的 DMRG 计算的 LUO（Localized Unrestricted Orbital）路径：

1. 分别定域化 alpha 和 beta MO（占据 + 空轨道块）
2. 通过 `_select_by_projection_threshold(mol, mo_coeff, cluster_info, threshold=0.05)` 选取活性轨道 — 将轨道投影到金属-d + 桥联-p 子空间，投影权重超过阈值的轨道被选为活性轨道
3. 返回独立的 `mo_coeff_alpha` 和 `mo_coeff_beta`，保留自旋极化信息

#### AVAS 路径（`cpt_cas_type="avas"`）

自动价层活性空间（AVAS, Auto-Valence Active Space，Sayfutyarova et al., JCTC 2017）：

1. `_construct_avas(mol, mf, cluster_info, avas_config)` 构建价层子空间
2. `avas_select(mol, mo_coeff, valence_orbitals, threshold=0.4)` — 将 MO 投影到目标 AO 子空间，投影权重超过阈值的轨道被选取
3. 价层轨道从内置知识库自动推断，或通过 `AVASConfig` 手动指定

### Step 4: 质量验证（Quality Validation）

**功能**：通过 NOON（Natural Orbital Occupation Number）分析评估活性空间质量。

**调用函数**：`validate_noon(active_orbitals, expected_types=None, noon_lo=0.02, noon_hi=1.98)`（来自 `CAS_quality.py`）

**返回**：`ActiveSpaceQuality` 数据类：

| 字段 | 说明 |
|------|------|
| `n_doubly_occupied` | NOON > `noon_hi` 的轨道数（惩罚：每个 -0.1） |
| `n_empty` | NOON < `noon_lo` 的轨道数（惩罚：每个 -0.05） |
| `missing_orbital_types` | 未找到的预期轨道类型（惩罚：每个 -0.1） |
| `quality_score` | 质量评分，0-1 分，由各项惩罚计算得出 |

**展示**：`print_quality_report(quality)` → 生成人类可读的质量报告

**评分逻辑**：基础分 1.0，根据以下情况扣分：
- 活性空间中包含应为核心轨道的双占据轨道（NOON 过高）
- 活性空间中包含应为空轨道的轨道（NOON 过低）
- 缺少预期的轨道类型（如金属-d 轨道、桥联-p 轨道等）

### Step 5: 状态保存与可视化（State Persistence & Visualization）

**函数**来自 `orbital_visualizer.py`：

**状态保存** — `save_cas_state(cas, mol, mf, output_dir)`：

保存到以下路径：
- `outputs/scf/chkfile`：PySCF 检查点文件
- `outputs/scf/scf_info.json`：能量、收敛信息
- `outputs/orbitals/cas_data.h5`：HDF5 格式文件，含 `mo_coeff_full`、`occupations_full`、`orbital_labels_full`

**可视化输出** — `plot_orbitals(cas, mol, output_dir, ...)`：

生成以下文件：
- `orbital_report.yaml`：每个轨道包含 `index`、`occupation`、`auto_label`、`chemical_label`、`block`、`selected` 字段
- `noon_plot.png`：NOON 柱状图，直观展示轨道占据数分布
- `cubes/`：轨道 cube 文件（可选，通过 `--no-cubes` 跳过）

**状态加载** — `load_cas_state(case_dir)`：

从磁盘恢复 `(CAS, mol, mf)` 三元组，用于后续 FCIDUMP 生成或其他分析。

### Step 6: FCIDUMP 生成（FCIDUMP Generation）

**功能**：将 AO 积分变换到活性空间 MO 基并写入标准 FCIDUMP 格式，供下游量子化学程序使用。

**工作流**（对应 CLI 命令 `apex-cas fcidump`）：

1. **加载状态**：`load_cas_state(case_dir)` → 恢复 CAS、mol、mf
2. **读取用户选择**：`load_user_selection(yaml_path)` → 读取用户编辑的 YAML 文件（标记为 `selected: true` 的轨道）
3. **生成 FCIDUMP**：`generate_fcidump_from_selection(mol, mf, mo_coeff_loc, occupations, selected_indices, output_path, target_spin)`：
   - 提取活性 MO 系数
   - 计算电子数：`n_electrons = round(sum(occupations[selected_indices]))`
   - 积分变换：`transform_active_integrals(mol, mf, mo_active, n_electrons, target_spin)` → 返回 h1e, eri, ecore, ms2
   - 写入文件：`write_fcidump(integrals, output_path)` → 标准 FCIDUMP 格式
4. **可选对比**：`compare_fcidumps(ref_path, new_path)` → 逐项对比 h1e/h2e/ecore 与参考值

---

## Python API 参考

APEX_CAS 同时提供完整的 Python API，适用于脚本化调用和集成到其他工作流中。以下为 `apex_cas/__init__.py` 导出的核心公开 API。

### 数据模型

| 类名 | 说明 |
|------|------|
| `MetalCenter` | 过渡金属中心（元素、索引、位置、近邻、配位数） |
| `BridgingAtom` | 桥联原子（元素、索引、桥接的金属） |
| `TerminalLigand` | 端基配体 |
| `ClusterInfo` | 簇信息汇总（含 metals, bridging_atoms, terminal_ligands 等） |
| `CAS` | 活性空间数据（含 mo_coeff_full, occupations_full, orbital_labels_full 等） |
| `ActiveSpaceQuality` | 活性空间质量评估结果 |
| `ComputationSettings` | 计算参数设置 |
| `AVASConfig` | AVAS 方法配置 |

### 核心函数

#### 结构解析

```python
from apex_cas import parse_structure

cluster_info = parse_structure("fe2s2.xyz", charge=-2, target_spin=5.0)
```

- `parse_structure(filepath, charge=0, target_spin=0.0)` — 解析 XYZ/PDB 文件，返回 `ClusterInfo`

#### 活性空间构建

```python
from apex_cas import build_computed_CAS, init_computing

# 初始化计算环境
cluster_info, settings = init_computing("fe2s2.xyz", charge=-2, spin=5.0)

# 构建 CAS（UNO 路径）
cas = build_computed_CAS(cluster_info, settings, cpt_cas_type="uno")
```

- `build_NC_CAS(cluster_info, ...)` — 基于规则（知识库）的 CAS 构建
- `build_computed_CAS(cluster_info, settings, cpt_cas_type="uno")` — 计算型 CAS 构建
- `init_computing(structure_path, ...)` — 初始化计算参数

#### 质量验证

```python
from apex_cas import validate_noon, print_quality_report

quality = validate_noon(cas)
print_quality_report(quality)
```

- `validate_noon(active_orbitals, expected_types=None, noon_lo=0.02, noon_hi=1.98)` — NOON 质量验证
- `print_quality_report(quality)` — 打印可读质量报告

#### 可视化与状态持久化

```python
from apex_cas import plot_orbitals, save_cas_state, load_cas_state

# 保存与可视化
save_cas_state(cas, mol, mf, "outputs/")
plot_orbitals(cas, mol, "outputs/")

# 加载
cas, mol, mf = load_cas_state("outputs/")
```

- `generate_orbital_report(cas, output_dir)` — 生成 orbital_report.yaml
- `generate_orbital_cubes(cas, mol, output_dir)` — 生成轨道 cube 文件
- `generate_noon_plot(cas, output_dir)` — 生成 NOON 柱状图
- `plot_orbitals(cas, mol, output_dir, ...)` — 完整可视化输出
- `save_cas_state(cas, mol, mf, output_dir)` — 保存计算状态
- `load_cas_state(case_dir)` — 加载计算状态

#### FCIDUMP 生成

```python
from apex_cas import (
    load_user_selection,
    transform_active_integrals,
    write_fcidump,
    compare_fcidumps,
    generate_fcidump_from_selection,
)

# 从用户选择生成 FCIDUMP
selection = load_user_selection("orbital_report.yaml")
generate_fcidump_from_selection(
    mol, mf, cas.mo_coeff_full, cas.occupations_full,
    selected_indices, "FCIDUMP", target_spin=0.0
)
```

- `load_user_selection(yaml_path)` — 加载用户编辑的轨道选择
- `transform_active_integrals(mol, mf, mo_active, n_electrons, target_spin)` — 积分变换
- `write_fcidump(integrals, output_path)` — 写入 FCIDUMP 文件
- `compare_fcidumps(ref_path, new_path)` — 对比两个 FCIDUMP 文件
- `generate_fcidump_from_selection(...)` — 一站式 FCIDUMP 生成

### 配置

```python
from apex_cas import PRESETS, apply_overrides

# 获取预设
settings = PRESETS["default"]  # 或 "fast"

# 应用覆盖
settings = apply_overrides(settings, xc_functional="PBE", basis_set_default="def2-SVP")
```

- `PRESETS` — 预设配置字典（`"default"` / `"fast"`）
- `apply_overrides(settings, **kwargs)` — 在预设基础上应用参数覆盖

---

## 复现脚本

位于 `scripts/` 目录，用于复现 Chan 组已发表论文中的计算结果。

### 脚本说明

| 脚本 | 论文 | 说明 |
|------|------|------|
| `run_fe2s2_2017.py` | Li & Chan, JCTC 2017 | Fe₂S₂ 计算（BP86/tzp-dkh/sf-X2C，无溶剂化） |
| `run_fe2s2_2019.py` | Li et al., JCP 2019 | Fe₂S₂ 对比计算（B3LYP/TZP-DKH+def2-SVP/sf-X2C/COSMO ε=4.0） |

两者目标活性空间均为 CAS(30e, 20o)，体系为 [Fe₂S₂(SCH₃)₄]²⁻。

### 输出目录

- 2017 复现：`examples/fe2s2/outputs_2017/`
- 2019 复现：`examples/fe2s2/outputs_2019/`

### 运行方式

```bash
# 复现 2017 年结果
python APEX_CAS/scripts/run_fe2s2_2017.py --root /path/to/APEX

# 复现 2019 年结果
python APEX_CAS/scripts/run_fe2s2_2019.py --root /path/to/APEX
```

### 2017 vs 2019 参数差异

| 参数 | 2017 | 2019 |
|------|------|------|
| 泛函 | BP86 | B3LYP |
| 基组 (Fe, S) | tzp-dkh | TZP-DKH |
| 基组 (C, H) | tzp-dkh | def2-SVP |
| 溶剂化 | 无 | COSMO ε=4.0 |
| max_cycle | 2000 | 2000 |

---

## 项目结构

```
APEX_CAS/
├── README.md                        # 英文文档
├── README_CN.md                     # 中文文档（本文件）
├── pyproject.toml                   # 项目配置与构建信息
├── requirements.txt                 # Python 依赖列表
├── apex_cas/
│   ├── __init__.py                  # 公开 API 导出
│   ├── __main__.py                  # 入口点
│   ├── _paths.py                    # 数据文件路径解析
│   ├── main.py                      # CLI（compute + fcidump 命令）
│   ├── structure_analyzer.py        # Step 1: XYZ/PDB 解析
│   ├── CAS_builder_computing.py     # Step 2-3: SCF + 轨道构建
│   ├── CAS_builder_noncomputing.py  # 基于规则的 CAS（知识库）
│   ├── computation_defaults.py      # 预设与配置工具
│   ├── CAS_quality.py                # Step 4: NOON 验证
│   ├── orbital_visualizer.py        # Step 5: 可视化与状态持久化
│   ├── FCIDUMP_generator.py         # Step 6: FCIDUMP 生成
│   ├── orbital_optimizer.py         # UCCSD 轨道优化
│   └── models.py                    # 从 shared.models 重导出
├── scripts/
│   ├── run_fe2s2_2017.py            # Fe₂S₂ 2017 论文复现
│   └── run_fe2s2_2019.py            # Fe₂S₂ 2019 论文复现
├── tests/
│   ├── test_structure_analyzer.py   # 结构解析测试
│   ├── test_active_space_builder.py # 活性空间构建测试
│   ├── test_CAS_quality.py          # 质量验证测试
│   ├── test_computation_settings.py # 计算设置测试
│   ├── test_orbital_optimizer.py    # 轨道优化测试
│   └── test_stage1_methods.py       # Stage 1 方法测试
└── ref/                             # 参考文献计算指南
```

---

## 来源

APEX_CAS 从 APEX（Automated Progressive Electronic structure eXploration）项目中独立提取，专注于过渡金属簇的活性空间自动构建功能。
