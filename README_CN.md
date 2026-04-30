# APEX

[English](README.md) | 中文

**APEX** — Automated Progressive Electronic structure eXploration，过渡金属簇的活性空间自动化分析、自旋/电子组态枚举与量子化学输入文件生成工具。

## 1. 概述

过渡金属簇的高精度量子化学计算（DMRG、耦合簇等）需要大量繁琐的手工设置：定义活性空间、枚举自旋/电子组态、构建轨道基组以及运行分层计算。本项目将整个工作流自动化，从结构输入到 FCIDUMP 积分文件生成。

典型的参考案例是固氮酶中的 **FeMo-辅因子**（Fe₇MoS₉C），遵循 Li 等人（JCP, 2019）和 Zhai 等人（2026）建立的工作流：

```
Fe₇MoS₉C → (113e, 76o) 活性空间 → 35 个自旋异构体 → 78,750 个电子组态
          → UHF/CCSD/CCSDT/CCSDTQ/DMRG 筛选漏斗 → FCIDUMP → 化学精度
```

### Agent 功能

给定分子结构文件（XYZ/PDB）及电荷和自旋信息：

1. **结构分析** — 识别金属中心、桥联原子、端基配体及近似点群对称性
2. **活性空间构建** — 基于知识库规则，在 minimal/standard/extended 三个层级上构建 (n_electrons, n_orbitals)
3. **自旋异构体枚举** — 枚举所有满足 Σ(±Sᵢ) = target Sz 的共线破缺对称自旋异构体，可选对称性约化
4. **电子组态枚举** — 氧化态优先枚举：先分配氧化态，再枚举自旋异构体，最后枚举 d 轨道选择
5. **筛选漏斗设计** — 设计分层协议：UHF → UCCSD → UCCSD(T) → DMRG → CCSDTQ → DMRG
6. **输入文件生成** — 生成可直接运行的输入脚本，支持 PySCF、BLOCK2、HAST-UCC、ORCA 和 Gaussian
7. **能量外推** — DMRG D-外推、CC 复合能量、FNO 外推和 MP2 空间校正
8. **结果解析与验证** — 解析 QC 输出（`.chk` 和 `.npz`）、验证收敛性、通过布居分析验证氧化态
9. **自动流水线执行** — 端到端执行渐进式筛选漏斗，含高自旋 UHF 去重和 FCIDUMP 生成

## 2. 完整工作流

```
XYZ/PDB 输入
     │
     ▼
┌──────────────────────────┐
│ 2.1 结构分析              │  parse_structure() → ClusterInfo
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 2.2 活性空间构建          │  build_active_space() → ActiveSpace
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 2.3 自旋异构体枚举        │  enumerate_spin_isomers() + 对称性约化
└──────────┬───────────────┘
           ▼
┌───────────────────────────────────────────────────────┐
│ 2.4 电子组态枚举（氧化态优先，v2 API）                │  generate_all_configs_v2()
└──────────┬────────────────────────────────────────────┘
           ▼
┌──────────────────────────┐
│ 2.5 对称性约化            │  reduce_configs_by_symmetry()
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 2.6 筛选漏斗设计          │  design_filtering_funnel() → FilteringPlan
└──────────┬───────────────┘
           ▼
┌───────────────────────────────────────────────────────┐
│ 2.7 渐进式流水线（UHF→UCCSD→DMRG→CCSDTQ）             │  run_pipeline()
└──────────┬────────────────────────────────────────────┘
           ▼
┌──────────────────────────┐
│ 2.8 FCIDUMP 生成         │  generate_fcidump() / generate_fcidump_for_results()
└──────────────────────────┘
```

### 2.1 结构输入

`parse_structure()` 通过 ASE 读取 XYZ 或 PDB 文件，返回 `ClusterInfo` 数据类。识别过渡金属中心（支持 30 种金属）、桥联原子、端基配体，检测近似点群对称性（C3、C4）。

```python
from agent.structure_analyzer import parse_structure

cluster_info = parse_structure("structure.xyz", charge=-2, target_spin=0.0)
# ClusterInfo: metals, bridging_atoms, terminal_ligands, formula, symmetry_group
```

参考：`agent/structure_analyzer.py`

### 2.2 活性空间构建

`build_active_space()` 使用知识库规则和簇模板匹配，在三个层级构建 `ActiveSpace`：

| 层级 | 包含的轨道 | 示例（Fe₂S₂） | 示例（FeMo-co） |
|------|-----------|---------------|----------------|
| **minimal** | 仅金属 d | (10e, 10o) | (56e, 40o) |
| **standard** | + 桥联 p + 间隙原子 | (22e, 16o) | (113e, 76o) |
| **extended** | + 端基配体给体 | 依体系而定 | (277e, 404o) |

参考：`agent/active_space_builder.py`

### 2.3 自旋异构体枚举

`enumerate_spin_isomers()` 枚举所有满足以下约束的共线破缺对称自旋异构体：

```
有效异构体：{σ ∈ {+1, -1}ᴺ | Σᵢ σᵢ·Sᵢ = target_Sz}
```

每个异构体标记为 `BSn-ijk`，其中 n = 少数自旋金属数。`apply_symmetry_reduction()` 将等价异构体归入 `SpinIsomerFamily`。FeMo-co：35 个异构体 → C₃ 下 10 个族。

参考：`agent/spin_config.py`

### 2.4 电子组态枚举

`generate_all_configs_v2()` 使用**氧化态优先**方法，这是推荐 API（旧的 `generate_all_configs()` 是自旋异构体优先）：

```
for 氧化态分配:
    per_metal_S = {i: get_local_spin(element_i, ox_i)}
    spin_isomers = enumerate_spin_isomers(cluster_info, oxidation_states=ox)
    for 自旋异构体:
        for d_轨道组合（笛卡尔积）:
            yield ElectronicConfig
```

**氧化态约束：** `sum(metal_ox) + ligand_charge = total_charge`

每个金属的允许氧化态来自知识库。使用 `forced_oxidation`（CLI：`--metals-oxidation`）可覆盖。

**d 轨道枚举：** 对于具有部分填充壳层的少数自旋金属位点，`enumerate_d_orbital_configs()` 确定额外电子占据哪个 d 轨道。例如 Fe(II) d⁶ 有 5 种选择。

**组态计数公式：**

```
Total = Σ_ox  Π_isomers  Π_sites_with_choice  n_d_choices(site)
```

FeMo-co：35 个异构体 × 18 种 Fe(II)/Fe(III) 分配 × 每个 Fe(II) 5 个 d 轨道选择 = 78,750 个组态。

参考：`agent/electronic_config.py`

### 2.5 对称性约化

`reduce_configs_by_symmetry()` 检测等价金属位点（相同元素、相似几何、配位环境），每个等价类仅保留一个代表。对于对称性较高的簇，可显著减少组态数量。

参考：`agent/electronic_config.py`

### 2.6 筛选漏斗设计

`design_filtering_funnel()` 创建分层筛选方案，提供三种风格：

| 风格 | 层级数 | 说明 |
|------|--------|------|
| `femoco` | 6 | 激进筛选（默认，为 FeMo-co 规模设计） |
| `conservative` | 4 | 每层保留更多组态 |
| `minimal` | 2 | UHF → 最终方法 |

FeMo-co 示例（femoco 风格）：

```
78,750 UHF       →   840  (按能量取前 24/异构体)
  840 UCCSD      →   420  (按能量取前 12/异构体)
  420 UCCSD(T)   →    35  (按能量取前 1/异构体)
   35 DMRG       →    11  (按能量)
   11 CCSDTQ     →     3  (按能量)
    3 DMRG       →     2  (最终候选)
```

参考：`agent/filtering.py`

### 2.7 渐进式流水线执行

`run_pipeline()` 逐级执行筛选漏斗。对 `FilteringPlan` 中的每个层级：

1. 为幸存组态生成输入脚本
2. 通过子进程执行
3. 解析 `.npz` 结果
4. 调用 `select_from_*` 选择最优组态
5. 传递到下一级

**高自旋 UHF 去重：** `_generate_uhf_inputs_deduped()` 将组态按 (charge, spin, basis, d_count_targets) 分组，每组运行一个高自旋 UHF 计算（`pyscf_uhf_highspin.py.j2`），然后各 BS-UHF 脚本加载共享的 NPZ 文件。

**流水线输出目录：**

```
pipeline_output/
├── level_0_UHF/
│   ├── group_XXXXX_highspin.py    # 高自旋共享计算
│   ├── BS2-24_..._uhf.py          # 每组态 BS-UHF 脚本
│   ├── *.npz                       # 结果
│   └── pipeline.log
├── level_1_UCCSD/
├── level_2_UCCSD(T)/
├── ...
├── pipeline_summary.json
└── report.md
```

**恢复执行：** `continue_from` 参数（CLI：`--continue-from`）允许从指定层级索引恢复流水线。

参考：`agent/pipeline.py`

### 2.8 FCIDUMP 生成

`generate_fcidump()` 和 `generate_fcidump_for_results()` 为下游量子化学程序（block2、CheMPS2、Dice、QCMaquis 等）生成标准 FCIDUMP 积分文件。

两种模式：
- **全空间**（`"full"`）：通过 PySCF 的自旋无关积分，使用 α 通道 MO 系数
- **活性空间**（`"active"`）：通过 CASCI 提取活性空间的 1e/2e 积分和核能量

输出文件：`{label}_full.FCIDUMP` 和 `{label}_cas{nelecas}_{ncas}.FCIDUMP`

参考：`agent/fcidump.py`

## 3. 项目结构

```
APEX/
├── agent/                          # 核心 Python 包
│   ├── __init__.py                 # 数据模型：14 个 dataclass + 1 个 enum
│   ├── __main__.py                 # 入口：python -m agent
│   ├── main.py                     # 命令行接口，5 个子命令
│   ├── structure_analyzer.py       # 模块 1：XYZ/PDB 解析，金属/配体识别
│   ├── active_space_builder.py     # 模块 2a：基于规则的活性空间构建
│   ├── orbital_constructor.py      # 模块 2b：DFT→UNO/LUO 流程（PySCF）
│   ├── orbital_ordering.py         # 模块 2c：Fiedler/GA 轨道重排序（DMRG）
│   ├── spin_config.py              # 模块 3：共线 BS 自旋异构体枚举
│   ├── electronic_config.py        # 模块 4：氧化态 + d 轨道占据
│   ├── bs_guess_builder.py         # 模块 4b：ElectronicConfig → 密度矩阵
│   ├── filtering.py                # 模块 5：分层筛选漏斗 + 选择函数
│   ├── input_generator.py          # 模块 6：基于模板的输入文件生成
│   ├── energy_extrapolation.py     # 模块 7：DMRG/CC/FNO/MP2 外推
│   ├── population_analysis.py      # 模块 8：Mulliken/Meta-Löwdin 布居分析
│   ├── result_parser.py            # 模块 9：QC 输出解析（.chk, .npz, stdout）
│   ├── pipeline.py                 # 模块 10：端到端渐进式筛选执行
│   ├── fcidump.py                  # 模块 11：FCIDUMP 积分文件生成
│   ├── report.py                   # Markdown/JSON 报告生成
│   └── llm_agent.py                # LLM Agent 层（Anthropic API）
├── knowledge_base/                 # YAML 知识库
│   ├── transition_metals.yaml      # 30 种过渡金属：d 电子数、自旋、半径
│   ├── ligand_database.yaml        # 桥联/给体原子，端基配体
│   ├── cluster_templates.yaml      # FeMo-co、Fe₄S₄、Fe₂S₂、Mn₄CaO₅ 参考数据
│   └── basis_sets.yaml             # 推荐基组和 DFT 泛函
├── templates/                      # Jinja2 输入文件模板
│   ├── pyscf_uhf.py.j2             # BS-UHF 自旋翻转初始猜测
│   ├── pyscf_uhf_highspin.py.j2    # 高自旋 UHF + d 计数编码（共享预计算）
│   ├── pyscf_ccsd.py.j2            # UCCSD/UCCSD(T) 含 T1 诊断
│   ├── pyscf_casscf.py.j2          # CASSCF/RASSCF 态平均
│   ├── block2_dmrg.py.j2           # DMRG 自适应 sweep 调度
│   ├── hast_ucc.py.j2              # 高阶量身定制 CC 流程
│   └── slurm_job.sh.j2             # SLURM 批量提交模板
├── tests/                          # 9 个测试文件 + 1 个演示脚本
├── examples/                       # 示例结构和脚本
│   ├── fe2s2/fe2s2.xyz             # Fe₂S₂ 二聚体
│   ├── fe4s4/fe4s4.xyz             # Fe₄S₄ 立方烷
│   ├── femoco/                     # FeMo-辅因子（参考数据）
│   ├── vh4/vh4.xyz                 # VH₄ 四面体
│   ├── vcl4/vcl4.xyz               # VCl₄ 四面体
│   ├── v2o3/v2o3.xyz               # V₂O₃
│   └── run_analysis.py             # 示例分析脚本
└── requirements.txt
```

## 4. 安装

### 前提条件

- Python >= 3.10
- pip

### 必需：PySCF

PySCF 是核心流水线功能的**必需依赖**：

```bash
pip install pyscf>=2.5
```

PySCF 用于：UHF/CCSD/CCSD(T) 计算、高自旋 UHF 预计算、FCIDUMP 积分文件生成、布居分析、轨道构建（UNO/LUO）。

### 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：

| 包 | 用途 |
|----|------|
| `numpy>=1.24` | 数值数组 |
| `scipy>=1.10` | 优化（曲线拟合、线性分配） |
| `ase>=3.22` | 结构文件解析（XYZ、PDB） |
| `pyyaml>=6.0` | 知识库加载 |
| `jinja2>=3.1` | 模板渲染 |
| `anthropic>=0.40` | LLM Agent 层（可选） |

### 可选依赖

| 包 | 用途 |
|----|------|
| `block2` | DMRG 计算层级 |
| `HAST-UCC` | CCSDTQ 计算层级 |

## 5. 使用方法

### 5.1 `analyze` — 完整分析流程

```bash
python -m agent analyze STRUCTURE [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `STRUCTURE` | （必填） | XYZ 或 PDB 结构文件路径 |
| `--charge` | `0` | 分子总电荷 |
| `--spin` | `0.0` | 目标自旋 S（如 1.5 表示 S=3/2） |
| `--level` | `standard` | 活性空间层级：`minimal`、`standard`、`extended` |
| `--output, -o` | （自动） | 输出目录（默认：`{结构目录}/pipeline_output`） |
| `--report-format` | `markdown` | 报告格式：`markdown` 或 `json` |
| `--generate-inputs` | 关闭 | 生成 QC 输入文件 |
| `--execute` | 关闭 | 执行渐进式筛选流水线 |
| `--n-final` | `5` | 最终保留的组态数（配合 `--execute`） |
| `--code` | `pyscf` | 目标 QC 程序：`pyscf`、`block2`、`orca`、`gaussian` |
| `--method` | `uhf` | QC 方法：`uhf`、`ccsd`、`casscf`、`dmrg` |
| `--basis` | `cc-pVDZ` | 基组名称 |
| `--max-configs` | （无） | 枚举电子组态的上限 |
| `--style` | `femoco` | 筛选漏斗风格：`femoco`、`conservative`、`minimal` |
| `--metals` | （自动） | 手动指定金属元素，覆盖自动检测 |
| `--metals-oxidation` | （无） | 强制氧化态，如 `0:3,1:3` 表示 site0=+3,site1=+3 |
| `--continue-from` | `0` | 从指定层级索引恢复流水线（从 0 开始，配合 `--execute`） |
| `--no-fcidump` | 关闭 | 禁止为最终结果生成 FCIDUMP |

示例：

```bash
# VH4 — 单金属体系
python -m agent analyze examples/vh4/vh4.xyz --charge 0 --spin 0.5 \
    --metals-oxidation "0:4"

# VCl4 — 强制氧化态
python -m agent analyze examples/vcl4/vcl4.xyz --charge 0 --spin 0.5 \
    --metals-oxidation "0:4"

# Fe2S2 — 执行流水线
python -m agent analyze examples/fe2s2/fe2s2.xyz --charge -2 --spin 0 \
    --execute --n-final 3

# Fe4S4 — 从第 2 级恢复
python -m agent analyze examples/fe4s4/fe4s4.xyz --charge -2 --spin 0 \
    --execute --continue-from 2 --n-final 5

# V2O3 — 禁止 FCIDUMP
python -m agent analyze examples/v2o3/v2o3.xyz --charge 0 --spin 0 \
    --execute --no-fcidump
```

### 5.2 `spin` — 自旋异构体枚举

```bash
python -m agent spin --metals ELEMENTS... --spin TARGET_SZ [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--metals` | （必填） | 空格分隔的金属元素符号 |
| `--spin` | （必填） | 目标总 Sz |
| `--oxidation` | （自动） | 每个金属的氧化态 |
| `--symmetry` | `C1` | 近似点群 |

示例：

```bash
python -m agent spin --metals Fe Fe Fe Fe Fe Fe Fe Mo --spin 2.0 --oxidation 3 3 3 3 3 3 3 3
```

### 5.3 `filter` — 筛选漏斗设计

```bash
python -m agent filter --n-configs N --n-electrons N --n-orbitals N [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-configs` | （必填） | 电子组态总数 |
| `--n-electrons` | （必填） | 活性空间电子数 |
| `--n-orbitals` | （必填） | 活性空间轨道数 |
| `--n-isomers` | （无） | 自旋异构体数量 |
| `--style` | `femoco` | 漏斗风格：`femoco`、`conservative`、`minimal` |

### 5.4 `orbitals` — 轨道构建（需要 PySCF）

```bash
python -m agent orbitals STRUCTURE [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `STRUCTURE` | （必填） | 结构文件路径 |
| `--charge` | `0` | 总电荷 |
| `--spin` | `0.0` | 目标自旋 S |
| `--basis` | `cc-pVDZ` | 基组 |
| `--functional` | `B3LYP` | UKS 的 DFT 泛函 |
| `--orbital-type` | `restricted_uno` | `restricted_uno`（UNO）或 `unrestricted_luo`（LUO） |
| `--output, -o` | （无） | 轨道数据的输出 .npz 文件 |

### 5.5 `fcidump` — FCIDUMP 生成

```bash
python -m agent fcidump STRUCTURE [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `STRUCTURE` | （必填） | 结构文件路径（XYZ/PDB） |
| `--charge` | `0` | 总电荷 |
| `--spin` | `0.0` | 目标自旋 S |
| `--basis` | `cc-pVDZ` | 基组 |
| `--uhf-npz` | （无） | UHF `*_uhf.npz` 路径（提供则跳过 SCF） |
| `--dmrg-npz` | （无） | DMRG `*_dmrg_results.npz` 路径（获取 ncore/cas 信息） |
| `--pipeline-dir` | （无） | 流水线输出目录，自动定位 NPZ 文件 |
| `--active-electrons` | （无） | 活性空间电子数（无 DMRG NPZ 时使用） |
| `--active-orbitals` | （无） | 活性空间轨道数（无 DMRG NPZ 时使用） |
| `--mode` | `both` | FCIDUMP 模式：`both`、`full`、`active` |
| `--output, -o` | （自动） | 输出目录（默认：`{结构目录}/fcidump/`） |

示例：

```bash
# 从流水线结果自动生成
python -m agent fcidump examples/vh4/vh4.xyz --charge 0 --spin 0.5 \
    --pipeline-dir examples/vh4/pipeline_output --mode both

# 指定 NPZ 文件
python -m agent fcidump examples/fe2s2/fe2s2.xyz --charge -2 --spin 0 \
    --uhf-npz path/to/uhf.npz \
    --dmrg-npz path/to/dmrg_results.npz \
    --active-electrons 22 --active-orbitals 16
```

## 6. 模块架构

系统采用四层架构：

### A 层：纯 Python（无 QC 依赖）

模块 1、2a、3、4、5 — 基于规则的分析与枚举。完全可单元测试，无需 PySCF。

```
structure_analyzer.py → active_space_builder.py → spin_config.py → electronic_config.py → filtering.py
```

### B 层：PySCF 驱动

模块 2b、2c、4b、6、9、11 — 轨道构建、输入生成、结果解析、FCIDUMP。需要 PySCF。

```
orbital_constructor.py → orbital_ordering.py → bs_guess_builder.py
input_generator.py → result_parser.py → fcidump.py
```

- **UNO**（非限制性自然轨道）：对总 (α+β) 1-RDM 对角化 → 自然占据数
- **LUO**（局域非限制性轨道）：分别对 α 和 β 轨道进行局域化，用于 DMRG
- **Fiedler 排序**：双轨道相互作用矩阵的谱重排序
- **遗传算法排序**：可选的 GA 优化器

### C 层：流水线编排

`pipeline.py` — 端到端执行渐进式筛选漏斗，含高自旋 UHF 去重、`continue_from` 恢复和 `generate_fcidump` 选项。

### D 层：LLM Agent

`llm_agent.py` — Claude API 封装，用于交互式推理、歧义消解和结果解读。

## 7. Python API

所有功能均可作为 Python 库使用：

### 结构分析

```python
from agent.structure_analyzer import parse_structure

cluster_info = parse_structure("structure.xyz", charge=-2, target_spin=0.0)
print(cluster_info.formula)           # "Fe2S2"
print(len(cluster_info.metals))       # 2
print(cluster_info.symmetry_group)    # "C1"
```

### 活性空间构建

```python
from agent.active_space_builder import build_active_space
from agent import ActiveSpaceLevel

aspace = build_active_space(cluster_info, ActiveSpaceLevel.STANDARD)
print(f"({aspace.n_electrons}e, {aspace.n_orbitals}o)")
```

### 自旋异构体枚举

```python
from agent.spin_config import enumerate_spin_isomers, apply_symmetry_reduction, label_isomers
import numpy as np

isomers = enumerate_spin_isomers(
    cluster_info, target_Sz=0.0, oxidation_states={0: 3, 1: 3},
)

metal_positions = np.array([m.position for m in cluster_info.metals])
families = apply_symmetry_reduction(isomers, "C3", metal_positions)
families = label_isomers(families)
```

### 电子组态枚举（v2 API）

```python
from agent.electronic_config import generate_all_configs_v2, reduce_configs_by_symmetry

# 氧化态优先枚举（推荐）
configs = generate_all_configs_v2(cluster_info, forced_oxidation={0: 3, 1: 3})
configs = reduce_configs_by_symmetry(configs, cluster_info)
print(f"Total: {len(configs)}")
```

### 筛选漏斗

```python
from agent.filtering import design_filtering_funnel

plan = design_filtering_funnel(
    n_configs=len(configs), active_space=aspace,
    n_spin_isomers=35, style="femoco",
)
```

### 流水线执行

```python
from agent.pipeline import run_pipeline

final_results = run_pipeline(
    configs=configs,
    active_space=aspace,
    cluster_info=cluster_info,
    plan=plan,
    workdir="pipeline_output",
    code="pyscf",
    basis_set="cc-pVDZ",
    n_final=5,
    continue_from=0,          # 从第 0 级恢复
    generate_fcidump=True,    # 为最终结果生成 FCIDUMP
)

for rank, r in enumerate(final_results, 1):
    print(f"  #{rank} {r.method} E = {r.energy:.12f} [{r.config.label}]")
```

### FCIDUMP 生成

```python
from agent.fcidump import generate_fcidump, generate_fcidump_for_results

# 单个组态
info = generate_fcidump(
    cluster_info, aspace, "path/to/uhf.npz", "output_dir/",
    dmrg_npz="path/to/dmrg_results.npz", mode="both",
)
print(info["full_space"])   # 全空间 FCIDUMP 路径
print(info["active_space"]) # 活性空间 FCIDUMP 路径

# 批量为流水线结果生成
info_list = generate_fcidump_for_results(
    final_results, cluster_info, aspace, "pipeline_output", plan,
    basis_set="cc-pVDZ",
)
```

### 能量外推

```python
from agent.energy_extrapolation import dmrg_d_extrapolation, cc_composite_energy

# DMRG D-外推
result = dmrg_d_extrapolation(
    bond_dims=[500, 1000, 2000, 5000, 10000],
    energies=[-100.05, -100.02, -100.01, -100.003, -100.001],
)
print(f"E_inf = {result.energy:.6f} +/- {result.uncertainty:.6f}")
```

### 结果解析

```python
from agent.result_parser import parse_npz_result, to_calculation_result

result = parse_npz_result("config_label_uhf.npz")
calc = to_calculation_result(result, electronic_config)
print(f"Energy: {calc.energy}, Converged: {calc.converged}")
```

## 8. LLM Agent

Agent 通过交互式 LLM 接口暴露所有分析工具：

```python
from agent.llm_agent import run_agent, run_interactive

result = run_agent(
    "Analyze the Fe7MoS9C cluster at charge=-1 for DMRG calculation. "
    "What active space and how many spin isomers should I expect?"
)

# 或交互式运行：
run_interactive()
```

需要 `anthropic` 包和 `ANTHROPIC_API_KEY` 环境变量。Agent 提供 9 个工具：

| 工具 | 说明 |
|------|------|
| `analyze_structure` | 解析结构文件，识别金属/配体 |
| `build_active_space` | 在指定层级构建活性空间 |
| `enumerate_spin_configs` | 列出所有自旋异构体（含对称性约化） |
| `enumerate_electronic_configs` | 完整电子组态枚举 |
| `design_filtering_protocol` | 分层筛选漏斗设计 |
| `generate_input_file` | 生成 QC 输入脚本 |
| `parse_results` | 解析 QC 输出文件 |
| `extrapolate_energy` | DMRG/CC 能量外推 |
| `query_knowledge_base` | 查询金属/配体/簇参考数据 |

## 9. 知识库

### `transition_metals.yaml`

**30 种过渡金属**（3d: Sc–Zn, 4d: Y–Cd, 5d: La, Hf–Hg）的属性：
- 电子组态、常见氧化态
- 高自旋 d 电子数和 S 值
- 共价半径（用于成键检测）
- 按氧化态的高自旋数据

### `ligand_database.yaml`

- 桥联元素（S、O、N、C、Cl、Se）及其典型电荷和活性轨道数
- 给体原子属性
- 常见端基配体（半胱氨酸硫醇盐、组氨酸咪唑等）

### `cluster_templates.yaml`

已知簇的验证参考数据：
- **FeMo-辅因子**：(113e, 76o) 活性空间、35 个自旋异构体、C₃ 对称性下 10 个 BS 族、78,750 个电子组态
- **Fe₄S₄ 立方烷**：(22e, 20o) 最小、(46e, 32o) 含硫
- **Fe₂S₂ 二聚体**：(10e, 10o) 最小、(22e, 16o) 含硫
- **Mn₄CaO₅ 放氧复合体**：(24e, 20o) S₁ 态最小

### `basis_sets.yaml`

过渡金属簇中常见元素的推荐基组和 DFT 泛函。

## 10. 输入文件模板

| 模板 | 程序 | 方法 | 特性 |
|------|------|------|------|
| `pyscf_uhf.py.j2` | PySCF | BS-UHF | 自旋翻转初始猜测、Mulliken 分析、DIIS 控制、NPZ 输出 |
| `pyscf_uhf_highspin.py.j2` | PySCF | 高自旋 UHF | 共享高自旋计算、d 计数编码、保存 NPZ 供 BS-UHF 复用 |
| `pyscf_ccsd.py.j2` | PySCF | UCCSD/UCCSD(T) | T₁ 诊断、密度拟合、冻结自然轨道、加载 UHF NPZ |
| `pyscf_casscf.py.j2` | PySCF | CASSCF/RASSCF | 态平均、轨道旋转、DMRG 接口 |
| `block2_dmrg.py.j2` | BLOCK2 | DMRG | 自适应 sweep 调度、Fiedler 排序、噪声项 |
| `hast_ucc.py.j2` | HAST-UCC | CCSDT/CCSDTQ | 高阶量身定制 CC 流程 |
| `slurm_job.sh.j2` | SLURM | — | 批量提交含资源配置 |

当 Jinja2 模板未找到时，`input_generator.py` 模块会回退到内置生成器，直接生成完整可运行的 Python 脚本（PySCF）或 shell 脚本（其他程序）。

---

## 附录

### 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定模块
pytest tests/test_pipeline.py -v
pytest tests/test_electronic_config.py -v

# 流水线演示
python tests/run_pipeline_demo.py --example fe2s2 --dry-run
python tests/run_pipeline_demo.py --example vh4 --charge 0 --spin 0.5
```

所有测试均为纯 Python，无需 QC 程序。测试覆盖范围：

| 测试文件 | 覆盖模块 |
|----------|----------|
| `test_core.py` | 所有核心模块（冒烟测试） |
| `test_structure_analyzer.py` | 结构解析、金属识别、对称性 |
| `test_active_space_builder.py` | 活性空间规则、氧化态 |
| `test_spin_config.py` | 自旋枚举、对称性约化、Heisenberg 排序 |
| `test_electronic_config.py` | 氧化态枚举、d 轨道选择、完整组态生成 |
| `test_energy_extrapolation.py` | DMRG 拟合、CC 复合、FNO、MP2 校正 |
| `test_input_generator.py` | 模板渲染、批量生成、批量提交脚本 |
| `test_pipeline.py` | NPZ 解析、选择函数、端到端流水线执行 |
| `run_pipeline_demo.py` | 交互式演示含预设示例 |

### 参考文献

- Zhai et al., "Classical solution of the FeMo-cofactor model to chemical accuracy and its implications," arXiv:2601.04621, 2026
- Li, Zhai, Chan, "The electronic complexity of the ground-state of the FeMo cofactor of nitrogenase as relevant to quantum simulations," JCP 150, 024302, 2019
- Reiher et al., "Elucidating reaction mechanisms on quantum computers," PNAS 114, 7555, 2017

### 许可证

本项目仅供研究使用。

**作者：** Song@Elab
