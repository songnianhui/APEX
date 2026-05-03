[English](README.md) | 中文

# APEX_Filter — 基于会话的过渡金属簇电子结构筛选

`APEX_Filter` 是 APEX 仓库中持续维护的下游筛选包。它从 `APEX_CAS` 的输出（`cas_data.h5`、`FCIDUMP`、结构元数据、可选的 `cluster_info.yaml`）出发，驱动一个基于会话的 active space 工作流：

`load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt -> dmrg-basis -> dmrg -> extrapolate -> report`

仓库中仍然保留一个高阶分支：

`ccsdt -> fno-uccsdtq -> cc-composite`

本包围绕两个核心原则设计：

- 保持 Hamiltonian 语义一致：`UHF / UCC / HAST-UCC / DMRG` 均在 `FCIDUMP` 定义的同一 active space 模型上操作。
- 保持筛选工作流可由人工操控：每个主要步骤都会生成机器可读的摘要和电子表格友好的选择文件，以便用户决定哪些构型进入下一阶段。

为了与文献基准工作流对齐，`APEX_Filter` 现在还区分了：

- `symmetry_group`：来自 `APEX_CAS` 元数据的全簇点群标签
- `metal_framework_symmetry`：仅金属骨架的对称性提示
- `reduction_symmetry`：用于族分组时实际使用的下游 `Cn` 风格标签
- `family_scheme`：用于定义特定簇类别族（family）的约定

对于 Chan 风格的复现，默认设置是保持 `config_reduction_mode: "none"`，即在族标记后保留所有文献构型，而非通过启发式方法进行裁剪。

会话引导文件 `filter_settings.yaml` 现在有意收窄了作用范围：

- 它定义了 `step1 load` 所需的 `APEX_CAS` 案例位置和来源信息
- 它不承载下游筛选策略

从 `step2 enumerate` 开始，数值和选择控制参数存放在会话本地的 `filter_session/method_controls.yaml` 中。

对维护中的主线来说，`step3-6` 的 observables 只记录计算结果。benchmark
compare 仍然可以作为独立验证工具使用，但不会再自动注入 production session
产物。

对已经闭合的 V1.0.0 主流程来说，维护中的 mainline 到：

`report`

当前 Fe2S2 重跑 walkthrough 位于：

- [../docs/example.md](/Users/snh/Projects/APEX/docs/example.md)
- [../examples/fe2s2/example.md](/Users/snh/Projects/APEX/examples/fe2s2/example.md)

---

## 安装

```bash
cd APEX_Filter
pip install -e .
```

要求 Python `>= 3.10`。

### 核心依赖

- `numpy`
- `scipy`
- `ase`
- `pyyaml`
- `jinja2`
- `apex-cas`

### 可选后端

- `pyscf`：用于 active space 的 `UHF / UCCSD / UCCSD(T)`
- `block2` + `pyblock2`：用于 DMRG
- `pyhast`：用于 `CCSDT / CCSDTQ`

### 本机 HAST-UCC 环境

编译好的 `hast-ucc` 源码树当前位于：

```bash
/Users/snh/hast-ucc
```

`~/.zshrc` 已配置好，新的 shell 会自动包含：

```bash
export PYTHONPATH=/Users/snh/hast-ucc:$PYTHONPATH
```

这意味着在新的终端会话中可以直接使用 `import pyhast`。

---

## 枚举词汇表

`APEX_Filter` 现在使用固定的层次词汇表来报告枚举结果，避免不同系统之间使用重载的 `config` 计数进行不恰当的比较：

- `raw spin patterns`：位点标记的共线破对称符号模式
- `spin families`：按对称性或基准分组的族
- `spin x oxidation guesses`：唯一的 `(spin pattern, oxidation assignment)` 配对
- `spin x oxidation x d guesses`：唯一的全展开电子构型猜测
- `total configs (saved)`：可选构型裁剪后保留的数量

与基准对齐的参考计数：

| 系统 | 原始自旋模式 | 自旋族 | 自旋 x 氧化态 | 自旋 x 氧化态 x d | 总构型数（已保存） |
|---|---:|---:|---:|---:|---:|
| `Fe2S2` | 2 | 2 | 2 | 2 | 2 |
| `Fe4S4(SCH3)4` | 6 | 3 | 24 | 24 | 24 |
| `Fe4S4H4` | 6 | 3 | 24 | 600 | 600 |
| `FeMo-co` LLDUC | 35 | 10 | 630 | 78750 | 78750 |

对于 Chan 风格的复现，关键点是 `Fe4S4 = 24` 和 `FeMo-co = 78750` *不在*同一层次。`APEX_Filter` 现在在 `step2 enumerate` 过程中会显式打印完整的层次栈。

---

## 快速示例 — 会话工作流

`APEX_Filter` 并非仅从原始结构出发。它期望一个已包含以下内容的 `APEX_CAS` 案例目录：

- SCF 检查点/状态
- 轨道 HDF5 状态
- FCIDUMP

示例案例目录结构：

```text
examples/fe4s4h4/
├── inputs/
│   ├── fe4s4h4.xyz
│   ├── fe4s4h4_cas_settings.yaml
│   ├── fe4s4h4_filter_settings.yaml
│   └── fe4s4h4_cluster_info.yaml   ← 可选但推荐
└── outputs/                     ← 由 apex-cas compute / fcidump 生成
```

### 步骤 0：准备 `APEX_CAS` 输出

典型命令：

```bash
apex-cas compute examples/fe4s4h4/inputs/fe4s4h4.xyz \
  --case-dir examples/fe4s4h4 \
  --cas-settings examples/fe4s4h4/inputs/fe4s4h4_cas_settings.yaml

apex-cas fcidump --case-dir examples/fe4s4h4
```

### 步骤 1：创建 filter 配置文件

从共享模板开始：

```bash
cp shared/config/filter_settings_template.yaml examples/fe4s4h4/inputs/fe4s4h4_filter_settings.yaml
```

最小必要字段：

```yaml
apex_cas_case_dir: examples/fe4s4h4
structure_path: inputs/fe4s4h4.xyz
cluster_info_path: inputs/fe4s4h4_cluster_info.yaml
fcidump_path: outputs/fcidump/FCIDUMP.*
```

路径相对于 `apex_cas_case_dir` 解析。`FCIDUMP.*` 会解析为实际的 FCIDUMP 文件，同时排除 `.ecore` 伴随文件。

对于由 CAS YAML 驱动的基准对齐运行，还应保留：

```yaml
benchmark_profile: ""
family_scheme: ""
config_reduction_mode: "none"
```

### 步骤 2：启动会话

```bash
apex-filter load \
  --config examples/fe4s4h4/inputs/fe4s4h4_filter_settings.yaml \
  --session examples/fe4s4h4/filter_session
```

`step1_load/` 现在记录：

- 解析后的 FCIDUMP 路径
- 从伴随 JSON 继承的有效 `APEX_CAS` 设置
- 当提供 `cluster_info.yaml` 时的簇注释信息
- 重新切分到最终 `selection.txt` / FCIDUMP 选择的 active space 数组

`filter_session/method_controls.yaml` 也会自动创建。该文件是后续方法参数（`enumerate`、`uhf`、`ccsd`、`ccsd_t`、`ccsdt`、`dmrg_basis`、`dmrg`、`fno_uccsdtq`）的会话本地控制界面。

如需从当前 fresh case 布局出发、逐步完成每个维护步骤的
`Fe2S2(SCH3)4^{2-}` 重跑指南，请参阅：

- [../docs/example.md](/Users/snh/Projects/APEX/docs/example.md)
- [../examples/fe2s2/example.md](/Users/snh/Projects/APEX/examples/fe2s2/example.md)

### 步骤 3：运行筛选链

```bash
apex-filter enumerate --session examples/fe4s4h4/filter_session
apex-filter uhf       --session examples/fe4s4h4/filter_session --pick all
apex-filter ccsd      --session examples/fe4s4h4/filter_session --pick "top-per-family 4"
apex-filter ccsd-t    --session examples/fe4s4h4/filter_session --pick "top-per-family 2"
apex-filter ccsdt     --session examples/fe4s4h4/filter_session --pick "top 4"
apex-filter dmrg-basis --session examples/fe4s4h4/filter_session --pick "top 2"
apex-filter dmrg      --session examples/fe4s4h4/filter_session --pick all
apex-filter extrapolate --session examples/fe4s4h4/filter_session
apex-filter report    --session examples/fe4s4h4/filter_session
```

仓库中保留但不属于当前主 benchmark 路径的高阶分支：

```bash
apex-filter fno-uccsdtq --session examples/fe4s4h4/filter_session --pick "top 2" --freeze-occ 2,4
apex-filter cc-composite --session examples/fe4s4h4/filter_session
apex-filter report       --session examples/fe4s4h4/filter_session
```

---

## 选择工作流

每个主要步骤都会将标准化的 **CSV 优先**选择产物写入对应的会话步骤目录：

- `selection_candidates.csv`
- `selection_worklist.csv`
- `selection_guide.md`

`selection_worklist.csv` 是主要的人工可编辑交接文件。它包含说明文件用途的头部注释，以及（当可用时）用于下游筛选的步骤能量值。

编辑 `selection_worklist.csv`，将 `keep` 列标记为以下任一值：

- `1`
- `true`
- `yes`
- `keep`
- `select`
- `include`
- `run`

然后运行：

```bash
apex-filter <next-step> --session <dir> --pick "file /path/to/selection_worklist.csv"
```

注意事项：

- 当预期默认行为为"将所有存活构型送入下一步"时，工作列表通常会预填充 `keep=1`。
- `picked_configs.json` 仍会被写入，但仅作为步骤实际消耗内容的内部溯源记录。它不是面向用户的主要选择界面。

---

## CLI 命令

### `apex-filter load`

加载 `APEX_CAS` 输出并创建会话。

```bash
apex-filter load --config filter_settings.yaml --session <dir>
```

`filter_settings.yaml` 是步骤 1 的引导文件。它设定案例目录、结构元数据和 `FCIDUMP` 来源信息。步骤 2 及后续的控制参数保存在 `filter_session/method_controls.yaml` 中。

### `apex-filter enumerate`

枚举自旋异构体和电子构型。

```bash
apex-filter enumerate --session <dir> [--max-configs N] [--forced-oxidation JSON]
```

枚举策略现在也通过 `filter_session/method_controls.yaml` 中的 `enumerate` 块来控制。CLI 标志在需要直接覆盖时仍然可用。

### `apex-filter uhf`

在 `FCIDUMP` Hamiltonian 上运行 active space 破对称 UHF。

```bash
apex-filter uhf --session <dir> --pick all
```

在能量数据存在之前，仅支持：

- `all`
- `labels ...`
- `file ...`

### `apex-filter ccsd`

从已保存的步骤 3 参考态运行 active space `UCCSD`。

### `apex-filter ccsd-t`

运行 active space `UCCSD(T)`。

### `apex-filter ccsdt`

通过 `HAST-UCC` 运行 active space `CCSDT`。

### `apex-filter dmrg-basis`

准备非限制性 DMRG 轨道基：

- `UCCSD natural orbitals`
- 分裂定域化
- alpha/beta 配对
- DMRG 特定排序

`step7` 现在还会写入常规的基组质量保证文件：

- `dmrg_basis_qc.json`
- `dmrg_basis_qc.csv`

这些是方法质量的诊断文件，不是状态选择标准。

### `apex-filter dmrg`

在选定的预制备基组上运行 active space DMRG。

`dmrg` 步骤现在：

- 从 `filter_session/method_controls.yaml` 读取会话本地控制参数
- 保留逐点的原始求解器日志（`*_dmrg.log`）
- 在摘要中记录 `converged=True/False`、`energy` 和 `wall_time_s`
- 即使某个点未完全收敛，也保留最后可用的能量值

同一会话本地控制文件也控制 `enumerate`，因此选择策略和数值后端设置集中在同一处。

### `apex-filter extrapolate`

将 DMRG 能量外推至无限 bond dimension。

### `apex-filter report`

构建最终的多方法摘要。

当前排序优先级：

- `CC_composite+DMRG_consensus`
- `CC_composite`
- `CCSDT+DMRG_consensus`
- `DMRG_extrapolated`
- `CCSDT`
- `CCSD(T)`

### `apex-filter fno-uccsdtq`

运行当前的高阶分支。重要提示：当前实现明确为：

`spin-resolved occupied-NO freeze (retain all virtual orbitals)`

它还不是完整的虚轨道阈值 FNO 外推工作流。

### `apex-filter cc-composite`

计算：

`E_CCSDT(full) + [E_CCSDTQ(FNO) - E_CCSDT(FNO)]`

并将结果回馈到最终报告层。

---

## 会话目录结构

示例会话目录：

```text
filter_session/
├── session.json
├── method_controls.yaml
├── step1_load/
├── step2_enumerate/
├── step3_uhf/
├── step4_ccsd/
├── step5_ccsd_t/
├── step6_ccsdt/
├── step7_dmrg_basis/
├── step8_dmrg/
├── step9_extrapolate/
├── step10_report/
├── step11_fno_uccsdtq/
└── step12_cc_composite/
```

每个步骤持久保存：

- 已选取的标签
- 结果摘要
- 选择产物
- 后端特定的结果文件（`npz`、临时输出等）

文件约定：

- JSON 摘要包含 `_file_role` / `_comment` 元数据字段。
- CSV 选择文件包含 `#` 头部注释，说明其用途。
- `picked_configs.json` 作为溯源记录保留；交互式选择通过 `selection_worklist.csv` 进行。

这使得工作流可以逐步恢复，而无需重新运行上游阶段。

---

## 当前范围

已纳入维护中的 V1.0.0 路径的内容：

- `load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt`
- `dmrg-basis -> dmrg -> extrapolate -> report`
- 标准化的人工介入选择产物

尚未成为完整维护工作流的内容：

- `step11+` 高阶分支不属于本轮闭合的 V1.0.0 rerun / cleanup 范围
- `fno-uccsdtq -> cc-composite` 不属于当前维护中的 benchmark 主路径
- 大 active space（`117o/180o/285o/404o`）流水线
- QM/MM 平均势能工作流
- 集成到主会话 CLI 中的布居分析
- 作为主流程步骤的 Heisenberg 拟合
- 完整的虚轨道阈值 FNO 外推
