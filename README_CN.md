# APEX

[English](README.md) | 中文

`APEX` 是一个面向过渡金属簇计算的 session-based 工作流。当前
V1.0.0 架构分成两个子项目：

- `APEX_CAS`
  - 负责准备结构元数据、执行 SCF、构建活性空间、生成 `FCIDUMP`
- `APEX_Filter`
  - 从 `APEX_CAS` 的输出出发，执行分步筛选与基准计算链

当前维护中的 canonical workflow 是：

```text
APEX_CAS:
prepare -> scf -> buildcas -> fcidump -> testcas

APEX_Filter:
load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt -> dmrg-basis -> dmrg -> extrapolate -> report
```

这条 production mainline 只负责计算。验证侧 compare、分析以及矩阵/张量比对都属于独立工作流，不会在运行主线时自动调用。

对于维护中的 Fe2S2 主线示例，请优先参考：

- [docs/example.md](/Users/snh/Projects/APEX/docs/example.md)
- [examples/fe2s2/README_step_by_step.md](/Users/snh/Projects/APEX/examples/fe2s2/README_step_by_step.md)

## 仓库结构

```text
APEX/
├── APEX_CAS/          # 活性空间准备与 FCIDUMP 生成
├── APEX_Filter/       # 基于 session 的活性空间筛选
├── shared/            # 共享数据模型、结构解析、配置模板、输入模板
├── examples/          # 可复现实例
├── docs/              # 面向用户的工作流文档
└── plans/             # 重建与实现计划
```

## Fe2S2 目录分工

维护中的正式示例目录为：

- `examples/fe2s2/`
  - 完整主线案例目录
  - 用于演示从 `prepare` 到 `report` 的 production workflow

## 核心原则

- finalized `cluster_info.yaml` 是后续流程唯一的 cluster annotation authority。
- `APEX_CAS prepare` 是生成该 authority 文件的唯一受支持入口。
- `filter_settings.yaml` 只负责 `APEX_Filter step1 load` 的 bootstrap。
- 从 `step2 enumerate` 开始，数值控制统一放在 session-local 的 `filter_session/method_controls.yaml`。
- `APEX_Filter` 的 active-space 路线直接消费上游 `FCIDUMP` 定义的哈密顿量，不会静默重建一个新的 AO 基问题。

## 快速开始

### 1. 准备 cluster metadata

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

如果你修改了 draft CSV，需要用下面的命令重新生成 finalized YAML：

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml \
  --finalize
```

### 2. 构建活性空间哈密顿量

```bash
apex-cas scf examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml

apex-cas buildcas examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml

apex-cas fcidump --case-dir examples/fe2s2
```

### 3. 启动 filter session

```bash
cp shared/config/filter_settings_template.yaml examples/fe2s2/inputs/fe2s2_filter_settings.yaml

apex-filter load \
  --config examples/fe2s2/inputs/fe2s2_filter_settings.yaml \
  --session examples/fe2s2/filter_session
```

之后依次执行：

```bash
apex-filter enumerate --session examples/fe2s2/filter_session
apex-filter uhf --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step2_enumerate/selection_worklist.csv"
apex-filter ccsd --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step3_uhf/selection_worklist.csv"
apex-filter ccsd-t --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step4_ccsd/selection_worklist.csv"
apex-filter ccsdt --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step5_ccsd_t/selection_worklist.csv"
apex-filter dmrg-basis --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step6_ccsdt/selection_worklist.csv"
apex-filter dmrg --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step7_dmrg_basis/selection_worklist.csv"
apex-filter extrapolate --session examples/fe2s2/filter_session
apex-filter report --session examples/fe2s2/filter_session
```

## 配置文件

- `shared/config/cas_settings_template.yaml`
  - `APEX_CAS` 的 SCF / active-space 设置模板
- `shared/config/filter_settings_template.yaml`
  - `APEX_Filter` 的 step-1 bootstrap 模板
- `shared/config/method_controls_template.yaml`
  - 各 step 数值控制模板，session 创建后会复制到本地 `method_controls.yaml`

## 当前主线范围

当前维护中的示例体系是氧化态 `Fe2S2(SCH3)4^{2-}`。仓库围绕一个干净的V1.0.0工作流组织，特点包括：

- 共享 authority 位于 `shared/`
- 不再允许隐藏式 `cluster_info` fallback 重建
- `method_controls.yaml` 负责 session 内后续数值控制
- `step8` 的 DMRG 路线可以通过 session controls 指向 `pyscf_dmrgci_sz`

`step11+` 的高阶分支代码仍保留在树中，但不属于这轮已经闭合的V1.0.0 rerun / cleanup / authority-validation 范围。

## 子项目文档

- [APEX_CAS/README.md](/Users/snh/Projects/APEX/APEX_CAS/README.md)
- [APEX_CAS/README_CN.md](/Users/snh/Projects/APEX/APEX_CAS/README_CN.md)
- [APEX_Filter/README.md](/Users/snh/Projects/APEX/APEX_Filter/README.md)
