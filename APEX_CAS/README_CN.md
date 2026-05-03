# APEX_CAS

`APEX_CAS` 是 `APEX` 中负责活性空间准备的子项目。V1.0.0 的正式工作流是：

```text
prepare -> scf -> buildcas -> fcidump -> testcas
```

它负责：

- 生成权威的 `cluster_info.yaml`
- 运行高自旋 SCF 参考态
- 构建活性空间轨道状态
- 写出 `FCIDUMP` 与 `ECORE`
- 可选执行一个小规模 DMRG smoke test

**[English README](README.md)**

## 安装

```bash
cd APEX_CAS
pip install -e .
```

## 工作流概览

### 1. `prepare`

`prepare` 是生成后续流程所用 cluster metadata 的唯一受支持入口。

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

会生成：

- `*_cluster_info_draft.csv`
- `*_structure_labeled.png`
- 使用 `--finalize` 后生成最终 `*_cluster_info.yaml`

推荐流程：

1. 先生成 draft CSV 和带标签结构图
2. 如有需要，人工修正 draft CSV
3. 再次运行并加 `--finalize`
4. 后续所有 `APEX_CAS` / `APEX_Filter` 计算都以 finalized `cluster_info.yaml` 为唯一 authority

### 2. `scf`

```bash
apex-cas scf examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

会写出：

- `outputs/scf/*.chk`
- `outputs/scf/*_scf_info.json`
- `outputs/scf/*_cas_info.json`

### 3. `buildcas`

```bash
apex-cas buildcas examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

该阶段会恢复 SCF 状态、构建活性空间轨道，并写出：

- `outputs/orbitals/*_cas_data.h5`
- `outputs/orbitals/*_selection.txt`
- `outputs/orbitals/*_orbital_report.md`
- `outputs/orbitals/*_noon_plot.png`

如果只想走一个命令，`compute` 仍可用，但文档主推荐路径仍是 `scf -> buildcas`。

### 4. `fcidump`

```bash
apex-cas fcidump --case-dir examples/fe2s2
```

会读取保存好的 CAS 状态与 `*_selection.txt`，写出：

- `outputs/fcidump/FCIDUMP.*`
- `outputs/fcidump/FCIDUMP.*.ecore`
- `outputs/fcidump/*_fcidump_info.json`

维护中的 `fcidump` 主线只负责生成哈密顿量产物；与 reference 的比较属于验证侧动
作，不再由 production `apex-cas fcidump` 命令自动触发。

### 5. `testcas`

```bash
apex-cas testcas examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh -M 500
```

这是对已生成活性空间哈密顿量做的小规模 DMRG smoke test。

## Fe2S2 示例

针对已验证的 Fe2S2 oxidized benchmark，可参考：

- [docs/example.md](/Users/snh/Projects/APEX/docs/example.md)
- [examples/fe2s2/example.md](/Users/snh/Projects/APEX/examples/fe2s2/example.md)

## CLI 摘要

### `apex-cas prepare`

```bash
apex-cas prepare structure.xyz --case-dir <case_dir> --cas-settings <yaml> [--finalize]
```

用于生成并最终确认 cluster annotations。finalized `cluster_info.yaml` 是下游唯一 authority。

### `apex-cas scf`

```bash
apex-cas scf structure.xyz --case-dir <case_dir> --cas-settings <yaml>
```

只运行 SCF 参考态。

### `apex-cas buildcas`

```bash
apex-cas buildcas structure.xyz --case-dir <case_dir> --cas-settings <yaml>
```

从已保存 SCF checkpoint 构建活性空间状态。

### `apex-cas compute`

```bash
apex-cas compute structure.xyz --case-dir <case_dir> --cas-settings <yaml>
```

这是 `scf -> buildcas` 的快捷封装，保留是为了工作流便利，不是正式版主推荐入口。

### `apex-cas fcidump`

```bash
apex-cas fcidump --case-dir <case_dir>
```

将活性空间哈密顿量写成 FCIDUMP 格式。

### `apex-cas testcas`

```bash
apex-cas testcas FCIDUMP_PATH -M 500 [--symm su2]
```

对已生成的 FCIDUMP 做一个小规模 DMRG 测试。

## 配置

主配置模板位于：

- `shared/config/cas_settings_template.yaml`

示例配置：

- `examples/fe2s2/inputs/fe2s2_cas_settings.yaml`

这些配置控制：

- SCF 方法
- 泛函
- 基组来源
- 相对论设置
- 溶剂模型
- 收敛阈值
- 轨道构建行为

## 数据 authority

V1.0.0 的 authority chain 是：

1. 结构文件
2. `apex-cas prepare`
3. finalized `cluster_info.yaml`
4. `scf`
5. `buildcas`
6. `fcidump`

后续步骤应消费已保存状态，而不是静默重新构造新的 cluster metadata。

## Fe2S2 重跑说明

当前仓库布局中：

- `examples/fe2s2/` 是 fresh rerun 工作目录
- `examples/fe2s2_bk2/` 是本地保留的 baseline 快照

因此最新的 Fe2S2 walkthrough 默认假设：`APEX_Filter` 的 bootstrap 文件可
在 rerun 过程中重新生成，而不是必须在仓库里预先提交完整一套。
