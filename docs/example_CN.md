[English](example.md) | 中文

# Fe2S2 氧化态完整复现指南

本指南复现已在本仓库中验证通过的氧化态 `Fe2S2(SCH3)4^{2-}` benchmark。

范围：
- `APEX_CAS`：`prepare -> scf -> buildcas -> fcidump -> testcas`
- `APEX_Filter`：`load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt -> dmrg-basis -> dmrg -> extrapolate -> report`

如果希望避免覆盖已提交的 benchmark 产物，请在一个全新的 clone 或 `examples/fe2s2/` 的副本中运行以下命令。

## 0. 所需文件

开始之前，请确认以下文件存在：

- `examples/fe2s2/inputs/fe2s2.xyz`
- `examples/fe2s2/inputs/fe2s2_cas_settings.yaml`
- `examples/fe2s2/inputs/fe2s2_filter_settings.yaml`
- `examples/fe2s2/inputs/fe2s2_cluster_info.yaml`
- `examples/fe2s2/filter_session/method_controls.yaml`

已提交的仓库中已包含这些文件。如果你是从仅含结构文件的新目录开始，请先使用 `apex-cas prepare` 重新生成 cluster 标注文件。

## 1. 准备 Cluster 元数据

`APEX_CAS prepare` 是生成或验证 `cluster_info.yaml` 的权威方式。

命令：

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

生成的产物：

- `examples/fe2s2/inputs/fe2s2_cluster_info_draft.csv`
- `examples/fe2s2/inputs/fe2s2_structure_labeled.png`
- `examples/fe2s2/inputs/fe2s2_cluster_info.yaml`（最终确认后生成）

需要确认或编辑的内容：

- 审查 draft CSV 中的标注
- 验证金属标签、桥原子和端基配体
- 将最终确认的 `cluster_info.yaml` 作为后续整个工作流的权威依据

如果在编辑 draft CSV 后需要重新生成权威文件，运行：

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml \
  --finalize
```

## 2. 运行 APEX_CAS SCF

命令：

```bash
apex-cas scf examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

生成的产物：

- `examples/fe2s2/outputs/scf/C4H12Fe2S6_uks_BP86_tzp-dkh.chk`
- `examples/fe2s2/outputs/scf/C4H12Fe2S6_uks_BP86_tzp-dkh_scf_info.json`
- `examples/fe2s2/outputs/scf/C4H12Fe2S6_uks_BP86_tzp-dkh_cas_info.json`

需要确认：

- SCF 收敛
- 目标电荷和自旋
- 高自旋 UHF/UKS 参考在化学上是合理的

## 3. 构建 Active Space

命令：

```bash
apex-cas buildcas examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

生成的产物：

- `examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_orbital_report.md`
- `examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_selection.txt`
- `examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_noon_plot.png`
- `examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_cas_data.h5`

需要确认或编辑：

- 检查 orbital 报告和 NOON 图
- 确保 active orbitals 在化学上是合理的
- 如有需要，在生成 FCIDUMP 之前编辑 `*_selection.txt`

对于本 benchmark，已提交的 selection 与已验证的运行结果一致。

## 4. 生成 FCIDUMP

命令：

```bash
apex-cas fcidump --case-dir examples/fe2s2
```

生成的产物：

- `examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh`
- `examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh.ecore`
- `examples/fe2s2/outputs/fcidump/C4H12Fe2S6_uks_BP86_tzp-dkh_fcidump_info.json`

需要确认：

- active space 为 `(20o,30e)`
- `ECORE` sidecar 文件存在
- FCIDUMP 文件名主干是下游 filter session 所使用的那个

## 5. 可选：在 APEX_CAS 中进行 DMRG 冒烟测试

命令：

```bash
apex-cas testcas examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh -M 500
```

可选 SU2 版本：

```bash
apex-cas testcas examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh -M 500 --symm su2
```

生成的产物：

- `examples/fe2s2/outputs/fcidump/dmrg/C4H12Fe2S6_uks_BP86_tzp-dkh_dmrg_info.json`
- `examples/fe2s2/outputs/fcidump/dmrg/C4H12Fe2S6_uks_BP86_tzp-dkh_dmrg_results.h5`
- `examples/fe2s2/outputs/fcidump/dmrg/C4H12Fe2S6_uks_BP86_tzp-dkh_noon_plot.png`

需要确认：

- DMRG 冒烟测试能量合理
- NOON 分布合理

## 6. 加载 Filter Session

命令：

```bash
apex-filter load \
  --config examples/fe2s2/inputs/fe2s2_filter_settings.yaml \
  --session examples/fe2s2/filter_session
```

生成的产物：

- `examples/fe2s2/filter_session/session.json`
- `examples/fe2s2/filter_session/step1_load/cas_arrays.npz`
- `examples/fe2s2/filter_session/step1_load/cas_meta.json`
- `examples/fe2s2/filter_session/step1_load/cluster_info.json`
- `examples/fe2s2/filter_session/step1_load/fcidump_ref.json`
- `examples/fe2s2/filter_session/step1_load/settings.json`（扁平的 step-1 bootstrap settings 快照）
- `examples/fe2s2/filter_session/method_controls.yaml`

需要确认或编辑：

- 验证解析后的 FCIDUMP 路径
- 验证 `cluster_info.yaml` 已被正确读取
- 在执行下游步骤之前检查 `filter_session/method_controls.yaml`

对于本 benchmark，`filter_settings.yaml` 仅是 step-1 的引导文件。从 step 2 开始的数值控制参数存放在 `method_controls.yaml` 中。

## 7. 枚举电子组态

命令：

```bash
apex-filter enumerate --session examples/fe2s2/filter_session
```

生成的产物：

- `examples/fe2s2/filter_session/step2_enumerate/enumeration.json`
- `examples/fe2s2/filter_session/step2_enumerate/enumeration_layers.json`
- `examples/fe2s2/filter_session/step2_enumerate/selection_candidates.csv`
- `examples/fe2s2/filter_session/step2_enumerate/selection_worklist.csv`
- `examples/fe2s2/filter_session/step2_enumerate/selection_guide.md`

需要确认或编辑：

- 确认氧化态 Fe2S2 族栈正确
- 在镜像态验证后，保留一个代表态用于主 benchmark ladder
- 在本指南中，代表态标签为 `Fe1↑Fe2↓|2xFe(III)|d:none`

枚举控制参数来自 `examples/fe2s2/filter_session/method_controls.yaml`。如果需要更改枚举的规模，调整其中的 `enumerate` 部分，然后重新运行 `apex-filter enumerate`。

## 8. 运行 UHF

命令：

```bash
apex-filter uhf \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step2_enumerate/selection_worklist.csv"
```

生成的产物：

- `examples/fe2s2/filter_session/step3_uhf/uhf_summary.json`
- `examples/fe2s2/filter_session/step3_uhf/results/*_uhf.npz`
- `examples/fe2s2/filter_session/step3_uhf/results/*_uhf.h5`
- `examples/fe2s2/filter_session/step3_uhf/results/*_post_scf_observables.json`

需要确认：

- `converged = true`
- `E_total` 与 `Fe2S2` 氧化态 benchmark 表一致
- `s2`、`two_s` 以及 Fe 位点自旋观测量合理

## 9. 运行 UCCSD

命令：

```bash
apex-filter ccsd \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step3_uhf/selection_worklist.csv"
```

生成的产物：

- `examples/fe2s2/filter_session/step4_ccsd/ccsd_summary.json`
- `examples/fe2s2/filter_session/step4_ccsd/scripts/*_ccsd_results.npz`
- `examples/fe2s2/filter_session/step4_ccsd/scripts/*_post_scf_observables.json`

需要确认：

- `E_total` 在 Chan Table 5 的 benchmark 容差范围内
- `s2`、`two_s`、`two_sz_fe1`、`two_sz_fe2` 已写入摘要和 sidecar JSON

## 10. 运行 UCCSD(T)

命令：

```bash
apex-filter ccsd-t \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step4_ccsd/selection_worklist.csv"
```

生成的产物：

- `examples/fe2s2/filter_session/step5_ccsd_t/ccsd_t_summary.json`
- `examples/fe2s2/filter_session/step5_ccsd_t/scripts/*_ccsd_t_results.npz`
- `examples/fe2s2/filter_session/step5_ccsd_t/scripts/*_post_scf_observables.json`

需要确认：

- 能量和自旋观测量仍在与 Chan 相同的 benchmark 范围内

## 11. 运行 UCCSDT

在此步骤之前，如需设置 HAST-UCC 环境：

```bash
export PYTHONPATH=/Users/snh/hast-ucc:$PYTHONPATH
```

命令：

```bash
apex-filter ccsdt \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step5_ccsd_t/selection_worklist.csv"
```

生成的产物：

- `examples/fe2s2/filter_session/step6_ccsdt/ccsdt_summary.json`
- `examples/fe2s2/filter_session/step6_ccsdt/scripts/*_ccsdt_results.npz`
- `examples/fe2s2/filter_session/step6_ccsdt/scripts/*_ccsdt_results.h5`
- `examples/fe2s2/filter_session/step6_ccsdt/scripts/*_post_scf_observables.json`

需要确认：

- `energy` 接近 Chan Table 5 的 `UCCSDT` 值
- `observables_complete = true`
- `lambda_converged = true`
- `.h5` checkpoint 文件存在，以便在需要时可以重启 observable 阶段

## 12. 准备 DMRG 基组

命令：

```bash
apex-filter dmrg-basis \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step6_ccsdt/selection_worklist.csv"
```

生成的产物：

- `examples/fe2s2/filter_session/step7_dmrg_basis/dmrg_basis_summary.json`
- `examples/fe2s2/filter_session/step7_dmrg_basis/dmrg_basis_qc.json`
- `examples/fe2s2/filter_session/step7_dmrg_basis/dmrg_basis_qc.csv`
- `examples/fe2s2/filter_session/step7_dmrg_basis/results/*_dmrg_basis.npz`
- `examples/fe2s2/filter_session/step7_dmrg_basis/results/*_dmrg_basis.h5`

需要确认：

- basis QA 指标健康
- basis 标签和排序稳定
- benchmark 运行仅保留代表态

已提交的 `method_controls.yaml` 已包含此阶段经过验证的 Fe2S2 氧化态 benchmark 设置。

## 13. 运行 DMRG

命令：

```bash
apex-filter dmrg \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step7_dmrg_basis/selection_worklist.csv"
```

生成的产物：

- `examples/fe2s2/filter_session/step8_dmrg/dmrg_summary.json`
- `examples/fe2s2/filter_session/step8_dmrg/results/*_dmrg.npz`
- `examples/fe2s2/filter_session/step8_dmrg/results/*_dmrg.h5`
- `examples/fe2s2/filter_session/step8_dmrg/results/*_dmrg.log`
- `examples/fe2s2/filter_session/step8_dmrg/results/*_scratch`

需要确认：

- bond dimension ladder 是经过验证的 Fe2S2 集合
- `M=100..2400` 的能量平滑下降
- `M=2000` 和 `M=2400` 已收敛
- `.h5` 文件包含调度信息、诊断数据和 `2pdm`

对于本 benchmark，DMRG 步骤用作能量 benchmark ladder。此处有意不要求自旋分辨的观测量。

## 14. 外推至无穷 DMRG Bond Dimension

命令：

```bash
apex-filter extrapolate --session examples/fe2s2/filter_session
```

生成的产物：

- `examples/fe2s2/filter_session/step9_extrapolate/dmrg_extrapolation_summary.json`

需要确认：

- 拟合使用了已收敛的 `M=100..2400` ladder
- 外推值接近 Chan 参考值

## 15. 生成最终报告

命令：

```bash
apex-filter report --session examples/fe2s2/filter_session
```

生成的产物：

- `examples/fe2s2/filter_session/step10_report/final_summary.json`
- `examples/fe2s2/filter_session/step10_report/final_report_energies.csv`
- `examples/fe2s2/filter_session/step10_report/final_report_observables.csv`

需要确认：

- 最终排名存在
- `final_summary.json` 和 `final_report_energies.csv` 中包含
  `CCSDT + DMRG consensus` 信息

主线 `report` 步骤只负责计算结果汇总；面向 benchmark 的详细 compare 报告属于独
立验证流程，不属于 production mainline。

## 16. 可选：高阶分支

这些步骤不是复现 Fe2S2 氧化态主 benchmark 所必需的，但在 session 中仍然可用：

- `apex-filter fno-uccsdtq`
- `apex-filter cc-composite`

对于当前的 benchmark 复现指南，可以在 `report` 之后停止。

## 17. 参考文件

当前保留的验证侧 compare 报告位于：

- [examples/fe2s2/fe2s2_rerun_compare_report_20260503.md](/Users/snh/Projects/APEX/examples/fe2s2/fe2s2_rerun_compare_report_20260503.md)

如果只需要 Chan-facing benchmark 资产，请直接查看：

- `examples/fe2s2/chan_ref/`
