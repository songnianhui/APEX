# APEX 项目缩写词典

> 整理自 APEX 代码库与参考文献（2017/2019/2026），涵盖电子结构方法、张量网络、轨道概念、自旋投影、基组泛函、相对论效应、FCIDUMP 格式、分子模拟及软件工具等领域。

---

## 一、电子结构方法

| 缩写 | 英文全称 | 中文全称 | 物理含义 |
|---|---|---|---|
| **HF** | Hartree-Fock | Hartree-Fock 方法 | 平均场近似下的多电子波函数方法，忽略电子关联 |
| **DFT** | Density Functional Theory | 密度泛函理论 | 以电子密度为基本变量处理多体问题的方法 |
| **UHF** | Unrestricted Hartree-Fock | 非限制性 Hartree-Fock | α 和 β 轨道独立优化的 HF 方法，可描述自旋极化 |
| **UKS** | Unrestricted Kohn-Sham | 非限制性 Kohn-Sham | α 和 β 轨道独立的 DFT 方法，用于开壳层体系 |
| **RKS** | Restricted Kohn-Sham | 限制性 Kohn-Sham | α = β 轨道的 DFT 方法，适用于闭壳层体系 |
| **SCF** | Self-Consistent Field | 自洽场 | HF/DFT 中迭代求解 Fock 方程直至收敛的过程 |
| **CAS** | Complete Active Space | 完全活性空间 | 选定一组轨道和电子做全 CI 展开，是 multireference 方法的基础 |
| **CASSCF** | CAS Self-Consistent Field | 完全活性空间自洽场 | 活性空间内全 CI + 活性空间外轨道优化 |
| **CASCI** | CAS Configuration Interaction | 完全活性空间组态相互作用 | 在给定轨道下对活性空间做全 CI，不优化轨道 |
| **RASSCF** | Restricted Active Space SCF | 限制性活性空间 SCF | 将活性空间分为 RAS1/RAS2/RAS3 子空间，限制激发等级以降低计算量 |
| **CASPT2** | CAS Second-Order Perturbation Theory | 完全活性空间二阶微扰理论 | 在 CASSCF 基础上用二阶微扰补充动态关联能 |
| **NEVPT2** | N-Electron Valence State PT2 | N 电子价态二阶微扰理论 | 类似 CASPT2 但无 intruder state 问题，更稳定 |
| **FCI** | Full Configuration Interaction | 全组态相互作用 | 在给定基组下的精确解（Hilbert 空间内全部 Slater 行列式） |
| **CCSD** | Coupled Cluster Singles & Doubles | 耦合簇单双激发 | 以指数算符 e^(T₁+T₂) 参数化波函数的高精度方法 |
| **CCSD(T)** | CCSD with perturbative Triples | 耦合簇单双+微扰三激发 | "化学精度的黄金标准"，在 CCSD 基础上微扰处理三重激发 |
| **CCSDT** | CC with Singles, Doubles, Triples | 耦合簇单双三激发 | 迭代处理到三重激发，比 CCSD(T) 更精确但更昂贵 |
| **CCSDTQ** | CC with Singles through Quadruples | 耦合簇至四重激发 | 包含四重激发的耦合簇，接近 FCI |
| **CCSDTQP** | CC with Singles through Quintuples | 耦合簇至五重激发 | 包含五重激发，几乎等于 FCI |
| **UCCSD** | Unrestricted CCSD | 非限制性耦合簇单双激发 | 基于 UHF 参考的 CCSD，可处理破对称解 |
| **UCCSD(T)** | Unrestricted CCSD(T) | 非限制性 CCSD(T) | 基于 UHF 参考的 CCSD(T) |

---

## 二、DMRG / 张量网络方法

| 缩写 | 英文全称 | 中文全称 | 物理含义 |
|---|---|---|---|
| **DMRG** | Density Matrix Renormalization Group | 密度矩阵重整化群 | 将多体波函数表示为 MPS、逐格点变分优化的方法，适用于一维强关联体系 |
| **UDMRG** | Unrestricted DMRG | 非限制性 DMRG | 基于 UHF/UKS 破对称参考轨道的 DMRG 计算 |
| **SA-DMRG** | State-Averaged DMRG | 态平均 DMRG | 对多个自旋态做等权平均优化的 DMRG |
| **MPS** | Matrix Product State | 矩阵乘积态 | DMRG 的波函数表示形式，Ψ = Σ Tr(A¹⋯Aᴷ) |σ₁⋯σₖ⟩ |
| **MPO** | Matrix Product Operator | 矩阵乘积算符 | Hamiltonian 等算符的张量网络表示，与 MPS 配合使用 |
| **spMPS** | Spin-Projected MPS | 自旋投影 MPS | 对破对称 MPS 做自旋投影以消除自旋污染的方法 |
| **SODS** | Spatial Orbital Direct Spin | 空间轨道直旋格式 | 每个空间轨道对应一个格点，含 α 和 β 两个自旋轨道的 DMRG 编码方式 |
| **SHCI** | Semi-stochastic Heat-bath CI | 半随机热浴组态相互作用 | 随机选取重要行列式 + 确定性变分优化的混合 CI 方法 |

---

## 三、轨道与电子结构概念

| 缩写 | 英文全称 | 中文全称 | 物理含义 |
|---|---|---|---|
| **AO** | Atomic Orbital | 原子轨道 | 以原子为中心的基函数（如 Gaussian），是 MO 的基本构成单元 |
| **MO** | Molecular Orbital | 分子轨道 | AO 的线性组合，描述电子在整个分子中的分布 |
| **UNO** | Unrestricted Natural Orbital | 非限制性自然轨道 | 由 UHF/UKS 总密度矩阵对角化得到的轨道，其占据数（NOON）偏离 0 或 2 的轨道即具有多参考特征 |
| **LUO** | Localized Unrestricted Orbital | 定域化非限制性轨道 | 将 α 和 β 轨道分别独立定域化后的轨道，保留自旋极化信息 |
| **NOON** | Natural Orbital Occupation Number | 自然轨道占据数 | 自然轨道上的电子占据数，闭壳层为 0 或 2，开壳层/关联效应使其偏离整数值 |
| **RDM** | Reduced Density Matrix | 约化密度矩阵 | 对全波函数求迹后得到的少体密度矩阵，1-RDM 用于 UNO，2-RDM 用于能量计算 |
| **BS** | Broken Symmetry | 破坏对称性（态） | 反铁磁耦合体系中，通过初猜打破自旋对称性获得的近似态（如 α↑β↓ 排列） |
| **HS** | High Spin | 高自旋（态） | 所有局域自旋平行排列（铁磁排列）的态，S = S_max |
| **AFM** | Antiferromagnetic | 反铁磁（耦合） | 相邻自旋反平行排列的磁相互作用模式，交换耦合常数 J < 0 |
| **PM** | Pipek-Mezey | Pipek-Mezey 定域化 | 最大化 Mulliken 原子布居平方和的轨道定域化方法，保留 σ/π 分离 |
| **Boys** | Boys localization | Boys 定域化 | 最小化轨道质心间距离平方和的定域化方法，产生空间紧凑的轨道 |
| **SCDM** | Selected Columns of Density Matrix | 密度矩阵选定列方法 | 通过选取密度矩阵的列构建正交轨道的方法，常用于虚轨道定域化 |
| **AVAS** | Automated Valence Active Space | 自动价层活性空间 | 通过投影到目标 AO 子空间自动选取活性轨道的方法 |
| **DIIS** | Direct Inversion in Iterative Subspace | 迭代子空间直接求逆 | SCF 收敛加速算法，通过外推消除残差振荡 |
| **SOSCF** | Second-Order SCF | 二阶 SCF | 利用 Fock 矩阵对轨道旋转的 Hessian 做牛顿步，用于 SCF 收敛的最后阶段 |
| **FNO** | Frozen Natural Orbital | 冻结自然轨道 | 基于 MP2/CCSD 密度矩阵截断虚轨道空间的技术，降低 CC 计算成本 |
| **Fiedler** | Fiedler Vector | Fiedler 向量 | 图拉普拉斯矩阵第二小特征值对应的特征向量，用于优化 DMRG 轨道排序 |
| **GA** | Genetic Algorithm | 遗传算法 | 通过选择、交叉、变异优化轨道排列顺序以最小化 DMRG 纠缠熵 |
| **T₁** | T₁ diagnostic | T₁ 诊断量 | CCSD 中单激发振幅的 RMS 值，用于判断单参考方法的可靠性（>0.02 表明多参考特征显著） |

---

## 四、自旋投影相关

| 缩写 | 英文全称 | 中文全称 | 物理含义 |
|---|---|---|---|
| **VAP** | Variation-After-Projection | 投影后变分 | 先定义投影空间，再在投影后的空间中做变分优化，能量严格变分（下界） |
| **PAV** | Project-After-Variation | 变分后投影 | 先优化 MPS，再投影到目标自旋态，非变分（可能高估能量） |
| **Sz** | Spin z-projection quantum number | 自旋 z 分量量子数 | 总自旋沿 z 轴的投影值 Mₛ，如 Sz = 0, ±1/2, ±1, ... |
| **J** | Exchange coupling constant | 交换耦合常数 | Heisenberg 模型 Ĥ = -2J·S₁·S₂ 中的参数，J<0 为反铁磁，J>0 为铁磁 |

---

## 五、基组与泛函

| 缩写 | 英文全称 | 中文全称 | 物理含义 |
|---|---|---|---|
| **def2-SVP** | def2 Split-Valence Polarization | def2 分裂价层极化基组 | Ahlrichs 组的双 ζ + 极化基组，适用于常规 DFT 计算 |
| **def2-TZVP** | def2 Triple-Zeta Valence Polarization | def2 三 ζ 价层极化基组 | Ahlrichs 组的三 ζ + 极化基组，精度更高 |
| **TZP** | Triple-Zeta Polarization | 三 ζ 极化基组 | 三 ζ 加极化基组（Jorge 2009 TZP-DKH 为其 DKH 优化版本，属 contracted Gaussian 基组） |
| **TZP-DKH** | TZP for Douglas-Kroll-Hess | DKH 专用三 ζ 极化基组 | 专为 DKH 标量相对论计算优化的基组，必须与 sf-X2C/DKH 配合使用 |
| **B3LYP** | Becke 3-parameter Lee-Yang-Parr | B3LYP 杂化泛函 | 20% HF 交换 + 80% B88 交换 + LYP 相关的杂化泛函 |
| **BP86** | Becke-Perdew 1986 | BP86 泛函 | 纯 GGA 泛函（B88 交换 + P86 相关），常用于 UKS 产生 UNO |
| **TPSSh** | TPSS hybrid | TPSSh 杂化泛函 | 含 10% HF 交换的 meta-GGA 杂化泛函 |

---

## 六、相对论与溶剂化

| 缩写 | 英文全称 | 中文全称 | 物理含义 |
|---|---|---|---|
| **sf-X2C** | Spin-Free Exact Two-Component | 自旋自由精确二分量 | 将四分量 Dirac 方程精确约化为二分量，仅保留标量相对论效应（质量-速度 + Darwin） |
| **X2C** | Exact Two-Component | 精确二分量 | 完整的 X2C 方法，包含自旋-轨道耦合 |
| **DKH** | Douglas-Kroll-Hess | Douglas-Kroll-Hess | 通过幺正变换逐阶消去 Dirac 方程中正负能态耦合的标量相对论方法 |
| **ddCOSMO** | Domain Decomposition COSMO | 区域分解 COSMO | COSMO 隐式溶剂模型的区域分解实现，计算效率更高 |
| **COSMO** | Conductor-like Screening Model | 类导体屏蔽模型 | 将溶剂视为连续介电体的隐式溶剂化模型 |

---

## 七、FCIDUMP 文件格式

| 缩写 | 英文全称 | 中文全称 | 物理含义 |
|---|---|---|---|
| **FCIDUMP** | Full CI Dump | 全 CI 积分文件 | 存储活性空间内一电子/二电子积分和核排斥能的标准文件格式 |
| **NORB** | Number of Orbitals | 轨道数 | FCIDUMP 头部参数，活性空间中的轨道数 |
| **NELEC** | Number of Electrons | 电子数 | FCIDUMP 头部参数，活性空间中的电子数 |
| **MS2** | 2 × Spin Projection | 二倍自旋投影 | FCIDUMP 头部参数，2Mₛ 值 |
| **ORBSYM** | Orbital Symmetry | 轨道对称性 | FCIDUMP 头部参数，各轨道的不可约表示编号 |
| **ECORE** | Core Energy | 芯能量 | 核排斥能 + 冻结芯层电子的贡献；DMRG 社区约定写 0，真实值存附属文件 |

---

## 八、生物化学与分子模拟

| 缩写 | 英文全称 | 中文全称 | 物理含义 |
|---|---|---|---|
| **FeMoco** | Iron-Molybdenum Cofactor | 铁钼辅因子 | 固氮酶活性中心的 [Mo₇Fe₉S₉C(homocitrate)] 簇 |
| **LLDUC** | Li-Li-Dattani-Umrigar-Chan | LLDUC 模型 | Chan 组 2019 年发表的 FeMoco 计算模型 |
| **QM/MM** | Quantum Mechanics / Molecular Mechanics | 量子力学/分子力学 | 高精度 QM 区域 + 经典力场 MM 区域的多尺度模拟方法 |
| **MD** | Molecular Dynamics | 分子动力学 | 数值积分牛顿方程模拟原子运动的分子模拟方法 |
| **NPT** | Constant N, P, T ensemble | 等温等压系综 | 粒子数、压力、温度恒定的统计力学系综 |
| **NVT** | Constant N, V, T ensemble | 等温等容系综 | 粒子数、体积、温度恒定的统计力学系综 |
| **TIP3P** | Transferable Intermolecular Potential 3-Point | 三点转移型分子间势 | 水分子的三点经验势场模型 |
| **UFF** | Universal Force Field | 通用力场 | 适用于全元素周期的经验分子力场 |
| **GFN2-xTB** | Geometry, Frequency, Noncovalent, v2, extended Tight Binding | GFN2 扩展紧束缚方法 | Grimme 组开发的半经验量子化学方法，用于快速几何优化和频率计算 |

---

## 九、软件与文件格式

| 缩写 | 英文全称 | 中文全称 | 说明 |
|---|---|---|---|
| **PySCF** | Python Simulations of Chemistry Framework | Python 化学模拟框架 | APEX 的核心量子化学引擎，提供 HF/DFT/CASSCF/积分变换等功能 |
| **Block** | Block DMRG code | Block DMRG 程序 | Chan 组开发的 DMRG 计算程序 |
| **Block2** | Block2 (next-gen DMRG) | Block2 下一代 DMRG | 支持 SU(2) 对称性的新一代 DMRG 程序 |
| **ASE** | Atomic Simulation Environment | 原子模拟环境 | Python 分子模拟工具包，用于解析结构文件 |
| **VESTA** | Visualization for Electronic and Structural Analysis | 电子与结构分析可视化 | 3D 晶体/轨道可视化软件 |
| **XYZ** | Cartesian coordinate format | 笛卡尔坐标格式 | 存储分子原子坐标的标准文本格式 |
| **HDF5** | Hierarchical Data Format 5 | 层次数据格式第5版 | 高性能科学数据存储格式，PySCF 的 chkfile 格式 |
| **YAML** | YAML Ain't Markup Language | YAML 配置语言 | APEX 中所有配置文件的格式 |
