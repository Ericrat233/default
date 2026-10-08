# OpenPOD 0.1 — 超声/无损检测可靠性软件

双语言、可复现、可审计的 POD 统计和超声前处理研究软件。目标基准是用户指定的 **mh1823 7.3.0**；本次已执行的外部参考是第三方公开仓库中的 **7.3.7 标准源码选定函数**和 R survival/GLM。尚未取得 7.3.0 安装包，不能把结果宣称为该版本的完整替代验收。

**交付状态：可运行的研究实现与验证包；不是已经获得工业资格认证、完成全部版本兼容验收的软件。** MATLAB 文件采用 R2014b–R2020a 范围语法，核心不依赖附加 Toolbox；已运行 GNU Octave 9.4.0 数值自测，MATLAB 各老版本实机测试待完成。Python 3.12 已运行 32 项测试。

## 快速运行

```bash
python -m pip install -e '.[validation]'
python -m pytest -q
python examples/run_demo.py
openpod fit data/generated/censored.csv --threshold 2 --missing mar --out signal.json --plot signal.png
openpod fit data/generated/hitmiss.csv --kind hitmiss --link logit --out hitmiss.json
openpod preprocess data/generated/bscan_raw.npy --out processed.npz
openpod preprocess data/generated/cscan_raw.npy --mode field --out cscan.npz
```

离线安装需要提前准备 numpy、scipy、matplotlib 的 wheels；软件正常分析不访问网络。示例、原始数据、参考数值和图已经随包提供，不必重新生成才能查看。

MATLAB：在本目录打开 MATLAB，运行：

```matlab
addpath('matlab');
result = op_selftest();
run('examples/run_matlab_demo.m');  % 另生成图，需要可用的图形渲染器
```

`op_selftest` 将结果写入 `validation/matlab/`，无需图形界面和附加工具箱。MATLAB 输出 `results.mat` 和 CSV，兼顾 R2014b 没有 JSON 编码接口的情况。

## 交付内容

| 位置 | 内容 |
|---|---|
| `docs/研究与算法说明.md` | POD 方法体系、似然/区间推导、近五年代表性研究、架构和伪代码 |
| `docs/使用与数据接口.md` | 数据字段、单位、索引、参数与双语言示例 |
| `docs/功能覆盖与验收.md` | 功能逐项覆盖、已验证/待验证范围、7.3.0 迁移验收流程 |
| `docs/验证报告.md` | R/参考函数/Python/MATLAB 接口对比、CFAR、覆盖率、局限 |
| `openpod/` | Python 核心、前处理、噪声、MAPOD/贝叶斯/层级/GP、诊断、CLI |
| `matlab/` | 对应统计和前处理函数、MAPOD/贝叶斯/层级/GP、历史网格、自测 |
| `data/generated/` | 可复现 A/B/C 扫与体数据、混合删失、hit/miss、分组数据和效果图 |
| `data/reference/` | 标准示例 1–4；保留来源说明与 Artistic License 2.0 |
| `validation/` | CSV/JSON 数值报告、实际运行日志、R 与跨语言复核脚本 |
| `tests/` | 32 项实际运行的 Python 自动测试 |
| `.github/workflows/validate.yml` | GitHub Python + Octave 核心检查流程 |

## 核心能力

- â–a 高斯 MLE，x/y 四种线性/对数组合；精确、左/右/区间删失；显式 MAR 响应缺失策略。
- hit/miss：logit、probit、cloglog、loglog；Fisher 或观测信息协方差；LR、联合域、单侧剖面和 Wald 检出限。
- a50、a90、a90/95、可配置 a80/90 等；重复测量兼容调整、聚类 bootstrap、分组模型比较。
- Gaussian/lognormal/Weibull/exponential/最大极值噪声；删失 MLE、PFA、SNR、dB 和阈值权衡。
- A/B 扫与含时间轴体数据；C 扫/仿真标量场单独处理；去噪、背景、SG、基线/标定、包络、对齐、CFAR 与图示。
- 模型辅助 POD：实验校准传递函数、模型差异项、混杂参数积分、固定核超参数 GP 代理。
- 高斯随机截距模型；精确共轭 Bayes；删失数据多链 Metropolis、split-Rhat/ESS。

这些是**方法级实现**；不复制 mh1823 Windows/TclTk 菜单、R 全局对象或 Workshop Addendum。`compat.py` 和 `op_legacy_band.m` 提供历史数值轮廓复核，而不是二进制/对象接口兼容。

## 关键统计约定

1. MAPOD 是 **Model-Assisted POD**。删失数据似然是独立模块；“缺失”不能默认当成“漏检”。MNAR 或缺失尺寸需要额外可识别模型，本包不自动臆造。
2. signal 的默认检出限使用尺寸方向 Delta/Wald；hit/miss 默认保留旧程序的 LR95、df=1 约定。单侧剖面使用 `z(.95)^2`，数值不同；切勿统一称作同一种 95% 区间。
3. 幅值、噪声、阈值必须用相同的特征和量纲。默认不进行逐条峰值归一化；这种归一化会消除幅值与缺陷尺寸的关系。
4. CFAR 的理论 PFA 建立在独立指数分布功率单元上。相关超声 RF、包络、图像不能直接沿用理论保证；须在独立背景上复核。
5. 本包拒绝完全分离/零残差/秩亏等没有正常 MLE 的输入。POD 上限、field-finds、尺寸测量误差不能强行套用标准模型。

## 许可证与可追溯性

新实现采用 MIT。参考示例与原包材料归各原作者所有，按 `licenses/ARTISTIC-2.0.txt` 保留原包声明。未打包原程序源码、R 安装包或用户上传的手册。来源、版本、SHA256 见 `docs/来源与版本.md` 和 `manifest.json`。研究方法的实现是独立实现；不表示逐篇论文已完整复现。
