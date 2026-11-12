# PJ1 相机标定

智能信息处理课程项目。基于张正友平面标定方法，使用 Python 与 OpenCV 完成单目相机标定，估计内参、畸变系数及各视图外参，并通过误差分析、去畸变和对照实验评估标定结果。

**提交截止：2026年11月18日 23:59（北京时间）**

当前已完成第1至4章。4幅图共308个亚像素角点用于30 mm基础标定，复算的总体重投影RMSE为0.174462 px，与标定返回值一致；第4章已完成去畸变、局部对照和直线性验证。

## 实验数据与技术方案

| 项目 | 配置 |
|---|---|
| 数据组 | 标定板2，左相机 |
| 图像 | `Im_L_1.png` 至 `Im_L_4.png` |
| 图像尺寸 | 1024 × 576像素 |
| 方格边长 | 30mm |
| 棋盘坐标 | Z = 0，三维坐标单位为mm |
| 基础模型 | 估计径向畸变 k1、k2，固定 k3、p1、p2 及像素轴斜切为0 |
| 评价指标 | 总体与逐图重投影RMSE、直线拟合残差、参数变化 |

实验流程为：角点检测与亚像素细化、基础标定、重投影误差分析、图像去畸变、畸变模型对比、图像子集稳定性分析。内角点行列数及对应点顺序在数据处理阶段核定，后续实验使用统一配置。

## 目录结构

```text
.
├── README.md
├── report/
│   └── camera-calibration.docx
└── code/
    ├── data/                 原始标定图像
    └── results/                 角点、参数、统计表与实验图片
```

仓库中的目录与文件名保持英文，README及课程报告采用中文。

实验脚本统一放在 `code/` 下，依赖版本记录于 `code/requirements.txt`。脚本使用相对路径定位数据与输出，原始图像保持不变。报告正文包含关键代码、实验结果、相关图片及分析，完整源代码另行保留。

## 成员分工与进度

| 成员 | 模块及实验内容 | 计划脚本 | 报告章节 | 计划完成日期 |
|---|---|---|---|---|
| 卢远卓 | 数据核对、棋盘坐标构造、角点检测与细化；检测成功率、位移和耗时分析 | `detect_corners.py` | 第1章 | 11月8日 |
| 于智博 | 公共标定函数、基础内外参及畸变估计；30mm与60mm方格尺度对照 | `calibrate.py` | 第2章 | 11月9日 |
| 竺钱成 | 公共重投影误差函数；逐图、中心与外围区域残差分析 | `evaluate.py` | 第3章 | 11月10日 |
| 张帅宇 | 图像去畸变、校正前后对照与局部放大；棋盘行列直线性验证 | `undistort.py` | 第4章 | 11月11日 |
| 杨晓芳 | 无畸变、径向、径向加切向三类模型的参数、误差及校正效果对比 | `compare_models.py` | 第5章 | 11月12日至13日 |
| 郑周锐 | 三图子集与四图全量标定对比；参数稳定性分析与结果归纳 | `subset_stability.py` | 第6章 | 11月14日至15日 |

各成员负责对应模块的代码、实验与报告，按表中顺序交付。11月16日至17日完成内容复核与讲解准备。`detect_corners.py`、`calibrate.py` 和 `evaluate.py` 已实现。

## 模块接口与实验规范

- **角点数据**：保存图像名、图像尺寸、内角点行列数、方格尺寸、细化前后二维角点及对应三维点，确保点序一致。
- **标定结果**：保存内参矩阵K、畸变系数及排列顺序、逐图旋转与平移向量、所用图像和模型设置。公共标定函数支持不同图像子集及畸变配置。
- **误差计算**：重投影RMSE定义为各角点二维欧氏距离平方的均值再开平方，单位为像素；逐图及总体统计采用相同口径。
- **去畸变评价**：记录输出内参和裁剪设置，将同一批角点转换至校正后的像素坐标，再进行直线拟合与比较。
- **对照实验**：模型对比固定图像及角点，子集实验固定畸变模型。60mm方格尺寸仅用于尺度对照，正式结果使用30mm。
- **结果解释**：训练重投影误差不等同于真实标定精度；四图全量参数作为稳定性比较的参考，不视为相机真值。保留失败配置及其说明。

## 版本与交付约定

项目采用 `main` 分支顺序协作。成员基于远端最新版本开展工作，完成后同步提交代码、实验输出及对应报告章节，并记录提交号和运行方式。公共函数由后续模块复用。

报告统一维护 `report/camera-calibration.docx`，各章从新页开始。Word文件按成员顺序串行编辑，避免二进制文件合并冲突；已完成章节及前序代码予以保留。提交说明应包含成员姓名、模块或章节及主要变更。

每个模块的交付包括：

1. 可运行源代码及依赖说明。
2. 参数文件、统计结果和实验图片。
3. 已填写的报告章节，包含实验设置、关键代码、结果及分析。
4. 运行命令、输入输出位置和后续模块所需配置。

## 运行记录

各模块完成后补充实际入口、依赖版本及输出位置。

| 模块 | 运行命令 | 主要输出 |
|---|---|---|
| 角点检测 | `python code/detect_corners.py` | `code/results/corners/`：角点NPZ、4张编号图、完整统计表与实验图 |
| 基础标定 | `python code/calibrate.py` | `code/results/calibration/` 中的30 mm与60 mm标定参数 |
| 误差评估 | `python code/evaluate.py` | `code/results/evaluation/` 中的逐点残差、逐图和区域统计、RMSE核对及结果图 |
| 去畸变 | `python code/undistort.py` | `code/results/undistortion/`：去畸变图、原图/校正图对照、局部图、角点坐标和直线性统计 |
| 模型对比 | 待补充 | 各模型参数、误差和校正效果 |
| 子集稳定性 | 待补充 | 各图像组合参数、误差与变化分析 |

## 提交要求

最终提交时，将 `report/camera-calibration.docx` 按课程要求重命名为“组号-相机标定.docx”，开头列明组员及负责内容，正文包含方法、单目图像和实验结果。报告与完整代码、输入数据、实验输出及运行说明一并打包为 `组号-相机标定.zip`，由一位成员上传至Elearning。提交包不包含 `.git`、虚拟环境或临时文件。

## 第1章复现与数据交接

建议使用Python 3.12，在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/detect_corners.py
python -m unittest discover -s code -p 'test_*.py' -v
```

详细字段、数组形状、点序定义、实验设置与后续标定读取示例见 [角点数据说明](code/results/corners/README.md)。正式观测使用 `corners.npz` 中的 `image_points_refined`，保持 `image_names` 的顺序，并与 `object_points` 逐幅对应。计时数据为本机实测值，重新运行会随负载变化。

## 第3章复现与数据交接

在仓库根目录运行：

```bash
python -m pip install -r code/requirements.txt
python code/evaluate.py
```

脚本从自身目录定位输入，直接读取第1章的 [亚像素角点](code/results/corners/corners.npz)、第2章的 [30 mm 基础参数](code/results/calibration/calibration_30mm.json) 和四张原图。保持角点与标定文件中的图像及逐点顺序一致。外参平移单位为 mm，畸变系数按 OpenCV 顺序 [k1, k2, p1, p2, k3]。总体 RMSE 与标定返回值的绝对差须小于或等于 10⁻⁶ px。逐点残差为观测减投影。中心为观测像素坐标满足 256 ≤ u < 768 且 144 ≤ v < 432 的点，其余归入外围。矢量图的箭头放大 40 倍。

[误差结果目录](code/results/evaluation) 中，points.csv 给出逐点观测、投影、残差及零基棋盘行列索引，summary.csv 给出逐图与总体的 RMSE 和最大误差，regions.csv 给出逐图及汇总的区域统计，verification.json 保存参数文件路径及 RMSE 核对结果。figures/ 含逐图 RMSE 柱状图和四张残差矢量图。

### 公共误差函数

后续脚本放在 code/ 时，可直接使用公共标定与误差函数。objects、observations 与外参均按相同的视图顺序排列，支持传入三图子集。总体误差对所有角点合并计算。

```python
import numpy as np
from calibrate import calibrate
from evaluate import reprojection_errors, rmse_px

rms, K, dist, rvecs, tvecs = calibrate(objects, observations, image_size)
views = reprojection_errors(objects, observations, K, dist, rvecs, tvecs)
per_view = [rmse_px(view.distances) for view in views]
pooled = rmse_px(np.concatenate([view.distances for view in views]))
```

第4章可沿用 30 mm 基础参数中的 K 和 dist_coeffs，以及 corners.npz 的 image_points_refined 和 pattern_size，以同一批角点比较校正前后的直线性。
