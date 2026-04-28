
# FFT-DI High-Performance Solver

### 基于 PyTorch 的高性能瑞利-索末菲衍射计算框架

**FFT-DI Solver** 是一款专为计算光学、全息显示及精密测量领域设计的**高性能数值传播计算引擎**。本项目基于标量衍射理论，实现了**第一类瑞利-索末菲衍射积分 (Rayleigh-Sommerfeld Diffraction Integral)** 的全自动精确求解。

传统的衍射计算往往面临“精度”与“速度”的权衡难题，且缺乏灵活的场探测手段。本项目采用了先进的 **FFT-DI (Direct Integration via FFT)** 算法，结合现代 **GPU 加速技术**与**多模式探测架构**，成功构建了一套科研级的光场仿真框架。

### 核心设计理念

本项目旨在解决传统光学计算中的核心痛点，提供了一套经过严格工程封装的解决方案：

-   **极致的计算速度**：  
    利用 **PyTorch** 的底层 CUDA 优化，将复杂的卷积积分由 CPU 串行计算转为 GPU 大规模并行计算。在处理高分辨率（如 4K/8K）全息图时，计算效率较传统 MATLAB/NumPy 方案提升数个数量级，实现了近实时的光场重建。
    
-   **多维度的场探测能力 (New)**：  
    打破了单一平面的计算限制，内置灵活的探测器模式：
    
    -   **平面探测 (Plane)**：获取任意 $z$ 处的完整 2D 复振幅分布。
        
    -   **体积扫描 (Volume)**：支持轴向快速切片扫描，直接生成 3D 光场体数据，用于焦深分析或粒子定位。
        
    -   **线型轮廓 (Line)**：针对光束整形等场景，快速提取特定截面的光强/相位分布。
        
-   **科研级的计算精度**：  
    摒弃粗糙的矩形求和，内置 **Simpson's 1/3 Rule (辛普森积分法则)**，显著提升了数值积分精度。结合严格的 **$2N-1$ 零填充策略**，有效抑制了离散化带来的边缘混叠效应。
    
-   **智能物理自检引擎**：  
    程序内置了基于 **奈奎斯特采样定理 (Nyquist Sampling Theorem)** 的卫语句（Guard Clauses）。自动检测输入场是否满足二次相位因子采样极限和最大衍射角限制。对于违背物理规律的参数（如欠采样），程序会拒绝计算并提供修正建议，确保输出结果的物理有效性。
    
-   **零代码交互体验**：  
    实现了计算核心与用户配置的完全解耦。用户无需编写代码，仅需编辑 `examples/config.ini` 或自定义 INI 配置文件，即可切换探测模式、调整波长距离等参数，并可在 Windows、Linux、macOS 上通过命令行运行。
    

### 适用场景

本工具广泛适用于以下光学仿真与工程领域：

-   **全息显示 (Holographic Display)**：大尺寸、高带宽全息图的快速重建与像质评估。
    
-   **三维光场分析**：激光焦点区域的 3D 形貌扫描与焦深 (DOF) 测量。
    
-   **无透镜成像 (Lensless Imaging)**：Gabor 同轴全息或离轴全息的数字重聚焦。
    
-   **衍射光学元件 (DOE)**：相位掩模版的设计验证与远场光斑整形分析。
## 2.核心功能

本系统采用模块化架构设计，围绕“高精度衍射”这一核心，实现了从源面数据解析、传播计算到多维场探测的全流程闭环。

### **2.1 源面定义与物理接入 (Source Definition)**

系统具备强大的外部数据兼容性，支持直接接入复杂的实验测量数据或第三方仿真数据。

-   **智能 MAT 数据解析**:
    
    -   原生支持 **MATLAB (.mat)** 格式文件的直接加载。
        
    -   **自适应元数据提取**: 能够自动识别并提取复振幅分布 (`u0`)、工作波长 (`lambda`) 以及物理尺寸信息。
        
    -   **坐标轴自动校准**: 能够解析 `x_vec`/`y_vec` 空间坐标向量，自动计算并校验采样间隔 (`dx`) 的均匀性，确保空间网格的物理严谨性。
        
-   **物理合规性预检 (Physics Pre-check)**:
    
    -   在计算启动前，系统会自动扫描源面参数。根据**奈奎斯特-香农采样定理**，结合传播距离 $z$ 和波长 $\lambda$，自动判定源面采样率是否满足无混叠传播条件。
        
    -   **严格模式 (Strict Mode)**: 用户可配置是否开启严格检查，对不满足 1/4 截止频率限制的输入源进行阻断，防止产生错误的数值干涉条纹。
        

### **2.2 多维探测器系统 (Multi-Mode Detectors)**

系统将传播计算与场探测解耦，内置了三种标准的探测模式，以满足不同维度的光场分析需求。

|探测器模式 (Mode)|描述 (Description)|典型应用场景 (Application)|
|--|--|--|
|**Plane Detecto**| 平面探测 (2D)计算指定距离 $z$ 处的完整二维复振幅分布 $U(x,y)$。 |全息再现像观察、成像焦面分析、DOE 远场光斑检测。|
|**Volume Detector**|体积探测 (3D)在 $[z_{start}, z_{end}]$ 范围内按指定步长进行轴向切片扫描，自动堆叠生成三维光场数据矩阵。|粒子场全息三维重建、焦深 (DOF) 测量、焦散面 (Caustic) 分析。|
|**Line Detector**|线型探测 (1D)仅提取目标平面中心线的复振幅分布，极大降低存储开销。|激光光束截面轮廓分析 (Beam Profiling)、轴向对准检测。|

### **2.3 数据输出与序列化 (Output & Serialization)**

计算结果采用标准化的科学计算格式存储，便于后续的可视化处理与二次开发。

-   **NPZ 归档格式**:
    
    -   所有输出均保存为 Python 标准的 **NumPy Compressed (.npz)** 格式，兼顾读写速度与存储空间。
        
-   **完整元数据封装**:
    
    -   输出文件不仅包含计算得到的场数据（`u_out` 或 `volume`），还自动封装了对应的物理上下文信息：
        
        -   **物理坐标**: 采样间隔 `dx`、传播距离 `z` (或 `z_vec`)。
            
        -   **物理属性**: 工作波长 `wavelength`。
            
-   **兼容性**:
    
    -   生成的 NPZ 文件可直接被 Python (`numpy`, `matplotlib`) 读取绘图，或通过 `scipy.io` 转换为 MATLAB 格式，实现跨平台数据交互。|

## 3.算法原理
**FFT-DI（基于快速傅里叶变换的直接积分法）** 是一种用于快速、精确计算 **瑞利-索末菲（Rayleigh-Sommerfeld, RS）衍射积分** 的数值算法。
### 1. 物理本质：卷积 (Convolution)
点光源在空间中产生的球面波分布$g(x,y,z) = \frac{1}{2\pi} \frac{e^{jkr}}{r} (\frac{1}{r}-jk) \frac{z}{r}$,
$r = \sqrt{x^2+y^2+z^2}$.
瑞利-索末菲衍射公式在数学上可以看作是 **“输入光场 $U_{in}$”** 与 **“脉冲响应函数（核函数） $g$”** 的卷积：



$$
\begin{aligned}
U(x, y, z) &= \iint_{A} U(\varsigma, \eta, 0) g(x-\varsigma, y-\eta, z) \, d\varsigma \, d\eta \\
&= \iint_{A} U(\varsigma, \eta, 0) \frac{\exp(jkr)}{2\pi r} \frac{z}{r} \left( \frac{1}{r} - jk \right) \, d\varsigma \, d\eta
\end{aligned}
$$
其中，$r = \sqrt{（x-\varsigma）^2+（y-\eta）^2+z^2}$.
它正是Rayleigh -Sommerfeld衍射积分公式，可用于近场和远场，无需任何近似。在大多数情况下，方程中的衍射积分式必须采用直接数值积分计算。在孔径平面上，将U采样为N$\times$N个等距网格。对于观测平面$x_m，y_n，z$上的一点，积分可以通过数值积分计算为黎曼和：
$$ U(x_m, y_n, z) = \sum_{i=1}^{N} \sum_{j=1}^{N} U(\varsigma_i, \eta_j, 0) g \times (x_m - \varsigma_i, y_n - \eta_j, z) \Delta \varsigma \Delta \eta $$
其中，$\Delta \varsigma$和$\Delta \eta$是孔径平面上的采样间隔。

### 2. 数学技巧：基于快速傅里叶变换的直接积分法 (Fast-Fourier-Transform Based Direct Integration Method)
直接在空间域计算上述积分（直接积分法，DI），计算复杂度是 $O(N^4)$，非常慢。  
FFT-DI 利用卷积定理：**时域/空域的卷积 = 频域的乘积**。离散卷积可以计算为：
$$ S = \text{IFFT2} [ \text{FFT2}(U) \cdot \times \text{FFT2}(H) ] \Delta \varsigma \Delta \eta $$
**关键步骤 —— 补零 (Zero Padding)：**

-   **问题**：直接用 FFT 计算的是 **循环卷积 (Circular Convolution)**，而物理上的光传播是 **线性卷积 (Linear Convolution)**。如果直接算，光会从左边“卷”到右边，导致混叠错误。
    
-   **解决**：必须将输入矩阵 $N \times N$ 进行 **补零扩充**，通常扩充到 **$2N-1$** 或更大。
    
    -   将 $U_{in}$ 放在大矩阵的角落，其余补 0。
        
    -   计算出对应大尺寸的核函数 $g$。

$$ U = \begin{bmatrix} U_0 & \mathbf{0} \\ \mathbf{0} & \mathbf{0} \end{bmatrix}_{(2N-1) \times (2N-1)} $$
$$ H = \begin{bmatrix} g(X_1, Y_1, z) & \dots & g(X_1, Y_{2N-1}, z) \\ \vdots & \ddots & \vdots \\ g(X_{2N-1}, Y_1, z) & \dots & g(X_{2N-1}, Y_{2N-1}, z) \end{bmatrix}_{(2N-1) \times (2N-1)} $$
$$ X_j = \begin{cases} x_1 - \varsigma_{N+1-j} & j = 1, \dots, N-1 \\ x_{j-N+1} - \varsigma_1 & j = N, \dots, 2N-1 \end{cases} $$
$$ Y_j = \begin{cases} y_1 - \eta_{N+1-j} & j = 1, \dots, N-1 \\ y_{j-N+1} - \eta_1 & j = N, \dots, 2N-1 \end{cases} $$


        
-   **结果**：FFT 计算后，裁剪出中心有效区域，即得到精确的线性卷积结果。
### 3. 精度优化：辛普森积分 (Simpson's Rule)

为了进一步减小数值离散化带来的误差，FFT-DI 通常结合 **辛普森积分规则**：


$$ U = \begin{bmatrix} W \cdot \times U_0 & \mathbf{O} \\ \mathbf{O} & \mathbf{O} \end{bmatrix}_{(2N-1) \times (2N-1)} $$

$$ W = B^{T} B $$

$$ B = \frac{1}{3} [1 \quad 4 \quad 2 \quad 4 \quad 2 \quad \dots \quad 2 \quad 4 \quad 1] $$


    
-   这使得数值积分的精度从矩形法的 $O(\Delta x^2)$ 提升到了 $O(\Delta x^4)$。
    

### 4. 适用范围与优势

-   **全空间精确**：与角谱法（ASM）不同，FFT-DI 不受旁轴近似限制，也不受“计算窗口太小导致光跑出去”的截断误差影响。只要采样率满足奈奎斯特准则，它在近场和远场都是准确的。
    
-   **采样要求**：要求采样间隔 $\Delta x$ 足够小，能够分辨出核函数 $g$ 的相位振荡（即 $dx \le \lambda/2$ 或满足文中提到的 $\delta_{min}/2$）。
    
-   **计算效率**：计算复杂度降低为 $O((2N)^2 \log (2N))$，比传统直接积分快几个数量级。
    

### 总结

FFT-DI 是 **“运用 FFT 方法代替直接积分”**。算出来的结果和一个个像素去积分（线性卷积）是几乎一样的，
但是通过补零和 FFT ，可以计算速度飞快。它是处理**大尺寸衍射**问题的高效准确算法。





## 4.项目结构

开源版工程已经整理为标准 Python 包结构，便于安装、测试和发布到 GitHub。原始研究说明保留为中文版文档，英文 `README.md` 作为仓库主页。

```text
FFT-DI/
├── README.md                 # 英文 GitHub 主页
├── README.zh-CN.md           # 中文说明文档
├── LICENSE                   # MIT 开源协议
├── CITATION.cff              # 引用信息
├── pyproject.toml            # Python 包配置与命令行入口
├── requirements.txt          # 基础依赖：numpy, scipy
├── docs/
│   └── algorithm.md          # FFT-DI 算法说明
├── examples/
│   ├── config.ini            # 示例运行配置
│   └── generate_sample.py    # 生成示例 .mat 输入数据
├── src/
│   └── fft_di/
│       ├── __init__.py
│       ├── __main__.py       # 支持 python -m fft_di
│       ├── cli.py            # 命令行运行入口
│       ├── io.py             # MAT/NPZ 数据读写辅助
│       ├── propagator.py     # FFT-DI 核心传播算法
│       └── sampling.py       # 采样条件检查
└── tests/
    ├── test_propagator.py    # 传播核心测试
    └── test_sampling.py      # 采样检查测试
```

运行示例时会生成以下本地文件，但它们已被 `.gitignore` 排除，不建议提交到 GitHub：

```text
examples/sample_input.mat
results/*.npz
__pycache__/
.venv/
```

## 5.系统需求

**操作系统**：Windows、Linux、macOS 均可运行。当前开源版不依赖 Windows 批处理脚本。

**Python 版本**：推荐 Python 3.9 或更高版本。

**基础依赖**：

- `numpy`
- `scipy`

**可选依赖**：

- `torch`：用于 PyTorch 张量和 GPU 工作流。不开启 PyTorch 后端时无需安装。
- `pytest`：仅开发测试时需要，也可以直接使用 Python 标准库的 `unittest`。

**硬件建议**：

- CPU 模式可直接运行小到中等规模网格。
- 大尺寸网格会占用较多内存，因为 FFT-DI 需要构造 `(2N-1) × (2N-1)` 的补零场和传播核。
- 若使用 PyTorch/CUDA 后端，可利用 NVIDIA GPU 加速大规模 FFT 计算。

## 6.使用方法

开源版提供两种使用方式：命令行配置运行，以及在 Python 代码中直接调用 API。默认后端为 NumPy/SciPy，安装后即可运行基础示例；PyTorch 后端为可选项。

## ✨ 特性 (Features)

-   **多模式仿真**：支持 Plane（单平面）、Volume（体积/切片扫描）、Line（中心线轮廓）三种输出模式。

-   **可选 GPU 后端**：默认使用 NumPy，安装 PyTorch 后可在配置中切换为 `torch` 后端。

-   **严格验证**：内置 Nyquist 采样定理检查与 Delta-rho 积分核采样检查，防止数值伪影。

-   **灵活输入**：支持 MATLAB (`.mat`) 格式输入，自动解析复振幅、波长和空间采样信息。

-   **Simpson 积分**：支持高精度的 Simpson 积分规则（要求输入矩阵 N 为奇数）。

## ⚙️ 安装 (Installation)

在仓库根目录执行：

```bash
python -m pip install -e .
```

如果需要 PyTorch 后端：

```bash
python -m pip install -e ".[torch]"
```

如果需要运行测试：

```bash
python -m pip install -e ".[dev]"
```

## 🚀 快速运行示例 (Quick Start)

1.  **生成示例输入数据**：

```bash
python examples/generate_sample.py
```

该命令会在 `examples/` 目录下生成 `sample_input.mat`。

2.  **运行 FFT-DI 传播**：

```bash
python -m fft_di examples/config.ini
```

或者在安装后使用命令行入口：

```bash
fft-di examples/config.ini
```

3.  **查看输出结果**：

程序会生成 `results/sample_output.npz`。该文件为 NumPy 压缩格式，可通过 `numpy.load()` 读取。

## ⚙️ 配置说明 (Configuration)

开源版使用 INI 格式配置文件，示例位于 `examples/config.ini`。

### 输入输出配置

|参数名|说明|示例|
|--|--|--|
|`input_mat`|输入 `.mat` 文件路径，相对路径以配置文件所在目录为基准|`sample_input.mat`|
|`output_npz`|输出 `.npz` 文件路径|`../results/sample_output.npz`|
|`field_key`|输入复振幅矩阵变量名|`u0`|
|`wavelength_keys`|波长变量名，可写多个候选名|`lambda,wavelength`|

### 运行模式

|模式|参数值|必需参数|描述|
|--|--|--|--|
|Plane|`plane`|`target_z_m`|计算特定距离处的 2D 复数波场。|
|Volume|`volume`|`z_start_m`, `z_end_m`, `z_steps`|计算一系列 z 切片，生成 3D 数据堆栈。|
|Line|`line`|`target_z_m`|计算特定距离处中心行的 1D 复数轮廓。|

### 物理参数与校验

|参数名|类型|默认/建议|说明|
|--|--|--|--|
|`backend`|String|`numpy`|计算后端，可选 `numpy` 或 `torch`。|
|`device`|String|`auto`|PyTorch 后端设备，可选 `auto`, `cpu`, `cuda` 等。|
|`target_z_m`|Float|-|目标传播距离，单位为米，用于 plane/line 模式。|
|`theta_max_deg`|Float|`0.0`|最大衍射角限制，用于检查 Nyquist 条件。|
|`rho_mode`|String|`corner`|积分半径模式，可选 `corner`, `edge`, `custom`。|
|`rho_custom_m`|Float/auto|`auto`|自定义积分半径，仅 `rho_mode=custom` 时使用。|
|`strict_checks`|Bool|`false` 或 `true`|是否在采样不满足条件时抛出错误并停止运行。|
|`use_simpson`|Bool|`true`|是否使用 Simpson 积分规则，要求输入矩阵 N 为奇数。|

**配置示例 (`examples/config.ini`)：**

```ini
[io]
input_mat = sample_input.mat
output_npz = ../results/sample_output.npz
field_key = u0
wavelength_keys = lambda,wavelength
x_key = x
y_key = y

[run]
mode = plane
backend = numpy
device = auto
target_z_m = 0.02
use_simpson = true

[validation]
strict_checks = false
theta_max_deg = 0.0
rho_mode = corner
rho_custom_m = auto
```

## 📥 输入数据格式 (Input Format)

程序接受 MATLAB `.mat` 格式文件。文件至少需要包含以下变量：

1.  **`u0`**：二维复数矩阵 `(N, N)`，表示初始波场。如果 `use_simpson=true`，N 必须为奇数。

2.  **`lambda` 或 `wavelength`**：标量，波长，单位为米。

3.  **空间坐标或采样信息**，以下方式任选一种：

    -   提供向量 **`x`** 和 **`y`**（推荐）。

    -   或提供标量 **`dx`**（采样间隔）。

    -   或提供标量 **`Lx`**、`LX`、`size_x`、`x_size` 中的任意一种作为物理尺寸。

## 🧩 Python API 示例

```python
import numpy as np

from fft_di import fft_di_propagate

n = 257
dx = 2.0e-6
wavelength = 532e-9
z = 0.02

x = (np.arange(n) - (n - 1) / 2) * dx
X, Y = np.meshgrid(x, x, indexing="xy")
u0 = np.exp(-(X**2 + Y**2) / (80e-6)**2)

u1 = fft_di_propagate(u0, dx, wavelength, z, use_simpson=True)
intensity = np.abs(u1) ** 2
```

## 📊 输出结果说明 (Output)

输出文件 (`.npz`) 可使用 `numpy.load()` 读取。包含的键值取决于运行模式：

```python
import numpy as np

data = np.load("results/sample_output.npz")

# 通用元数据
print(data["dx"])          # 采样间隔
print(data["wavelength"])  # 波长

# Mode: plane
u_out = data["u_out"]      # 传播后的 2D 复数场

# Mode: volume
volume = data["volume"]    # (Steps, N, N) 3D 复数数组
z_vec = data["z_vec"]      # Z 轴坐标向量

# Mode: line
profile = data["line_profile"]  # 1D 复数数组
```
## 7.基本测试
对于(5991*5991)的时域有限差分法生成的相复振幅数据(如光源示例1所示)进行试验。原算法7张RTX509032G联算用时大于一小时，基于FFT-DI的计算方法单张卡
计算用时1分02秒

![项目截图](/mark1.png)
<p align="center">光源示例1

通过对计算结果进行对比验证，结果显示光强分布在单位量级下具有高度一致性，
最大绝对误差仅为0.04。相位分布表现出极高的仿真精度，其最大偏差(约0.5rad)
仅出现在光强趋于零的奇异区域。此类相位误差主要归因于低信噪比区域下复振幅
实部与虚部极小值导致的数值计算不稳定性，在有效物理孔径内，模型展现了较高的数值保真度。(如测试结果1所示)’

![项目截图](/mark2.png)
<p align="center">测试结果1

另一组测试数据如光源示例2与测试结果2展示。

![项目截图](/mark3.png)
<p align="center">光源示例2


![项目截图](/mark4.png)
<p align="center">测试结果2

## 8.常见问题
-   **Error: Simpson sampling requires odd N**
    
    -   原因：启用了 `use_simpson = true`，但输入矩阵 `u0` 的尺寸是偶数。
        
    -   解决：在预处理中裁剪/填充矩阵至奇数尺寸，或在配置中设置 `use_simpson = false`。
        
-   **Error: Sampling too coarse for RS kernel**
    
    -   原因：传播距离 `z` 较小或物理尺寸较大，导致采样间隔 `dx` 不满足 Rayleigh-Sommerfeld 积分核的振荡要求。
        
    -   解决：减小 `dx` (增加分辨率) 或增加传播距离。
        
-   **Warning: Nyquist condition violated**
    
    -   原因：衍射角度过大，超过了当前采样率支持的限制。
        
    -   解决：该警告不影响运行，但结果可能包含混叠误差。
