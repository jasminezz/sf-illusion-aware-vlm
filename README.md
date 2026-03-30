# CVPR 2026 VI-Probe Competition: Optical Illusion VQA

本项目是 CVPR 2026 Workshop VI-Probe Competition 的参赛方案，任务为光学幻觉图像的二分类视觉问答。

核心方法：**针对性图像预处理 + 反幻觉Prompt + 5票多数投票集成**，覆盖 18 种经典光学幻觉类型，共 630 张测试图片。

---

## 快速开始

### 1. 环境准备

> **Python 版本要求**：3.10+

```bash
pip install pillow numpy scipy
```

### 2. 模型调用方式

本方案支持两种模型调用方式：

#### 2.1 配置模型 API

代码通过 **Anthropic Claude API** 格式调用大语言模型。在 `challenge_runner_final.py` 顶部配置以下常量：

```python
API_URL   = "your-api-url"    # API 端点
API_KEY   = "your-api-key"    # API Key
MODEL     = "your-model"      # 模型名称
NUM_VOTES = 5                 # 每张图片投票次数
```

#### 2.2 Claude Code 订阅调用

本方案通过 **Claude Code 订阅**调用模型，开启多个实例，以问答交互的方式将预处理后的图片和反幻觉 Prompt 发送至模型进行推理。每张图片调用 **5 次**，取多数投票作为最终答案。

### 3. 准备数据

确保测试数据 CSV 文件和图片目录就位：

```
inc-vsap-clos-illusion-aware-vlm/final_code/
├── test.csv          # 输入 CSV (列: index, image_path, prompt, answer)
└── test/             # 图片目录
    ├── 0.png
    ├── 1.png
    └── ...
```

CSV 格式示例：

```csv
index,image_path,prompt,answer
0,test/0.png,"Are the two vertical bands of the same color?...",1
1,test/1.png,"Are the two vertical bands of the same color?...",1
```

### 4. 运行推理

```bash
cd inc-vsap-clos-illusion-aware-vlm/final_code/

# 完整运行 (630 张图片)
python challenge_runner_final.py \
    --input-csv test.csv \
    --output-txt predictions.txt \
    --output-json model.json
```

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `--input-csv` | `test.csv` | 输入 CSV 文件路径 |
| `--output-txt` | `predictions.txt` | 输出预测文件，每行格式 `index answer` |
| `--output-json` | `model.json` | 输出模型元信息 JSON |

### 5. 查看结果

`predictions.txt` 格式（answer ∈ {0, 1, -1}，-1 表示推理失败）：

```
0 1
1 1
2 0
3 1
...
```

---

## 结果波动说明

> **每次完整运行的预测结果之间存在偏差。**

可能的原因：

| # | 原因 | 说明 |
|:-:|:-----|:-----|
| 1 | **模型推理的非确定性** | 大语言模型在相同输入下每次生成的文本可能不同，尤其是对视觉细节的描述和判断存在随机性 |
| 2 | **投票边界效应** | 采用 5 票多数投票机制，当某张图片的投票结果为 3:2 时，任何一次调用的微小波动都可能翻转最终结果 |
| 3 | **采样参数不可控** | 本方案通过 Claude Code 订阅方式调用模型，该模式下 `temperature` 等采样参数由平台侧控制，用户无法将其设置为 0 以获得确定性输出，因此每次推理结果天然存在随机波动 |
| 4 | **网络与服务端状态** | API 调用的响应时间和服务端负载可能影响超时重试逻辑，导致部分图片的有效投票数不足 5 票 |

因此，多次运行的结果一致率约在 **80%~90%** 之间。
