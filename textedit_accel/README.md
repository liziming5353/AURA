# Qwen-Image-Edit 单卡 H800 文字编辑加速

本项目现已收窄到一个明确环境：

- 模型：`Qwen/Qwen-Image-Edit` 基础版
- GPU：单张 NVIDIA H800 80GB
- 精度：BF16，允许 TF32
- 输入：单张约 1024×1024 图像
- 任务：把一个或多个区域中的原文字替换为指定文字

不支持 `Qwen-Image-Edit-2509/2511` 的 Plus pipeline，也不再提供 FLUX 入口。这样可以避免不同 pipeline、RoPE 和 latent packing 规则造成结果不可比。

## 已实现的推理路径

```text
输入图像
  ├─ OCR/手工文字框
  │    └─ 原框 ∪ 目标文字估算框 ∪ padding
  ├─ 低分辨率 Qwen 编辑草稿（每轴缩小 4 倍）
  │    └─ Qwen VAE decoder 多层感知差异
  └─ 强制编辑 token mask
       └─ 官方 SpotEdit
            ├─ mask 内始终完整计算
            ├─ mask 外由 SpotSelector 决定复用
            └─ SpotFusion 保持边界和背景上下文
```

最终强制计算区域为：

```text
dilate(原文字框 ∪ 目标文字估算框)
∪ Qwen-VAE 语义差异区域
∪ 稀疏全局稳定 token
```

这解决了纯 SpecEdit 对小字、标点和相似字形可能漏选的问题。即使低分辨率草稿没有识别出笔画差异，OCR 区域仍不会被缓存。

## 当前完成度

可以直接运行：

- Qwen-Image-Edit 基础版加载；
- 单 H800 BF16/TF32 配置和硬件校验；
- H800 上原生 SDPA、FlashAttention 或 FlashAttention-3 后端选择；
- 多文字框和长短文字替换；
- Qwen VAE decoder `conv_in`、`mid_block`、首个 up block 感知差异；
- 官方 SpotEdit 强制 ROI 注入；
- 端到端延迟、峰值显存和 token 比例诊断；
- baseline / OCR+SpotEdit / 完整 hybrid 三路 benchmark。

尚未完成：

- SpecEdit mixed-resolution token 序列尚未接入 Qwen Transformer；
- OCR 检测模型暂时只有接口；
- 字体、字号和目标排版引擎不在本项目范围。

`dynamic_resolution.py` 已实现 mixed-resolution token 打包、位置坐标和恢复，但不能直接送入 Qwen Transformer。正式接入仍需处理 Qwen joint attention 的变长 RoPE 和 rectified-flow 噪声一致性，因此当前性能收益主要来自 SpotEdit token skip，而不是宣称复现了论文全部加速。

## 环境安装

建议 CUDA 12.x、Python 3.10+，只向进程暴露一张 H800：

```bash
export CUDA_VISIBLE_DEVICES=0

cd textedit_accel
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[qwen,test]"
```

SpotEdit 没有发布明确的软件许可证，因此不复制其源代码，运行时使用独立 checkout：

```bash
mkdir -p third_party
git clone https://github.com/Biangbiang0321/SpotEdit.git third_party/SpotEdit
git -C third_party/SpotEdit checkout 85e3fda
```

FlashAttention 是可选项。它必须与当前 PyTorch、CUDA 和 Python ABI 匹配，不能随意安装不匹配的 wheel：

```bash
# 安装与服务器环境匹配的 flash-attn wheel 后：
textedit-accel ... --attention-backend flash
```

默认 `--attention-backend auto`：检测到 `flash_attn` 时使用 `flash`，否则使用 PyTorch 原生 SDPA。H800 是 Hopper 架构，但 `--attention-backend _flash_3` 只有在环境确实安装兼容 FlashAttention-3 时才应启用。

## 单区域运行

```bash
export CUDA_VISIBLE_DEVICES=0

textedit-accel \
  --spotedit-path third_party/SpotEdit \
  --image assets/sign.png \
  --output outputs/sign-edited.png \
  --box '[120,220,600,330]' \
  --source-text 'OLD STORE' \
  --target-text '新人工智能体验中心' \
  --prompt 'Replace "OLD STORE" with "新人工智能体验中心". Preserve the original font style, perspective, lighting and background.' \
  --steps 50 \
  --draft-steps 12 \
  --roi-padding 32 \
  --attention-backend auto
```

默认模型就是 `Qwen/Qwen-Image-Edit`。如果使用本地权重：

```bash
textedit-accel --model /models/Qwen-Image-Edit ...
```

程序会拒绝非 H800 GPU。仅做兼容性调试时可以添加 `--no-strict-h800`，但仍要求 Hopper 9.x；该参数不代表其他 GPU 已经过性能验证。

## 多区域文字替换

按相同顺序重复 `--box`、`--source-text` 和 `--target-text`：

```bash
textedit-accel \
  --spotedit-path third_party/SpotEdit \
  --image assets/menu.png \
  --output outputs/menu-edited.png \
  --box '[80,120,420,210]' \
  --source-text 'COFFEE' \
  --target-text '咖啡' \
  --box '[80,250,500,340]' \
  --source-text 'DESSERT' \
  --target-text '今日甜点' \
  --prompt 'Replace "COFFEE" with "咖啡" and "DESSERT" with "今日甜点". Keep all other pixels and typography style unchanged.'
```

目标文字比原文长时，会按照字符视觉宽度扩展目标框。生产环境最好由排版模块直接通过 `TextRegion.metadata["target_box"]` 提供精确目标框。

## 针对文字尺寸的配置

### 小字、复杂中文、标点

优先保证准确率：

```bash
--steps 50 \
--draft-steps 12 \
--roi-padding 40 \
--semantic-quantile 0.70 \
--uniform-stride 6 \
--spot-threshold 0.12
```

较低的 semantic quantile 会选择更多差异 token；更大的 padding 为字形边缘、阴影和透视变形保留上下文。

### 普通招牌或海报标题

推荐平衡配置：

```bash
--steps 50 \
--draft-steps 12 \
--roi-padding 24 \
--semantic-quantile 0.82 \
--uniform-stride 8 \
--spot-threshold 0.15
```

### 大字且编辑框明确

可以先关闭草稿，测量 SpotEdit 本身：

```bash
--disable-draft --roi-padding 24
```

关闭草稿可能更快，但失去对文字框外布局变化的自动补充。不要在小字或目标文字明显变长时使用过小 padding。

## H800 执行策略

项目采用以下策略：

- 模型、VAE 和文本编码组件全部驻留单张 H800；
- 不启用 CPU offload，避免 PCIe 传输吞掉 token skip 收益；
- 使用 BF16；
- 开启 TF32 供仍使用 FP32 的矩阵运算；
- batch size 固定为 1；
- 不默认使用 `torch.compile`：SpotEdit 每步 token 数可能变化，容易触发多次重编译；
- 每次正式测量前执行 CUDA synchronize；
- 同时报告 wall-clock latency 和 `max_memory_allocated`。

如果出现 OOM，应先确认没有其他进程占用显存，再减少分辨率或关闭 VAE 特征验证（`--pixel-verifier`）。不建议第一时间启用 sequential CPU offload。

## Benchmark

下面的命令在同一进程、同一模型、同一 seed 下比较：

1. Qwen 原始完整推理；
2. OCR 强制框 + SpotEdit；
3. OCR + SpecEdit 式草稿验证 + SpotEdit。

```bash
textedit-benchmark-h800 \
  --spotedit-path third_party/SpotEdit \
  --image assets/sign.png \
  --output-dir outputs/benchmark-sign \
  --box '[120,220,600,330]' \
  --source-text 'OLD STORE' \
  --target-text '新人工智能体验中心' \
  --prompt 'Replace "OLD STORE" with "新人工智能体验中心". Preserve everything else.' \
  --steps 50 \
  --draft-steps 12 \
  --warmup 1 \
  --repeats 3
```

输出包括三组图片以及 `benchmark.json`：

```json
{
  "baseline": {
    "median_seconds": 0,
    "peak_memory_gib": 0
  },
  "spotedit_ocr": {
    "median_seconds": 0,
    "speedup": 0
  },
  "hybrid": {
    "median_seconds": 0,
    "speedup": 0
  }
}
```

不要直接引用论文中的 6× 或 10×。必须以这份 H800 benchmark 的实测 wall-clock 为准；草稿和 VAE 验证的时间都包含在 hybrid 结果中。

## Python API 与 OCR 接口

```python
from textedit_accel import (
    Box,
    EditRequest,
    HybridTextEditPipeline,
    TextRegion,
    configure_h800,
    load_qwen_image_edit,
)
from textedit_accel.backends import SpotEditBackend
from textedit_accel.ocr import OCRProvider

class MyOCR(OCRProvider):
    def detect(self, image):
        return [
            TextRegion(
                box=Box(120, 220, 600, 330),
                source_text="OLD STORE",
                confidence=0.99,
            )
        ]

runtime = configure_h800()
model = load_qwen_image_edit(runtime=runtime)
backend = SpotEditBackend(
    model,
    spotedit_path="third_party/SpotEdit",
    attention_backend=runtime.attention_backend,
)
editor = HybridTextEditPipeline(backend, ocr=MyOCR())
result = editor(EditRequest(
    image=input_image,
    prompt='Replace "OLD STORE" with "新人工智能体验中心"',
    target_text="新人工智能体验中心",
))
```

CLI 默认启用 Qwen VAE 感知验证。自行构建 Python API 时，可以组合 `QwenLowResolutionDraftGenerator`、`SemanticVerifier` 和 `QwenVAEFeatureExtractor`，方式参见 `cli.py`。Qwen 专用草稿适配器会同时降低生成流和条件图像流的 token 分辨率；只给原始 Diffusers pipeline 传较小的 `width/height` 并不能做到这一点。

## 测试

不下载模型的单元测试：

```bash
pip install -e ".[test]"
pytest
ruff check src tests
```

测试覆盖：

- 长短文字目标框估算；
- pixel mask 到 Qwen token grid 的保守映射；
- mixed-resolution token 打包和恢复；
- OCR 与语义 mask 合并；
- Qwen VAE 三层特征提取；
- 多文字框 CLI；
- 注意力后端选择。

真实评测还应报告 OCR exact match、CER/WER、编辑框内 LPIPS、编辑框外 PSNR/SSIM，并按照字号、字符数和编辑面积分桶。

## 参考

- [SpotEdit paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Qin_SpotEdit_Selective_Region_Editing_in_Diffusion_Transformers_CVPR_2026_paper.pdf)
- [SpotEdit official implementation](https://github.com/Biangbiang0321/SpotEdit)
- [SpecEdit image-editing paper](https://arxiv.org/abs/2605.02152)
