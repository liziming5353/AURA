# TextEdit Accel

面向扩散模型图像文字替换的推理加速实验项目。实现以下组合方案：

1. **OCR 强制路由**：原文字框与估算后的目标文字框取并集，框内 token 永不被缓存或降分辨率。
2. **SpecEdit 式 draft-and-verify**：先生成低分辨率编辑草稿，通过多尺度感知差异找出可能变化的 token。
3. **SpotEdit 选择性计算**：将前两步的强制编辑 mask 注入官方 SpotEdit，其他区域继续使用 SpotSelector/SpotFusion。
4. **动态分辨率核心**：提供细粒度 token 与粗粒度 token 的打包、坐标和恢复操作，供后续 Qwen/FLUX Transformer 原生适配。

> 当前默认可运行路径是“语义草稿 + OCR guardrail + 官方 SpotEdit token skip”。`dynamic_resolution.py`
> 已实现与模型无关的 mixed-resolution token 算法，但没有冒充已经完成的 Qwen/FLUX 内核：
> 两个模型的 RoPE、joint attention、scheduler/noise 对齐均需单独适配。在完成该适配前，论文中的
> SpecEdit 数倍动态分辨率收益不能视为已经复现。

## 为什么采用组合方案

纯 SpotEdit 对小范围编辑较稳，但加速通常约 1.6–2 倍。纯 SpecEdit 使用低分辨率草稿决定高分辨率区域，小字、标点和相似字符容易漏选。本项目把 OCR 区域设为不可跳过的硬约束，再用语义草稿补充 OCR 框之外的布局变化，最后交给 SpotEdit 保持背景和边界一致性。

目标文字长度变化时，项目会根据字符视觉宽度估算目标框，并使用：

```text
强制编辑区域 = dilate(原文字框 ∪ 目标文字估算框) ∪ 语义差异区域 ∪ 稀疏全局 token
```

OCR 暂时只有接口；调用方可直接传框，因此不阻塞模型实验。

## 安装

Python 3.10+、PyTorch 2.2+。建议独立环境：

```bash
cd textedit_accel
pip install -e ".[qwen,test]"

# SpotEdit 未发布软件许可证，因此本项目不复制其代码，运行时使用独立 checkout。
git clone https://github.com/Biangbiang0321/SpotEdit.git third_party/SpotEdit
git -C third_party/SpotEdit checkout 85e3fda
```

参考实现验证时使用的 SpotEdit 提交为 `85e3fda`。模型和显存要求参照 SpotEdit 上游说明；1024×1024 的 Qwen/FLUX 推理通常需要高显存 GPU。

## 快速运行

OCR 尚未接入时，通过 `--box '[x0,y0,x1,y1]'` 提供原文字框：

```bash
textedit-accel \
  --model Qwen/Qwen-Image-Edit \
  --spotedit-path third_party/SpotEdit \
  --family qwen \
  --image assets/sign.jpg \
  --output outputs/sign-edited.png \
  --box '[120,220,600,330]' \
  --source-text 'OLD STORE' \
  --target-text '新人工智能体验中心' \
  --prompt 'Replace "OLD STORE" with "新人工智能体验中心", preserve font style and background' \
  --steps 50 \
  --draft-steps 12
```

若要只测 OCR guardrail + SpotEdit：

```bash
textedit-accel ... --disable-draft
```

Qwen-Image-Edit-2509/2511 使用不同的 Diffusers pipeline，应指定 `--family qwen_plus`。这与基础版 `Qwen/Qwen-Image-Edit` 不可混用。

## Python API 与 OCR 接口

```python
from textedit_accel import Box, EditRequest, HybridTextEditPipeline, TextRegion
from textedit_accel.backends import SpotEditBackend
from textedit_accel.ocr import OCRProvider

class MyOCR(OCRProvider):
    def detect(self, image):
        # 可接 PaddleOCR、TextSnake 或自有服务。
        return [
            TextRegion(
                box=Box(120, 220, 600, 330),
                source_text="OLD STORE",
                confidence=0.99,
            )
        ]

backend = SpotEditBackend(
    diffusers_pipeline,
    spotedit_path="third_party/SpotEdit",
    family="qwen",
)
editor = HybridTextEditPipeline(backend, ocr=MyOCR())
result = editor(EditRequest(
    image=input_image,
    prompt='Replace the sign text with "新人工智能体验中心"',
    target_text="新人工智能体验中心",
))
result.image.save("output.png")
```

也可以通过 `TextRegion.metadata["target_box"]` 提供排版引擎计算出的精确目标框，从而跳过长度启发式估计。

## SpecEdit 部分实现

### Preliminary draft

`DiffusersDraftGenerator` 将图像每个轴缩小 4 倍，即将空间 token 数量降低约 16 倍，然后运行较短的完整去噪轨迹。草稿只参与区域验证，不作为正式轨迹的初始 latent。

### Semantic verification

`SemanticVerifier` 对原图和草稿计算归一化多尺度特征距离，再投影到模型 token 网格。默认实现不依赖特定 VAE；传入 `feature_extractor` 后可替换为论文使用的 VAE decoder 中间特征：

```python
verifier = SemanticVerifier(config, feature_extractor=my_vae_feature_extractor)
```

阈值由正差异 token 的分位数确定，并对 mask 膨胀；此外加入稀疏均匀 token，避免全局几何漂移。

### Mixed-resolution token

`dynamic_resolution.py` 实现：

- 语义区域所在 coarse block 展开为全部 fine tokens；
- 非编辑 block 以均值聚合成一个 coarse token；
- 输出 `(y, x, scale)` 坐标供模型构造位置编码；
- Transformer 更新后恢复完整 token 网格。

示例：

```python
from textedit_accel.dynamic_resolution import build_plan, pack_tokens, restore_tokens

plan = build_plan(edit_mask, coarse_factor=2, uniform_stride=8)
packed = pack_tokens(tokens_bhwc, plan)
updated = transformer_adapter(packed.tokens, packed.coordinates)
packed.tokens = updated
full_tokens = restore_tokens(packed)
```

模型适配器必须确保 coarse/fine token 使用正确的 RoPE 坐标，并在 rectified-flow 每一步保持噪声方差一致，不能简单把变长序列传给原始 Qwen/FLUX Transformer。

## 配置建议

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `roi_padding` | 24 px | 文字框外高分辨率上下文 |
| `draft_downsample` | 4/轴 | 草稿约 16× 空间 token 降低 |
| `draft_steps` | 12 | 草稿去噪步数 |
| `discrepancy_quantile` | 0.82 | 语义差异阈值 |
| `dilation_radius` | 1 token | 扩张语义区域 |
| `uniform_stride` | 8 tokens | 全局稳定采样 |
| SpotEdit `reuse_mode` | `velocity` | 减少硬粘贴边界 |

小字、中文复杂字形和标点建议增大 `roi_padding`，并由 OCR/排版引擎提供目标框。编辑面积较大时 SpotEdit 会自然退化为接近完整计算，这是正确的保质量行为。

## 测试

```bash
cd textedit_accel
pip install -e ".[test]"
pytest
```

单元测试不下载大模型，覆盖区域扩张、mask 投影、动态分辨率打包/恢复及组合流水线。真实模型评测还应报告：

- OCR exact match、CER/WER；
- 编辑框内 LPIPS；
- 编辑框外 PSNR/SSIM；
- 字号、字符数量、编辑面积分桶后的 wall-clock latency；
- 峰值显存和实际 token reuse ratio。

## 已知边界

- 上游 SpotEdit 当前没有明确软件许可证，因此只做运行时桥接。
- Diffusers 内部 API 变化可能影响 SpotEdit；请固定 SpotEdit 提交和已验证的 Diffusers 版本。
- 默认图像多尺度差异是可移植近似；严格论文复现应实现对应模型的 VAE decoder feature extractor。
- 当前没有 OCR 模型，也不负责字体识别和目标文字排版。
- 多个文字框时，`target_text` 应传与框顺序一致的字符串列表。

## 论文与代码

- [SpotEdit paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Qin_SpotEdit_Selective_Region_Editing_in_Diffusion_Transformers_CVPR_2026_paper.pdf)
- [SpotEdit official implementation](https://github.com/Biangbiang0321/SpotEdit)
- [SpecEdit: Training-Free Acceleration for Diffusion based Image Editing via Semantic Locking](https://arxiv.org/abs/2605.02152)
