# ComfyUI MidFar Extract

轻量、通用的中景 / 远景分层提示编译与软边抠图节点。没有固定景物名单或远景高度限制，不需要下载额外分割模型。

## Install

在 ComfyUI 的 `custom_nodes` 目录克隆此仓库，使用 **ComfyUI 实际的 Python 环境**安装依赖，然后重启。ByteArtist 可在 Manager → 通过 URL 下载 → 插件下载输入仓库 Git URL。

```sh
git clone https://github.com/372342890/ComfyUI-MidFar-Extract.git
python -m pip install -r ComfyUI-MidFar-Extract/requirements.txt
```

ComfyUI 已提供 PyTorch；本插件不重装 PyTorch。安装后应出现 `BAMidFarPlanPrompt`、`BAMidFarMatte` 两个节点。

## Workflow

`Reference image → vision LLM scene JSON → MidFar Scene Prompt (one per layer) → image editor with original reference → MidFar Soft Matte → SaveImage`

视觉模型需输出以下结构，所有实体仅来自当前参考图。没有某层就输出空数组，不编造：

```json
{
  "foreground": ["近处实体的具体描述"],
  "midground": ["中景实体及位置、外观"],
  "background": ["远景实体及位置、外观"],
  "sky": "原天空描述",
  "style": "画风、视角和照明",
  "occlusion": "谁遮挡了哪一层，如何连续补全"
}
```

`key_mode=auto` 根据参考图色彩选择绿色、品红或蓝色中冲突最少的色键。把 `prompt` 送给支持参考图编辑的模型，同时把 **同一个** `key_color` 连到对应抠图节点。生成尺寸和比例应与参考图一致。

### Outputs and controls

- `rgba`：带真实 Alpha 的 4 通道 IMAGE，**直接连 SaveImage**，不要再次 JoinImageWithAlpha。
- `foreground_mask`：1=主体不透明，0=背景透明。与 ComfyUI LoadImage 的透明度 MASK 方向相反。
- `preview`：棋盘格检查图，便于确认软边、细枝与孔洞。
- `quality`：每张图的主体覆盖率、软边比例和基础告警；不能代替人工分层质量审查。
- `softness`：纯色底容差，默认 0.04；色底略有偏差可升至 0.06。不做固定腐蚀或膨胀。
- `despill`：解算并去除混色边缘中的色键颜色，默认开启。
- 远离不透明主体的低 Alpha 色键残留会被清除，避免深色底上出现透明布纹；不透明细枝及其软边邻域保留。

## Limits

这是色键抠图，不能从普通照片直接推断语义层次。最终分层与遮挡补全依赖所使用的视觉模型 / 图片编辑模型；被遮挡区域是生成的重建，不是原图中可恢复的真实像素。与色键完全相同的主体颜色无法可靠区分，可换另一色键重新生成。玻璃、烟雾与复杂半透明材质需要人工复核。

## Tests

```sh
python -m unittest test_core -v
```

14 项测试覆盖场景 JSON 验证、动态提示、自动选色、软 Alpha 解算、颜色恢复、细枝保留、远离主体的色键残留、质量告警、批次与输入检查。测试仅需 NumPy / OpenCV；运行节点使用 ComfyUI 的 PyTorch。

MIT License. No credentials, service endpoints, private workflows or project assets are included.
