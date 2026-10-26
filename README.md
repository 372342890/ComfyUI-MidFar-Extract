# ComfyUI MidFar Extract

通用的中景 / 远景**单个物体**提取、软边抠图及透明裁切节点。建筑、树、独立山体等按照参考图识别结果逐个处理，没有固定景物名单，不需要下载额外分割模型。旧版整层节点仍保留兼容。

## 一次运行输出三份完整中远景素材

使用根目录 `midfar-three-autonomous.byteartist.json`。安装本仓库并重启后，
粘贴导入整个 JSON，在第一个 LoadImage 换参考图，设置输出目录前缀，
点击一次“全局运行”。分析、选材、生成、检查和修正都在这一个工作流内，
不需要再次导入展开结果。所有图片节点固定使用 **Seedream 5.0 pro**，
2048×2048、PNG、关闭水印。视觉判断分成候选分析、交叉复核和逐版审查，
按阶段限制 JSON 回复长度，避免原生文本 carrier 长思考超过服务时限；
图像生成仍保留 Seedream 的提示思考设置。

流程：原图 → 多候选分析 → 原图边界硬过滤与真实局部图表 →
视觉复核三份不同完整单元 → 三个独立生成分支 →
每份初稿、两轮自动审查修正 → 选择合格 RGBA → 三张透明 PNG。
这是一次工作流任务，内部共九次图片模型调用和十一次视觉文本调用，
会比只生成三张初稿耗时更多。最终只保存三张合格素材。

保留真实相连峰、侧柱、共享岩台和自然岩基；二维重叠、相邻雪地或覆草
不能作为物体连接的证据。排除草堆、树林、湖水和地面风景条，
保留山体自身覆草覆雪。候选接触原图边界或已经裁断会被淘汰，
不能通过向内缩 bbox 恢复资格。遮挡处只重建该目标自身的短自然山脚。
所有目标和提示由当前图的原生视觉模型产生，没有按文件名指定素材。
自动兼容模型的0–1比例框和0–1000整数框，保留原始坐标证据；
混合单位的错误框仍被拒绝，归一化不会放宽源边界安全检查。

新增节点：`BAMidFarCandidateReview`、`BAMidFarThreePlans`、
`BAMidFarAutonomousReference`、`BAMidFarAssetInspect`、
`BAMidFarRepairPrompt`、`BAMidFarChooseAsset`。
三份计划是三个独立 STRING 槽，不使用 ByteArtist 不支持的动态列表映射。
自定义节点不执行模型请求，不保存密钥，也不依赖私有服务 SDK。

最终选择同时要求数值检查通过、七个语义布尔项全部为真且评分至少85。
缺峰、断底、遗漏相连部件、混入无关物体、风格不符、背景不干净、
贴画布边缘或多主体都会失败。检查阶段返回缺陷供模型纠错；
同一分支三版均未通过时明确报错，不能用不完整图凑三张。
生成模型和视觉审查仍会出错；这些检查不能保证任意输入绝对成功。
原图不足三个安全完整目标时同样明确失败。

```sh
python -m unittest discover -v
```

## Selected connected mid/far assets

For reusable natural scenery, use the selected-assets nodes:

`Current image → vision selection JSON → BAMidFarSelectionExpand → save/import exported JSON → independent cropped-reference carriers → BAMidFarSelectedMatte → PNG`

The selector takes at most 2 or 3 suitable rocks, hills or mountains. Connected pillars,
cap rocks and peaks sharing a natural base are **one complete asset**, not separate
targets. Grass tufts and scattering details are excluded in this mode; a hill keeps
its own grass cover. The older instance and terrain-generation modes remain available.

Each selection has `id`, `unit_id`, `name`, `kind` (`rock/hill/mountain`), `layer`
(`midground/background`), `location`, `appearance`, `occlusion`, normalized `bbox`,
`asset_unit` (`connected_formation/independent`), `connected_parts`, `selection_reason`,
`priority` (0–100), and `is_single:true`. Parts with the same `unit_id` are merged
before ranking; overlapping boxes alone do not merge separate objects. Available
mid/far layers are both covered before taking the next highest-priority unit.
Fewer than two suitable targets are reported instead of invented.

`BAMidFarSelectedAsset` accepts exactly one target plus the current image. Its
`reference_crop` is the corresponding bbox crop without rescaling; connect both
`prompt` and `reference_crop` to the same image editor. This avoids neighboring
large formations replacing smaller targets. Changed images require a fresh plan.
Keep the prompt node's key and filename connected to `BAMidFarSelectedMatte`.

The matte uses the observed background color, including tinted/compressed green,
magenta or blue screens, to solve soft Alpha and recover boundary colors.
It rejects a missing screen, empty subject, multiple substantial disconnected
subjects and canvas-cut output in strict mode. These numeric checks still need
visual review for semantic grouping, source fidelity and appearance.

```sh
python -m unittest test_core test_objects test_terrain test_selection -v
```

## Individual objects (legacy instance mode)

`Reference → vision LLM objects JSON → BAMidFarObjectPlan → image editor → BAMidFarObjectMatte → SaveImage`

视觉模型必须识别**实例**：两栋楼是两项，不把建筑群、整片树林、连续山脉写成一项。每项的 id 唯一，bbox 为归一化的 `[left, top, right, bottom]`：

```json
{
  "style": "原图画风、照明、透视",
  "objects": [
    {
      "id": "o01",
      "name": "左侧红砖楼",
      "layer": "midground",
      "location": "原图左侧，拱桥后方",
      "appearance": "瓦顶、三层红砖墙",
      "occlusion": "底部被近处栏杆遮挡",
      "bbox": [0.12, 0.3, 0.34, 0.8],
      "is_single": true
    }
  ]
}
```

ObjectPlan 输出 `object_prompt`、`reference`、`key_color`、`filename_prefix` 四个**对齐的动态列表**。标准 ComfyUI 自动逐项映射；**ByteArtist BALLMImage carrier 当前只执行第一项，不能直接把动态列表接入一个图片生成节点**。ByteArtist 请使用下文的 WorkflowExpand。标准 ComfyUI 连接：

- ObjectPlan.object_prompt → 图片编辑.prompt（ByteArtist API 名称 `prompt0`）
- ObjectPlan.reference → 图片编辑.image（ByteArtist API 名称 `image0`）
- 图片编辑.image → ObjectMatte.image
- ObjectPlan.key_color、filename_prefix → ObjectMatte 对应输入
- ObjectMatte.rgba → SaveImage.images
- ObjectMatte.filename_prefix → SaveImage.filename_prefix

每个物体居中、完整输出，保留原视角并补全合理的遮挡部分，不带相邻实例和地面底座。图片编辑模型必须只生成一张变体。ObjectMatte 保留 RGBA 与柔边，只裁掉空白，不改变主体像素。每张素材尺寸可不同，因此不把素材强制合为同尺寸 batch。

`layer=both` 提取中景与远景，按两层交替输出，亦可只选其中一层。`max_objects` 默认 8，范围 1–32；超过上限的数量在 `manifest.truncated` 中明确显示。兼容模型混用的 0–1 / 0–1000 位置坐标，规范化记录在 `bbox_normalized`；整片树林、天际线、水面等被剔除，并在 `rejected_objects` 保留原因。无可辨认实例则报错，不编造物体。`strict=true` 拒绝空图、多个显著分离主体或触及画布边界的结果；检测不能识别全部语义粘连，仍需目视复核。细节本来分离的物体可人工确认后关闭 strict。

遮挡补全由图片模型推断，不保证不可见纹理与真实原物体完全一致。模型质量、输入清晰度与色键冲突会影响效果。

### ByteArtist: independent carriers

`Reference → vision LLM → BAMidFarWorkflowExpand → SaveText → import generated ByteArtist JSON → run`

WorkflowExpand 根据当前图片的分析结果展开独立分支，每项一个 ObjectPlan、一个 BALLMImage、一个 ObjectMatte 和一个 SaveImage。每个 ObjectPlan 只包含该实例，不会把第一张输出重复保存为其他对象。`workflow_template` 是提供的 ByteArtist 图模板 JSON，`reference_filename` 填当前已上传参考图的文件名，`max_objects` 控制本次输出数。先运行分析工作流，导入它生成的 `workflow_json` 后运行提取。更换参考图要重新运行分析，不要沿用旧图的实例描述。

随附 `objects.expand_workflow()` 可用于离线展开已获得的真实分析计划。工作流无固定类别、景物或场景模板；分支由当前图的有效实例数量与配置上限决定。已展开的测试 JSON 仅针对各自参考图。

## Install

在 ComfyUI 的 `custom_nodes` 目录克隆此仓库，使用 **ComfyUI 实际的 Python 环境**安装依赖，然后重启。ByteArtist 可在 Manager → 通过 URL 下载 → 插件下载输入仓库 Git URL。

```sh
git clone https://github.com/372342890/ComfyUI-MidFar-Extract.git
python -m pip install -r ComfyUI-MidFar-Extract/requirements.txt
```

ComfyUI 已提供 PyTorch；本插件不重装 PyTorch。安装后应出现 `BAMidFarObjectPlan`、`BAMidFarObjectMatte`、`BAMidFarWorkflowExpand` 及旧版 `BAMidFarPlanPrompt`、`BAMidFarMatte`。

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

## Automatic three-asset workflow with target localization

Import `midfar-three-autonomous.byteartist.json` in ByteArtist, replace the
reference image, then click global run once. The fixed graph examines candidate
connections and occlusions, selects three units, localizes each visible target
with a separate vision judgment, and isolates source pixels as a generation
guide. Guides are intermediate references only, never final assets.

Every image call uses Seedream 5.0 pro. Two feedback rounds regenerate from the
original isolated guide, avoiding repeated copying of a previous wrong image.
The guide marks the occluded lower foot as a local edit region. A scoped
completion node composites only that region from the native generated image,
preserving the visible source upper geometry and background inside the workflow.
Every composite still requires fresh numeric and visual inspection; it is not
accepted merely because the upper reference geometry was retained.
Final selection still requires every numeric and visual gate to pass, including
all attached parts, a natural complete base, reference style and clean alpha.
Tiny disconnected key-colored compression noise is recorded separately; real
detached foreground parts remain blocking defects.

The graph contains 9 image calls and 14 text calls. Generation may fail its
quality gates; an unsuccessful run never delivers a rejected image as success.

## Reference-guided natural terrain assets

`Style image + shape image + brief → vision LLM assets JSON → BATerrainWorkflowExpand → import JSON → independent image carriers → RGBA PNGs`

This mode **generates** individual rocks, grassy hills, grass tufts and mountains. It is separate from strict existing-instance extraction. A hill keeps its own grass cover and soil/rock volume; a tuft keeps its connected roots. Neither receives an additional platform or landscape strip.

The scene image controls palette, texture and rendering style. A single-object view or four-view sheet controls silhouette and proportions. The generator receives the style image as image 1 and shape image as image 2. A four-view sheet is a shape reference, not the requested output; every generated PNG contains one object in one view.

**ByteArtist clipboard format:** connect both references to the **same** image aggregation handle, in style-then-shape order. The frontend serializes these as `image0` and `image1`. Do not invent a separate UI input named `image1`; it is silently omitted on import. The same rule applies to the visual planner and the image generator.

The visual planner must produce the following schema from the **current images and user brief**, with variable target names, shapes, surfaces and variations. Valid kinds: `rock`, `hill`, `grass_tuft`, `mountain`.

```json
{
  "style": "Observed palette, simplified materials, lighting and rendering style",
  "assets": [{
    "id": "a01",
    "name": "Individual grassy hill",
    "kind": "hill",
    "shape": "One low wide mound with a complete natural base",
    "surface": "Short grass on its own rounded volume",
    "variation": "Slightly asymmetric rise",
    "is_single": true
  }]
}
```

`BATerrainWorkflowExpand` takes `plan`, `workflow_template`, `style_filename`, `shape_filename`, `max_assets` and emits a ByteArtist clipboard workflow JSON plus a manifest. The template contains two LoadImage nodes (style first, shape second), one BATerrainAssetPrompt, BALLMImage, BAMidFarObjectMatte, SaveImage, and optional SaveText nodes for quality/manifest output. Each generated branch contains **exactly one** asset prompt and its own carrier. Changed references or briefs require a fresh visual plan.

`BATerrainAssetPrompt` takes one-asset JSON, `style_reference`, `shape_reference`, `key_mode`, `filename_prefix`. Outputs are `prompt`, `style_reference`, `shape_reference`, `key_color`, `filename_prefix`, `manifest`. Wire the first three to the image carrier's prompt, image 1 and image 2. Wire key/filename to BAMidFarObjectMatte. Unlike the older list node, this node explicitly rejects multiple assets; use the expander.

GLBs are **reference inputs**, not generated by this plugin. Render a GLB to an image/four-view sheet in your 3D tool and use that as the shape image; observed bounds may be included in the user brief. The plugin cannot reconstruct or promise matching 3D topology from a 2D output.

```sh
python -m unittest test_core test_objects test_terrain -v
```

34 tests cover previous extraction/matting behavior plus natural-asset validation, hill/tuft completeness, complementary-color edge recovery and separate two-reference carrier expansion. Strict checks detect empty, border-cut or separated subjects; semantic style and silhouette still need visual review.
