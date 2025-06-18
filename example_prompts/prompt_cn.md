现在再来明确一下prompt应该规范api输出的内容：对于直接生成图像的任务，我需要让图像中尽可能只有这个物体占据主体，这个物体应该是一些完整的物品或者人物头像或上半身，整张图应该尽可能突出该人物或者物体，并且占据35%以上的构图内容，及其表明材质、纹理、以及反光，光线应该尽可能具有自然的变化，应当突出光线在物体或者人物上的表现；对于relight图像编辑任务我需要只让光照、环境、阴影发生改变，而核心物体必须保持其原本位置，并且保留其形状与纹理反光材质；对于视频我需要尽可能保持中心主体内容的形状纹理材质位置，相机视角可以进行适当移动和缩放，整个画面中应当突出的是光线的变化，例如延时摄影这种。

### 背景
Editing the illumination in images is a fundamental task in deep learning and image editing. Classic
computer graphics methods often model the appearance of images using physical illumination models.
More recently, large diffusion-based image generators have introduced unique applications and
flexible paradigms in this area, handling a wider range of “in-the-wild” lighting effects beyond simply
changing the distribution of light sources, e.g., generating backlighting or rim light, adding special
effects like glow, glare, or the Tyndall effect, simulating shadows cast through tree shade or venetian
blinds, and even manipulating human-drawn, composed, artistic, or non-photorealistic lighting
conditions. These applications also provide tools for artists and designers to modify the foreground
or background (e.g., product images, commercial posters, etc.) while maintaining harmonious
illumination. These illumination editing applications with generative image models hold unique
industrial value for visual content creation and manipulation.

（来自 IC-Light Intro)

我们现在希望基于已有的重光照工作上进行视频重光照的任务：

视频重光照旨在对已有视频施加新的光照条件，实现照片级真实且时序稳定的光照效果。然而，这一任务极具挑战：算法需确保结果的光照变化真实可信，同时跨帧保持一致。传统方法通常依赖昂贵且特殊的数据和模型。逐帧独立地应用图像重光照模型也会导致严重问题：不同帧间光源效果不一致、目标外观跳变，从而引发视频闪烁伪影。

- **项目需求：**
    - **多条件输入**：支持以多种形式提供目标光照信息，例如环境光照的全景HDR图、目标背景图像等，灵活指定希望的视频光照效果。此外模型将考虑遮挡信息，使生成结果符合真实光影遮挡规律。
    - **通用场景适用**：不同于以往聚焦人脸或特定主体的重光照方法，我们面向**通用场景视频**（室内外、**多物体**、不限定主体类型），这将显著拓宽重光照技术在视频编辑、电影后期中的应用范围。
    - **时序一致性**：通过模型设计和训练策略，最大程度保证视频帧间光照变化的平滑与一致，避免闪烁和不稳定。在条件扩散模型中显式引入时序建模模块和约束，解决逐帧生成导致的不连续问题。
    - **功能一统**：同时兼顾单图和视频

Video relighting requires an accurate simulation of light
transport through complex materials across both time and
space. Image-based relighting [12] allows a portrait to be
relit when the subject has been recorded from a dense array
of lighting directions, such as a one-light-at-a-time (OLAT)
reflectance field. Using a high-speed camera and a synchronized LED stage, Wenger et al. [76] record OLATs at movie
frame rates, allowing for cinematic relighting of a facial performance. However, the technique relies on complex equipment and cannot be generalized to new subjects.（来自 Lux Post Facto Intro)

任务需求：

因此除了使用极少数的OLAT数据集之外，我们希望创建更多In the wild 数据集来提升模型的训练质量，我们现在希望通过已有的图像生成或者视频生成模型来产生以下类型的数据：

1. 来自不同光照角度但是保持主体物体不变的图像
    1. 先产生特定的prompt用图像生成模型生成带有明确光照信息与主体物体信息的图片
    2. 然后使用图像编辑模型用prompt引导使得保持主体物体信息不动并改变环境以及光源信息（这些信息可以后续通过模型重新提取）
    3. 预期产出：1) prompt 图像对， 2) 多光照情况图像
2. 通过prompt来产生光源和环境变化情况下的视频
    1. （I2V模型方案）先对给定具有明确光照信息与主体物体信息的图片通过特定的prompt用I2V视频生成模型生成光源和环境变化情况下的视频
    2. （T2V模型方案）直接通过特定的prompt用I2V视频生成模型生成光源和环境变化情况下的视频
    3. 预期产出：prompt 光照变化视频对

现在我需要你产生的prompt满足以下条件

我们的数据集不需要非常花哨奇怪的光彩，我们的任务是创造一个更加通用自然的数据集，比如说车灯扫过，时间变化，房间内环境光源的变化，窗口光源的映射下主要物体的光照变化这种，可能更倾向于**“自然的光照舞台”**这种类型，最主要的主角应该是物体！它应该占据图片的35%以上区域

我们需要关注的不是环境光的变化，而是**光线在特定物体或人物表面上的动态变化**。这意味着我们的prompt需要将“镜头”推得更近，用更具描述性的语言来描绘光线与物体材质的互动。核心在于**将物体或人物作为主角，置于一个自然、可信的光照环境中，并捕捉光线变化在该物体表面的具体表现**。

我们将严格遵循以下原则来设计prompt：

1. **物体或人物优先**：明确定义主体，并用构图指令（如`close-up`, `medium shot`, `fills the frame`）确保它在画面中占据主导地位（>35%）。
2. **自然光源**：只使用日常生活中常见的光源，如窗户、车灯、台灯、屏幕、日出日落等。
3. **聚焦表面互动**：详细描述光线如何在物体表面产生高光、阴影、色彩变化和纹理展现，而不是描述光源本身。
4. **写实风格**：强调`photorealistic`、`naturalistic`、`cinematic`等风格，避免奇幻或过度艺术化的效果。

例如：A image focused on a detailed marble statue of a Greek goddess. The morning sun drapes a soft, golden light over one side of the statue, carving out its delicate features, emphasizing the weathered texture of the marble. 

对于视频任务的prompt我希望满足：

公式：**[场景描述]+[主题与动作]+[灯光细节]+[氛围/风格]+[相机拍摄/移动]**

- 照明细节：使用动态照明、体积照明、上帝之光、焦散、镜头光斑、充满活力的霓虹灯、移动阴影、高对比度、戏剧性的背光和黄金时段等词语。
- 风格：指定一种风格来影响整体外观，如电影、超现实主义、幻想、赛博朋克或色电影
- 相机：提及延时摄影、相机拍摄（广角、特写）和移动（慢平移、推拉变焦）可以增加活力。

例如通过上面prompt生成的图像再进行I2V：A time-lapse video focused on a detailed marble statue of a Greek goddess. As the sun moves across the sky from dawn to noon, the light and shadows dramatically change. The morning sun drapes a soft, golden light over one side of the statue, carving out its delicate features. As noon approaches, the harsh overhead sun flattens the details, then as it sets, long shadows are cast, re-sculpting the statue's form and emphasizing the weathered texture of the marble. Static camera.

### 生成提示词

---

**图文对**

现在我希望你以json格式产生满足以上规则的3个系列的英文prompt，每个系列包含一个 prompt 以及 3 种控制光源变化的prompt用于图像编辑；
格式为 {"generation": {"prompt":"prompt_here", "editing":["prompt1", "prompt2", ...]}}

---

**T2V视频文字对**

现在我希望你以json格式产生满足以上规则的3个英文prompt用于T2V视频生成模型;
格式为 ["prompt1", "prompt2", ...]

---

**I2V视频文字对**

给定用于生成图像的prompt："prompt_here"

现在我希望你根据此图像以json格式生成prompt产生满足以上规则的3个英文prompt用于I2V视频生成模型;
格式为 {"generation": {"prompt":"prompt_here", "image_path": "path", editing":["prompt1", "prompt2", ...]}}

---

负面prompt（如果有）：

blur, deformation, disfigurement, low quality, collage, graininess, logo, abstraction, illustration, computer-generated, distortion