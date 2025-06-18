好的，这是一个经过翻译、完善和结构优化的英文版Prompt。您可以将此内容直接用于调用大语言模型（如GPT-4等）的API，以自动生成您需要的图像和视频提示词。

我在其中优化了JSON结构，使其更具可读性和扩展性，并对核心要求进行了提炼和强化，以确保模型能更好地理解任务的本质。

---

### **Refined and Translated Prompt for LLM API**

#### **Objective**
You are an expert prompt engineer specializing in generating highly detailed, photorealistic prompts for advanced text-to-image and text-to-video diffusion models.

Your mission is to create a structured set of prompts in JSON format. These prompts will be used to generate a synthetic dataset for training a **universal video relighting model**. The goal is to simulate realistic and temporally consistent lighting changes on various objects and scenes.

---

#### **Background & Core Problem**

Modern diffusion models have unlocked unprecedented capabilities in image editing, especially in **relighting**. They can handle complex, "in-the-wild" lighting effects far beyond simple light source changes, such as creating cinematic backlighting, simulating shadows cast by objects (e.g., tree leaves, window blinds), or even manipulating artistic lighting.

We aim to extend this to **video relighting**, which is significantly more challenging. The primary obstacle is maintaining **temporal consistency**. Applying image relighting models frame-by-frame results in flickering, as light sources and their effects "jump" between frames. Traditional methods require specialized hardware (like OLAT stages) and are not scalable.

Therefore, we need to generate a large-scale, diverse, and naturalistic dataset of images and videos showing consistent lighting changes on static subjects. This dataset will be the foundation for training a model that can relight general-purpose videos.

---

#### **Guiding Philosophy: The "Natural Lighting Stage"**

The core creative principle is to treat the subject as the protagonist on a **"natural lighting stage."** The focus is **not** on the environment or the light source itself, but on the **dynamic interplay of light and shadow across the subject's surface**. The prompts must "push the lens closer" to capture how light reveals texture, form, and material.

**Strict Prompt Design Principles:**

1.  **Subject-First Composition:** The subject (an object or person) is paramount. It must be clearly defined and dominate the frame (occupying >35% of the image area). Use compositional keywords like `close-up shot`, `medium shot`, `product shot`, `centered`, `fills the frame`.
2.  **Naturalistic Light Sources:** Exclusively use common, real-world light sources. Examples: sunlight through a window, passing car headlights, a desk lamp, a flickering TV screen, sunrise/sunset glow, streetlights, firelight. **Avoid fantasy or sci-fi lighting.**
3.  **Focus on Surface Interaction:** The prompt's language must meticulously describe *how* light interacts with the subject's material and form. Use descriptive terms for highlights, shadows, caustics, subsurface scattering, reflections, and how light affects color and texture.
4.  **Photorealistic & Cinematic Style:** Emphasize realism. Use keywords like `photorealistic`, `ultra-realistic`, `cinematic`, `natural lighting`, `soft lighting`, `sharp focus`, `8K`, `shot on ARRI Alexa`.

---

#### **Task 1: Generate Image & Editing Prompt Sets**

**Goal:** Create pairs of prompts. The first generates a base image with a clear subject and lighting. The subsequent prompts edit *only the lighting* on that image, keeping the subject and camera static.

**Proposed JSON Structure (More Descriptive):**
I've refined your proposed JSON to be more explicit and scalable.

```json
{
  "image_sets": [
    {
      "set_id": "series_001",
      "description": "A brief, human-readable description of the scene, e.g., 'A ceramic vase by a window during different times of day.'",
      "base_generation_prompt": "The detailed prompt to generate the initial image.",
      "relighting_prompts": [
        "A prompt describing the first lighting variation.",
        "A prompt describing the second lighting variation.",
        "A prompt describing the third lighting variation."
      ]
    }
  ]
}
```

**Instruction:** Now, generate **3** unique sets following the `image_sets` JSON structure above.

---

#### **Task 2: Generate Text-to-Video (T2V) Prompts**

**Goal:** Create standalone prompts for a T2V model to generate a complete video with dynamic lighting changes from a single text description.

**Video Prompt Formula:**
`[Scene Description] + [Subject & Its State] + [Dynamic Lighting Details] + [Mood/Style] + [Camera Shot & Movement]`
*   **Lighting Keywords:** `dynamic lighting`, `volumetric light`, `god rays`, `caustics`, `lens flare`, `shifting shadows`, `high-contrast`, `dramatic backlighting`, `golden hour`, `blue hour`, `time-lapse`.
*   **Camera Keywords:** `time-lapse`, `static camera`, `fixed shot`, `slow pan`, `dolly zoom`. **(Prefer static camera to isolate the lighting effect).**

**Proposed JSON Structure:**
A simple list is fine, but wrapping it in an object allows for better organization.

```json
{
  "t2v_prompts": [
    "A complete, detailed prompt for the first video.",
    "A complete, detailed prompt for the second video.",
    "A complete, detailed prompt for the third video."
  ]
}
```

**Instruction:** Now, generate **3** unique prompts following the `t2v_prompts` JSON structure above.

---

#### **Task 3: Generate Image-to-Video (I2V) Prompts**

**Goal:** Given a pre-existing image (generated from a prompt), create several new prompts to animate it into a video with changing light.

**Context:** Assume an image has been generated using the prompt:
`"A photorealistic close-up of a weathered leather-bound book resting on an old wooden desk. The book is the central focus, filling the frame. Soft morning light from a side window gently illuminates the cover, highlighting the cracked texture of the leather and the gold foil of the title."`

**Proposed JSON Structure:**
This structure clearly links the source image prompt to the new video generation prompts.

```json
{
  "i2v_tasks": [
    {
      "source_image_prompt": "The prompt used to generate the original image.",
      "source_image_path": "/path/to/generated/image.png",
      "video_generation_prompts": [
        "A prompt to animate the image with the first lighting change.",
        "A prompt to animate the image with the second lighting change.",
        "A prompt to animate the image with the third lighting change."
      ]
    }
  ]
}
```

**Instruction:** Based on the example `source_image_prompt` provided above, generate **3** unique `video_generation_prompts` following the `i2v_tasks` JSON structure.

---

#### **Negative Prompts (To be used for all tasks)**
`blur, blurry, deformation, disfigurement, low quality, low-res, collage, graininess, noisy, logo, watermark, signature, text, abstraction, illustration, painting, anime, cgi, 3d render, distortion, unrealistic, unnatural.`