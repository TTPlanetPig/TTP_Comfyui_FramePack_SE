# TTP_ComfyUI_FramePack_SE

**Provide ComfyUI support for FramePack start-and-end image reference**

---

First, thanks to [lvmin Zhang](https://github.com/lllyasviel) for the **FramePack** application—it offers a very interesting approach that makes video generation easier and more accessible.  
Original repository:  
https://github.com/lllyasviel/FramePack

---

## Changes in This Project

- Based on the original repo, we inject a simple `end_image` to enable start-and-end frame references  
- Check out my PR for the full diff:  
  https://github.com/lllyasviel/FramePack/pull/167  
- This PR addresses the “frozen background” criticism

---

## Issues Encountered

1. When the start and end frames differ too greatly, the model struggles and often produces “slideshow-style” cuts.  
2. Although it’s been attributed to “needing further training,” I believe:  
   - The Hunyuan model handles Human static poses well but lacks smooth dynamic transitions  
   - With lvmin Zhang’s improved the Hunyuan base model, we can unlock more possibilities

---

## My Optimizations

- **Tweaked the generation pipeline code** to strike a balance between variation and frame consistency  
- Tested and tuned several parameters to ensure smooth transitions in most scenarios


<div style="display:flex; align-items:center; max-width:900px; margin:auto; border:1px solid #ddd;">

  <!-- 左侧：视频 -->
  <div style="flex:1; padding:4px;">
    <video
      src="https://github.com/user-attachments/assets/4710bc34-0cc0-42f9-bd77-e7f82890c344"
      controls
      style="width:100%; height:auto; max-height:500px; object-fit:cover;"
    ></video>
  </div>

  <!-- 右侧：上下两张图 -->
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; gap:8px; padding:4px;">
    <img
      src="https://github.com/user-attachments/assets/444ec179-8ad5-4686-8264-3f7079d8e668"
      alt="DSC07174"
      style="width:20%; height:auto; max-height:240px; object-fit:contain;"
    />
    <img
      src="https://github.com/user-attachments/assets/cdcd714b-a83b-4803-9299-025c593ab005"
      alt="DSC07189"
      style="width:20%; height:auto; max-height:240px; object-fit:contain;"
    />
  </div>

</div>

## Model Download & Location

You can either download each model manually from Hugging Face or use the bundled model package.

### 1. Manual Download

- **HunyuanVideo**  
  [https://huggingface.co/hunyuanvideo-community/HunyuanVideo/tree/main](https://huggingface.co/hunyuanvideo-community/HunyuanVideo/tree/main)
- **Flux Redux BFL**  
  [https://huggingface.co/lllyasviel/flux_redux_bfl/tree/main](https://huggingface.co/lllyasviel/flux_redux_bfl/tree/main)
- **FramePackI2V**  
  [https://huggingface.co/lllyasviel/FramePackI2V_HY/tree/main](https://huggingface.co/lllyasviel/FramePackI2V_HY/tree/main)

- Baidu: https://pan.baidu.com/s/17h23yvJXa6AczGLcybsd_A?pwd=mbqa
- Quark: https://pan.quark.cn/s/80ff4f39c15b

### 2. Model location

Copy the contents into the `models/` folder, information ref from [HM-RunningHub/ComfyUI_RH_FramePack](https://github.com/HM-RunningHub/ComfyUI_RH_FramePack) 

```text
comfyui/models/
├── flux_redux_bfl
│   ├── feature_extractor/
│   │   └── preprocessor_config.json
│   ├── image_embedder/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   ├── image_encoder/
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── model_index.json
│   └── README.md
├── FramePackI2V_HY
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00003.safetensors
│   ├── diffusion_pytorch_model-00002-of-00003.safetensors
│   ├── diffusion_pytorch_model-00003-of-00003.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── README.md
└── HunyuanVideo
    ├── config.json
    ├── model_index.json
    ├── README.md
    ├── scheduler/
    │   └── scheduler_config.json
    ├── text_encoder/
    │   ├── config.json
    │   ├── model-00001-of-00004.safetensors
    │   ├── model-00002-of-00004.safetensors
    │   ├── model-00003-of-00004.safetensors
    │   ├── model-00004-of-00004.safetensors
    │   └── model.safetensors.index.json
    ├── text_encoder_2/
    │   ├── config.json
    │   └── model.safetensors
    ├── tokenizer/
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── tokenizer.json
    ├── tokenizer_2/
    │   ├── merges.txt
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── vocab.json
    └── vae/
        ├── config.json
        └── diffusion_pytorch_model.safetensors
```


---

## Parameter Guide

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/3e9a0954-73b7-47e4-b92e-07aa0c72e58b"
    alt="Parameter Diagram"
    style="max-width: 30%;"
  />
</p>

- **padding_mode**  
  - Still experimental—use `optimized` for now  
- **end_condition_strength** & **enable_feature_fusion**  
  - Mutually exclusive; pick only one  
  - Lower `end_condition_strength` grants more freedom but reduces end-frame similarity  
- **history_weight**  
  - Controls history influence, default 100%  
- **history_decay**  
  - Linearly decays history weight; increase `decay` if you need more variation

---
## Examples
![TTP_FramePack_Start_End_Image_example](https://github.com/user-attachments/assets/0468ea9c-a5fe-4067-9ef6-6e89b6d58754)

---
> Feel free to share feedback or suggestions in Issues or PRs!

## **Star History**
<a href="https://star-history.com/#TTPlanetPig/TTP_Comfyui_FramePack_SE&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/TTP_Comfyui_FramePack_SE&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/TTP_Comfyui_FramePack_SE&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=TTPlanetPig/TTP_Comfyui_FramePack_SE&type=Date" />
 </picture>
</a>

