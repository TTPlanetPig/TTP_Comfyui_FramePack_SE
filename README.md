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
   - The Hunyuan model handles static poses well but lacks smooth dynamic transitions  
   - With lvmin Zhang’s improvements as a base, we can unlock more possibilities

---

## My Optimizations

- **Tweaked the generation pipeline code** to strike a balance between variation and frame consistency  
- Tested and tuned several parameters to ensure smooth transitions in most scenarios

---

## Parameter Guide

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/3e9a0954-73b7-47e4-b92e-07aa0c72e58b"
    alt="Parameter Diagram"
    style="max-width: 50%;"
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

> Feel free to share feedback or suggestions in Issues or PRs!

## **Star History**
<a href="https://star-history.com/#TTPlanetPig/TTP_Comfyui_FramePack_SE&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/TTP_Comfyui_FramePack_SE&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/TTP_Comfyui_FramePack_SE&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=TTPlanetPig/TTP_Comfyui_FramePack_SE&type=Date" />
 </picture>
</a>

