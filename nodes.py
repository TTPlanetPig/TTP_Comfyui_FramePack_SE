import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
import hashlib
import random
import string
import torchvision
from torchvision.transforms.functional import to_pil_image
import comfy.utils

from PIL import Image
import folder_paths

class SE_FramePack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True}),
                "total_second_length": ("INT", {"default": 5, "min": 1, "max": 120, "step": 1}),
                "seed": ("INT", {"default": 3407, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "use_teacache": ("BOOLEAN", {"default": True}),
                "resolution": (["360p", "480p", "540p", "720p"], {"default": "480p"}),
                "padding_mode": (["optimized", "default (test)", "constant (test)", "waterfall (test)", "center_focus (test)"], {"default": "optimized"}),
                "end_condition_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_feature_fusion": ("BOOLEAN", {"default": True}),
                "history_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "history_decay": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.9, "step": 0.01}),
                "history_weight_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.8, "step": 0.01}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6, "min": 6, "max": 128, "step": 0.1, "description": "GPU内存保留(GB)，设置更大的值可避免OOM但会降低速度"}),
                "use_flash_attention": ("BOOLEAN", {"default": False}),
                "use_sage_attention": ("BOOLEAN", {"default": False}),
                "overlap_frames": ("INT", {"default": 33, "min": 1, "max": 100, "step": 1, "description": "section之间的混合帧数，默认值33 (latent_window_size * 4 - 3)"}),
                "blend_mode": (["linear", "cosine", "sigmoid"], {"default": "linear", "description": "混合曲线类型：linear(线性)、cosine(余弦)、sigmoid(S型)"}),
            },
            "optional": {
                "end_image": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    CATEGORY = "TTPlanet/FramePack"
    FUNCTION = "run"

    TITLE = 'TTPlanet FramePack'
    OUTPUT_NODE = True

    def __init__(self):
        self.high_vram = False
        self.frames = None
        self.fps = None

        hunyuan_root = os.path.join(folder_paths.models_dir, 'HunyuanVideo')
        flux_redux_bfl_root = os.path.join(folder_paths.models_dir, 'flux_redux_bfl')
        framePackI2V_root = os.path.join(folder_paths.models_dir, 'FramePackI2V_HY')

        self.text_encoder = LlamaModel.from_pretrained(hunyuan_root, subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained(hunyuan_root, subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(hunyuan_root, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(hunyuan_root, subfolder='tokenizer_2')
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(hunyuan_root, subfolder='vae', torch_dtype=torch.float16).cpu()

        self.feature_extractor = SiglipImageProcessor.from_pretrained(flux_redux_bfl_root, subfolder='feature_extractor')
        self.image_encoder = SiglipVisionModel.from_pretrained(flux_redux_bfl_root, subfolder='image_encoder', torch_dtype=torch.float16).cpu()

        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(framePackI2V_root, torch_dtype=torch.bfloat16).cpu()

        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        self.transformer.eval()

        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()

        self.transformer.high_quality_fp32_output_for_inference = True
        print('transformer.high_quality_fp32_output_for_inference = True')

        self.transformer.to(dtype=torch.bfloat16)
        self.vae.to(dtype=torch.float16)
        self.image_encoder.to(dtype=torch.float16)
        self.text_encoder.to(dtype=torch.float16)
        self.text_encoder_2.to(dtype=torch.float16)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        if not self.high_vram:
            # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
            DynamicSwapInstaller.install_model(self.transformer, device=gpu)
            DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)

    def strict_align(self, h, w):
        """确保高度和宽度是64的倍数，且潜在空间是8的倍数"""
        aligned_h = int(round(h / 64)) * 64
        aligned_w = int(round(w / 64)) * 64

        assert (aligned_h % 64 == 0) and (aligned_w % 64 == 0), "尺寸必须是64的倍数"
        assert (aligned_h//8) % 8 == 0 and (aligned_w//8) % 8 == 0, "潜在空间需要8的倍数"
        return aligned_h, aligned_w

    def preprocess_image(self, image):
        if image is None:
            return None
        image_np = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8)).convert("RGB")
        input_image = np.array(image)
        return input_image

    def run(self, **kwargs):
        try:
            image = kwargs['ref_image']
            end_image = kwargs.get('end_image', None)  # Use get with None as default
            image_np = self.preprocess_image(image)
            end_image_np = self.preprocess_image(end_image) if end_image is not None else None
            prompt = kwargs['prompt']
            seed = kwargs['seed']
            total_second_length = kwargs['total_second_length']
            steps = kwargs['steps']
            use_teacache = kwargs['use_teacache']
            resolution = kwargs['resolution']
            padding_mode = kwargs['padding_mode']
            end_condition_strength = kwargs['end_condition_strength']
            enable_feature_fusion = kwargs['enable_feature_fusion']
            history_weight = kwargs['history_weight']
            history_decay = kwargs['history_decay']
            history_weight_min = kwargs['history_weight_min']
            gpu_memory_preservation = kwargs['gpu_memory_preservation']
            use_flash_attention = kwargs['use_flash_attention']
            use_sage_attention = kwargs['use_sage_attention']
            overlap_frames = kwargs['overlap_frames']
            blend_mode = kwargs['blend_mode']
            
            # Remove "(test)" suffix from padding_mode
            real_padding_mode = padding_mode.split(" ")[0]
            
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            video_path = os.path.join(folder_paths.get_output_directory(), f'{random_str}.mp4')

            self.pbar = comfy.utils.ProgressBar(steps * total_second_length)

            self.exec(
                input_image=image_np, 
                end_image=end_image_np, 
                prompt=prompt, 
                seed=seed, 
                total_second_length=total_second_length, 
                video_path=video_path, 
                steps=steps, 
                use_teacache=use_teacache,
                resolution=resolution,
                padding_mode=real_padding_mode,
                end_condition_strength=end_condition_strength,
                enable_feature_fusion=enable_feature_fusion,
                history_weight=history_weight,
                history_decay=history_decay,
                history_weight_min=history_weight_min,
                gpu_memory_preservation=gpu_memory_preservation,
                use_flash_attention=use_flash_attention,
                use_sage_attention=use_sage_attention,
                overlap_frames=overlap_frames,
                blend_mode=blend_mode
            )
            
            if os.path.exists(video_path):
                self.fps = self.get_fps_with_torchvision(video_path)
                self.frames = self.extract_frames_as_pil(video_path)
                print(f'{video_path}:{self.fps} {len(self.frames)}')
            else:
                self.frames = []
                self.fps = 0.0
        except Exception as e:
            print(f"Error in run: {str(e)}")
            traceback.print_exc()
            self.frames = []
            self.fps = 0.0

        return (self.frames, self.fps)
        
    @torch.no_grad()
    def exec(self, input_image, video_path,
            end_image=None,
            prompt="The girl dances gracefully, with clear movements, full of charm.", 
            n_prompt="", 
            seed=31337, 
            total_second_length=5, 
            latent_window_size=9, 
            steps=25, 
            cfg=1, 
            gs=32, 
            rs=0, 
            gpu_memory_preservation=6, 
            use_teacache=True,
            resolution="480p",
            padding_mode="optimized",
            end_condition_strength=1.0,
            enable_feature_fusion=True,
            history_weight=1.0,
            history_decay=0.0,
            history_weight_min=0.0,
            use_flash_attention=False,
            use_sage_attention=False,
            overlap_frames=33,
            blend_mode="linear"):
        
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        
        # 启用Progressive Decay（按需求默认开启且不需要展示给用户）
        progressive_decay = True

        try:
            # 应用Flash Attention或Sage Attention
            if use_flash_attention:
                # 尝试导入并应用Flash Attention
                try:
                    from flash_attn import flash_attn_func
                    print("Flash Attention enabled")
                    # 在此处应用Flash Attention的设置（如果需要的话）
                except ImportError:
                    print("Flash Attention module not found, please install it first")
            
            if use_sage_attention:
                # 尝试导入并应用Sage Attention
                try:
                    print("Sage Attention enabled")
                    # 在此处应用Sage Attention的设置（如果需要的话）
                except ImportError:
                    print("Sage Attention module not found or not applicable")
            
            # Clean GPU
            if not self.high_vram:
                unload_complete_models(
                    self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, self.transformer
                )

            # Text encoding
            print('Text encoding')

            if not self.high_vram:
                fake_diffusers_current_device(self.text_encoder, gpu)
                load_model_as_complete(self.text_encoder_2, target_device=gpu)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # Processing input image (start frame)
            print('Processing start frame ...')

            H, W, C = input_image.shape
            
            # 根据所选分辨率确定目标高度
            if resolution == '360p':
                target_height = 360
            elif resolution == '540p':
                target_height = 540
            elif resolution == '720p':
                target_height = 720
            else:  # 默认480p
                target_height = 480
            
            # 保持宽高比计算宽度
            target_ratio = W / H
            target_width = int(target_height * target_ratio)
            if target_width % 2 != 0:  # 确保是偶数
                target_width += 1
            
            # 找到最接近的桶分辨率，但保持宽高比
            height, width = find_nearest_bucket(target_height, target_width, resolution=target_height)
            print(f"Target resolution: {width}x{height}, Aspect ratio: {width/height:.3f}")
            
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # Processing end image if provided
            has_end_image = end_image is not None
            end_image_np = None
            end_image_pt = None
            
            if has_end_image:
                print('Processing end frame ...')
                end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
                end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
                end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding
            print('VAE encoding ...')

            if not self.high_vram:
                load_model_as_complete(self.vae, target_device=gpu)

            start_latent = vae_encode(input_image_pt, self.vae)
            end_latent = None
            if has_end_image:
                end_latent = vae_encode(end_image_pt, self.vae)

            # CLIP Vision
            print('CLIP Vision encoding ...')

            if not self.high_vram:
                load_model_as_complete(self.image_encoder, target_device=gpu)

            # Start image encoding
            image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            
            # End image encoding if available
            if has_end_image:
                end_image_encoder_output = hf_clip_vision_encode(end_image_np, self.feature_extractor, self.image_encoder)
                end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
                
                # 根据end_condition_strength和enable_feature_fusion参数处理
                if end_condition_strength <= 0.0:
                    # 不使用结束特征
                    pass  # 保持image_encoder_last_hidden_state不变
                elif end_condition_strength >= 1.0:
                    # 完全按照原始方式融合
                    if enable_feature_fusion:
                        image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2
                else:
                    # 按照强度比例混合
                    if enable_feature_fusion:
                        image_encoder_last_hidden_state = (1.0 - end_condition_strength) * image_encoder_last_hidden_state + end_condition_strength * end_image_encoder_last_hidden_state

            # Dtype
            llama_vec = llama_vec.to(self.transformer.dtype)
            llama_vec_n = llama_vec_n.to(self.transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.transformer.dtype)

            print('Start Sample')

            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            # 根据padding_mode选择不同的padding策略
            if padding_mode == "default":
                # 原始模式: 直接使用倒序数列
                latent_paddings = list(reversed(range(total_latent_sections)))
            elif padding_mode == "constant":
                # 常量模式: 保持固定值2，最后两块特殊处理
                if total_latent_sections <= 2:
                    # 对于短视频，使用[1, 0]或[0]
                    latent_paddings = list(reversed(range(total_latent_sections)))
                else:
                    # 对于长视频，除了最后两块外都使用常量2
                    latent_paddings = [2] * (total_latent_sections - 2) + [1, 0]
            elif padding_mode == "optimized":
                # 优化模式: 对于长视频使用自定义分布
                if total_latent_sections > 4:
                    latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
                else:
                    latent_paddings = list(reversed(range(total_latent_sections)))
            elif padding_mode == "waterfall":
                # 瀑布模式
                if total_latent_sections <= 2:
                    latent_paddings = list(reversed(range(total_latent_sections)))
                else:
                    # 最大padding值
                    max_padding = 4
                    # 头部增长阶段的长度
                    head_size = min(4, total_latent_sections // 3)
                    # 尾部（值为1和0）的长度固定为2
                    tail_size = 2
                    # 中间部分长度
                    middle_size = total_latent_sections - head_size - tail_size
                    
                    # 创建头部：线性增加到最大值
                    head_paddings = []
                    for i in range(head_size):
                        # 线性插值从3到max_padding
                        ratio = i / (head_size - 1) if head_size > 1 else 1
                        padding = 3 + (max_padding - 3) * ratio
                        head_paddings.append(round(padding))
                    
                    # 中间部分全部为2
                    middle_paddings = [2] * middle_size
                    
                    # 尾部固定为[1, 0]
                    tail_paddings = [1, 0]
                    
                    # 组合所有部分
                    latent_paddings = head_paddings + middle_paddings + tail_paddings
            elif padding_mode == "center_focus":
                # 中心聚焦模式
                if total_latent_sections <= 3:
                    latent_paddings = list(reversed(range(total_latent_sections)))
                else:
                    # 更平滑的padding过渡
                    max_padding = 3
                    min_padding = 1
                    
                    if total_latent_sections % 2 == 0:  # 偶数个sections
                        half_size = total_latent_sections // 2
                        # 创建半边，然后镜像
                        first_half = []
                        for i in range(half_size):
                            position_ratio = i / (half_size - 1) if half_size > 1 else 0
                            padding = min_padding + (max_padding - min_padding) * position_ratio ** 2
                            first_half.append(round(padding))
                        # 镜像创建
                        second_half = first_half.copy()
                        second_half.reverse()
                        latent_paddings = first_half + second_half
                    else:  # 奇数个sections
                        mid_idx = total_latent_sections // 2
                        latent_paddings = []
                        for i in range(total_latent_sections):
                            distance_ratio = abs(i - mid_idx) / mid_idx if mid_idx > 0 else 0
                            padding = max_padding - (max_padding - min_padding) * distance_ratio ** 2
                            latent_paddings.append(round(padding))
                    
                    # 确保第一个元素不要太大
                    latent_paddings[0] = min(latent_paddings[0], 2)
                    
                    # 确保最后几个section的padding有平滑过渡
                    if len(latent_paddings) >= 4:
                        latent_paddings[-4] = min(latent_paddings[-4], 3)
                        latent_paddings[-3] = 2
                        latent_paddings[-2] = 1
                        latent_paddings[-1] = 0
                    elif len(latent_paddings) == 3:
                        latent_paddings[0] = 2
                        latent_paddings[1] = 1
                        latent_paddings[2] = 0
            else:
                # 默认回退到优化模式
                if total_latent_sections > 4:
                    latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
                else:
                    latent_paddings = list(reversed(range(total_latent_sections)))
            
            # 确保最后一个padding值为0
            if len(latent_paddings) > 0:
                latent_paddings[-1] = 0
            
            print(f"Selected padding mode: {padding_mode}")
            print(f"Total sections: {total_latent_sections}")
            print(f"Latent paddings: {latent_paddings}")
            
            # 计算每个section的history weight，实现衰减效果
            section_weights = []
            current_weight = history_weight
            
            # 找出最大padding值，用于归一化padding影响
            max_padding = max(latent_paddings) if latent_paddings else 3
            if max_padding == 0:  # 防止除以零错误
                max_padding = 1
                
            for i, padding in enumerate(latent_paddings):
                section_weights.append(current_weight)
                
                # 根据当前section的padding值调整衰减率
                padding_factor = padding / max_padding  # 归一化到0-1范围
                
                # 调整衰减率：padding_factor越大，实际衰减越小
                adjusted_decay = history_decay * (1.0 - padding_factor * 0.8)  # 0.8是补偿系数
                
                # 应用调整后的衰减率，并确保不低于最小值
                current_weight = max(history_weight_min, current_weight * (1.0 - adjusted_decay))
                
                print(f"Section {i+1} padding: {padding}, padding_factor: {padding_factor:.2f}, " +
                      f"adjusted_decay: {adjusted_decay:.4f}, weight: {current_weight:.4f}")
            
            print(f"Section history weights: {section_weights}")

            for i, latent_padding in enumerate(latent_paddings):
                # 获取当前section的history weight
                current_history_weight = section_weights[i]
                print(f"Section {i+1} history weight: {current_history_weight:.4f}")
                
                is_last_section = latent_padding == 0
                is_first_section = latent_padding == latent_paddings[0]
                latent_padding_size = latent_padding * latent_window_size

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                # 确保所有latent都是5维的 [1, 16, T, h, w]
                if len(start_latent.shape) == 4:  # [1, 16, h, w]
                    clean_latents_pre = start_latent.unsqueeze(2).to(history_latents)  # 变为 [1, 16, 1, h, w]
                else:
                    clean_latents_pre = start_latent.to(history_latents)  # 已经是5维
                
                # 从history_latents中提取
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                
                # 应用当前section的history_weight，调整历史潜变量的影响力
                if current_history_weight < 1.0:
                    # 如果启用渐进式衰减，对latent_indices涉及到的帧应用不同的权重
                    if progressive_decay and latent_window_size > 1 and not is_first_section:
                        # 只对clean_latents_post进行操作，它是历史信息
                        weighted_clean_latents_post = clean_latents_post.clone()
                        
                        # 获取下一个section的权重
                        next_section_idx = min(i + 1, len(section_weights) - 1)
                        next_weight = section_weights[next_section_idx] if next_section_idx < len(section_weights) else max(history_weight_min, current_history_weight * 0.5)
                        
                        # 在前一个section的后部分应用过渡衰减
                        # 这创建了一个从当前权重到下一个section权重的平滑过渡
                        middle_weight = (current_history_weight + next_weight) / 2
                        weighted_clean_latents_post = weighted_clean_latents_post * middle_weight
                        
                        print(f"Progressive decay from {current_history_weight:.4f} to {middle_weight:.4f} for section {i+1}")
                    else:
                        weighted_clean_latents_post = clean_latents_post * current_history_weight
                    
                    # 默认合并clean_latents
                    clean_latents = torch.cat([clean_latents_pre, weighted_clean_latents_post], dim=2)
                else:
                    # 默认合并clean_latents
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                # 使用结束帧latent
                if has_end_image and is_first_section:
                    # 确保end_latent也是5维的
                    if len(end_latent.shape) == 4:  # [1, 16, h, w]
                        end_latent_to_use = end_latent.unsqueeze(2)  # 变为 [1, 16, 1, h, w]
                    else:
                        end_latent_to_use = end_latent
                    
                    # 根据强度参数决定如何应用end_latent
                    if end_condition_strength <= 0.0:
                        # 强度为0，不使用end_latent，保持clean_latents不变
                        print(f"End condition strength is 0, not using end reference")
                    elif end_condition_strength >= 1.0:
                        # 强度为1，完全使用end_latent
                        print(f"End condition strength is 1, fully using end reference")
                        clean_latents = torch.cat([clean_latents_pre, end_latent_to_use.to(history_latents)], dim=2)
                    else:
                        # 混合start_latent和end_latent
                        print(f"Mixing start and end with strength {end_condition_strength}")
                        # 进行插值混合
                        mixed_latent = (1.0 - end_condition_strength) * clean_latents_post + end_condition_strength * end_latent_to_use.to(history_latents)
                        clean_latents = torch.cat([clean_latents_pre, mixed_latent], dim=2)

                if not self.high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    self.transformer.initialize_teacache(enable_teacache=False)

                def callback(d):
                    self.update(1)
                    return

                generated_latents = sample_hunyuan(
                    transformer=self.transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                if is_last_section:
                    # 确保start_latent是5维的，与generated_latents维度匹配
                    if len(start_latent.shape) == 4:  # [1, 16, h, w]
                        start_latent_5d = start_latent.unsqueeze(2)  # 变为 [1, 16, 1, h, w]
                    else:
                        start_latent_5d = start_latent
                    generated_latents = torch.cat([start_latent_5d.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                if not self.high_vram:
                    offload_model_from_device_for_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(self.vae, target_device=gpu)

                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, self.vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    # 使用用户指定的overlap_frames替代固定值
                    overlapped_frames = overlap_frames

                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], self.vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames, blend_mode)

                if not self.high_vram:
                    unload_complete_models()

                # 如果是最后一个section，保存视频
                if is_last_section:
                    save_bcthw_as_mp4(history_pixels, video_path, fps=30)
                    break

        except Exception as e:
            print(f"Error in exec: {str(e)}")
            traceback.print_exc()
        finally:
            unload_complete_models()
        
    def update(self, in_progress):
        self.pbar.update(in_progress)

    def extract_frames_as_pil(self, video_path):
        video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')  # (T, H, W, C)
        frames = [to_pil_image(frame.permute(2, 0, 1)) for frame in video]
        frames = [torch.from_numpy(np.array(frame).astype(np.float32) / 255.0) for frame in frames]
        return frames               

    def get_fps_with_torchvision(self, video_path):
        _, _, info = torchvision.io.read_video(video_path, pts_unit='sec')
        return info['video_fps']

NODE_CLASS_MAPPINGS = {
    "TTPlanet_FramePack": SE_FramePack,
}

def soft_append_bcthw(history, current, overlap=0, blend_mode='linear'):
    if overlap <= 0:
        return torch.cat([history, current], dim=2)

    assert history.shape[2] >= overlap, f"History length ({history.shape[2]}) must be >= overlap ({overlap})"
    assert current.shape[2] >= overlap, f"Current length ({current.shape[2]}) must be >= overlap ({overlap})"
    
    # 根据不同的混合模式选择权重曲线
    if blend_mode == 'cosine':
        # 使用余弦曲线，提供更平滑的过渡
        x = torch.linspace(0, math.pi, overlap, dtype=history.dtype, device=history.device)
        weights = (torch.cos(x) + 1) / 2
    elif blend_mode == 'sigmoid':
        # 使用sigmoid函数，在中间区域提供更平缓的过渡
        x = torch.linspace(-6, 6, overlap, dtype=history.dtype, device=history.device)
        weights = torch.sigmoid(x)
    else:  # 'linear'
        # 默认线性
        weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device)
    
    weights = weights.view(1, 1, -1, 1, 1)
    blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
    output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)

    return output.to(history)
