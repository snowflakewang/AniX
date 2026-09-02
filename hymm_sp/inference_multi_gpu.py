import os
from pathlib import Path
from loguru import logger
import torch
import torchvision
from einops import rearrange
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
#from hymm_sp.config import parse_args
from hymm_sp.config_train import parse_args
#from hymm_sp.sample_inference import HunyuanVideoSampler
from hymm_sp.sample_inference_ar_v2v_lora import HunyuanVideoSampler
#from hymm_sp.data_kits.video_dataset import JsonDataset
from hymm_sp.data_kits.video_dataset_lora import JsonDataset, VideoDataset
from hymm_sp.data_kits.data_tools import save_videos_grid
import numpy as np
import datetime
import time
import json
import imageio

import pdb
from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)

def is_video_openable(filepath):
    """
    使用 imageio 检查一个视频文件是否真实可用。
    
    这包括三个层面的检查：
    1. 路径是否存在且为文件。
    2. Imageio 能否成功解析文件头并获取读取器。
    3. 解析出的视频时长是否大于0。
    
    参数:
    filepath (str): 视频文件的路径。
    
    返回:
    bool: 如果视频可用则返回 True，否则返回 False。
    """
    # --- 层面一：文件系统检查 (快速失败) ---
    if not os.path.isfile(filepath):
        # 如果路径不存在，或者它是个文件夹，直接返回 False
        print(f"❌ [系统检查] 失败: '{filepath}' 不是一个有效的文件路径。")
        return False
        
    # --- 层面二 & 三：内容格式检查 (核心) ---
    try:
        # 尝试使用 imageio 获取视频读取器
        # 'ffmpeg' 是处理视频最可靠的插件
        # 使用 with 语句确保资源被正确释放
        with imageio.get_reader(filepath, 'ffmpeg') as reader:
            # 成功获取读取器意味着文件头和容器是可读的
            # 接下来检查元数据中的时长
            meta = reader.get_meta_data()
            
            # 使用 .get() 安全地获取 'duration' 键
            duration = meta.get('duration')
            
            # 检查时长是否有效 (存在且大于0)
            if duration is None or float(duration) <= 0:
                print(f"❌ [内容检查] 失败: '{filepath}' 可以加载，但时长为0或无效。")
                return False
        
        # 如果代码能执行到这里，说明文件可读且时长有效
        # print(f"✅ [综合检查] 成功: '{filepath}' 是一个有效的视频文件，时长为 {duration:.2f} 秒。")
        return True
        
    except Exception as e:
        # 捕获所有 imageio 在尝试读取文件时可能抛出的异常
        # 例如：文件损坏、编码不支持、权限问题等
        print(f"❌ [内容检查] 失败: 解析 '{filepath}' 时发生错误。文件可能已损坏或格式不受支持。")
        print(f"   错误详情: {e}")
        return False

def main():
    args = parse_args()
    models_root_path = Path(args.ckpt)
    print("*"*20) 
    initialize_distributed(args.seed)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    print("+"*20)
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    rank = 0
    vae_dtype = torch.float16
    device = torch.device("cuda")
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = torch.distributed.get_rank()

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    # Get the updated args
    args = hunyuan_video_sampler.args
    
    if args.video_condition:
        json_dataset = VideoDataset(args, device=device)
    elif args.audio_condition:
        raise NotImplementedError
    else:
        raise NotImplementedError
    sampler = DistributedSampler(json_dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False)
    json_loader = DataLoader(json_dataset, batch_size=1, shuffle=False, sampler=sampler, drop_last=False)
    for batch_index, batch in enumerate(json_loader, start=1):
        pixel_value_llava = batch['pixel_value_llava'].to(device)
        pixel_value_ref = batch['pixel_value_ref'].to(device)
        uncond_pixel_value_llava = batch['uncond_pixel_value_llava']
        prompt = batch['prompt'][0]
        negative_prompt = batch['negative_prompt'][0]        
        name = batch['name'][0]
        save_name = batch['data_name'][0]
        seed = batch['seed']
        audio_prompts = batch['audio_prompts'][0].to(device) if 'audio_prompts' in batch else None
        audio_path = batch['audio_path'][0] if 'audio_path' in batch else None

        output_path = f"{batch['output_video'][0]}.mp4"
        if os.path.exists(output_path) and is_video_openable(output_path):
            print(f"{output_path} already exist. Skipping...")
            continue

        if 'pixel_value_input_video' in batch.keys():
            pixel_value_input_video = batch['pixel_value_input_video']
        else:
            pixel_value_input_video = None
        pixel_value_ref = pixel_value_ref * 2 - 1.
        #pixel_value_ref_for_vae = rearrange(pixel_value_ref,"b c h w -> b c 1 h w")
        if pixel_value_ref.ndim == 4: # single image
            pixel_value_ref_for_vae = rearrange(pixel_value_ref,"b c h w -> b c 1 h w")
        elif pixel_value_ref.ndim == 5: # multi-view images
            pixel_value_ref_for_vae = rearrange(pixel_value_ref,"b f c h w -> b c f h w")

        if pixel_value_llava.ndim == 5: # 4-view condition [b, f, c, h, w]
            if pixel_value_llava.shape[0] == 1: # batch_size=1
                pixel_value_llava = pixel_value_llava.squeeze(0) # [1, f, c, h, w] -> [f, c, h, w]
            else:
                raise Exception(f"Unsupported batch_size:  {pixel_value_llava.shape[0]}. Only batch_size=1 is supported now.")
        if uncond_pixel_value_llava.ndim == 5: # 4-view condition [b, f, c, h, w]
            if uncond_pixel_value_llava.shape[0] == 1: # batch_size=1
                uncond_pixel_value_llava = uncond_pixel_value_llava.squeeze(0) # [1, f, c, h, w] -> [f, c, h, w]
            else:
                raise Exception(f"Unsupported batch_size:  {uncond_pixel_value_llava.shape[0]}. Only batch_size=1 is supported now.")
        assert pixel_value_llava.shape == uncond_pixel_value_llava.shape
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
            # each view is encoded by VAE separately
            for f in range(pixel_value_ref_for_vae.shape[2]):
                if f > 0:
                    ref_latents = torch.cat([ref_latents, hunyuan_video_sampler.vae.encode(pixel_value_ref_for_vae[:, :, f:f+1, :, :].clone()).latent_dist.sample()], dim=2)
                    uncond_ref_latents = torch.cat([uncond_ref_latents, hunyuan_video_sampler.vae.encode(torch.ones_like(pixel_value_ref_for_vae[:, :, f:f+1, :, :])).latent_dist.sample()], dim=2)
                elif f == 0:
                    ref_latents = hunyuan_video_sampler.vae.encode(pixel_value_ref_for_vae[:, :, f:f+1, :, :].clone()).latent_dist.sample()
                    uncond_ref_latents = hunyuan_video_sampler.vae.encode(torch.ones_like(pixel_value_ref_for_vae[:, :, f:f+1, :, :])).latent_dist.sample()
            print(f"ref_latents.shape: {ref_latents.shape}, uncond_ref_latents.shape: {uncond_ref_latents.shape}")
            #ref_latents = hunyuan_video_sampler.vae.encode(pixel_value_ref_for_vae.clone()).latent_dist.sample()
            #uncond_ref_latents = hunyuan_video_sampler.vae.encode(torch.ones_like(pixel_value_ref_for_vae)).latent_dist.sample()
            ref_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)
            uncond_ref_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)

        if pixel_value_input_video is not None:
            pixel_value_input_video = pixel_value_input_video.to(device=device, dtype=vae_dtype)
            pixel_value_input_video = pixel_value_input_video / 255.0
            normalize_fn = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            pixel_value_input_video = normalize_fn(pixel_value_input_video)
            if pixel_value_input_video.ndim == 4:
                pixel_value_input_video = pixel_value_input_video.unsqueeze(0)
            pixel_value_input_video = rearrange(pixel_value_input_video, "b f c h w -> b c f h w")
            
            pixel_value_input_video = pixel_value_input_video[:, :, -((args.num_clean_latent_frames-1)*4+1):, :, :] # -34 is wrong???
            print(f"pixel_value_input_video.shape: {pixel_value_input_video.shape}")

            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                if args.cpu_offload:
                    hunyuan_video_sampler.vae.to('cuda')
                
                hunyuan_video_sampler.vae.enable_spatial_tiling()
                hunyuan_video_sampler.vae.eval()
                with torch.no_grad():
                    pre_frame_latents = \
                        hunyuan_video_sampler.vae.encode(pixel_value_input_video).latent_dist.mode() # `mode()` copied from https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/blob/1481c1d5ae88e9905f54f2a3c6a1b68ef2a10528/hyvideo/hyvae_extract/run.py#L85C13-L85C60
                hunyuan_video_sampler.vae.disable_spatial_tiling()

                print(f"pre_frame_latents.shape: {pre_frame_latents.shape}")

                pre_frame_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)

                if args.cpu_offload:
                    hunyuan_video_sampler.vae.to('cpu')
                    torch.cuda.empty_cache()
        else:
            pre_frame_latents = None
        
        if args.video_condition:
            if args.only_text_scene:
                bg_latents = None
            else:
                pixel_value_bg = batch['pixel_value_bg'].to(device) * 2 - 1.                        
                pixel_value_mask = batch['pixel_value_mask'].to(device) * 2 - 1.
                
                # pixel_value_input_scene_video = pixel_value_input_scene_video[:, :, -33:, :, :]
                print(f"pixel_value_bg.shape: {pixel_value_bg.shape}")
                print(f"pixel_value_mask.shape: {pixel_value_mask.shape}")

                with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                    
                    hunyuan_video_sampler.vae.enable_spatial_tiling()
                    hunyuan_video_sampler.vae.eval()
                    with torch.no_grad():
                        # bg_latents = hunyuan_video_sampler.vae.encode(pixel_value_bg).latent_dist.sample()
                        # mask_latents = hunyuan_video_sampler.vae.encode(pixel_value_mask).latent_dist.sample()
                        bg_latents = hunyuan_video_sampler.vae.encode(pixel_value_bg).latent_dist.mode()
                        mask_latents = hunyuan_video_sampler.vae.encode(pixel_value_mask).latent_dist.mode()
                        
                    hunyuan_video_sampler.vae.disable_spatial_tiling()

                    bg_latents = torch.cat([bg_latents, mask_latents], dim=1)

                    bg_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)

                    _, _, t, h, w = bg_latents.shape
                    args.video_size = (h * 8, w * 8)
                    args.sample_n_frames = (t - 1) * 4 + 1

                    print(f"bg_latents.shape: {bg_latents.shape}")
        else:
            raise NotImplementedError

        prompt = args.add_pos_prompt + prompt
        negative_prompt = args.add_neg_prompt + negative_prompt
        outputs = hunyuan_video_sampler.predict(
                prompt=prompt,
                name=name,
                size=args.video_size,
                seed=seed,
                pixel_value_llava=pixel_value_llava,
                uncond_pixel_value_llava=uncond_pixel_value_llava,
                ref_latents=ref_latents,
                uncond_ref_latents=uncond_ref_latents,
                bg_latents=bg_latents,
                audio_prompts=audio_prompts,
                audio_strength=args.audio_strength,
                pre_frame_latents=pre_frame_latents,
                video_length=args.sample_n_frames,
                guidance_scale=args.cfg_scale,
                num_images_per_prompt=args.num_images,
                negative_prompt=negative_prompt,
                infer_steps=args.infer_steps,
                flow_shift=args.flow_shift_eval_video,
                use_linear_quadratic_schedule=args.use_linear_quadratic_schedule,
                linear_schedule_end=args.linear_schedule_end,
                use_deepcache=args.use_deepcache,
        )

        if rank == 0:
            samples = outputs['samples']
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                out_path = f"{batch['output_video'][0]}.mp4"
                save_videos_grid(sample, out_path, fps=25, quality=8)
                logger.info(f'Sample save to: {out_path}')
        else:
            time.sleep(8)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
