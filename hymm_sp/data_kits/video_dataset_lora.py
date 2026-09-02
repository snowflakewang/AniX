import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from hymm_sp.data_kits.data_tools import *
from decord import VideoReader


class DataPreprocess(object):
    def __init__(self):
        self.llava_size = (336, 336)
        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(self.llava_size, interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def get_batch(self, image_path, size):
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = Image.open(image_path).convert('RGB')
        llava_item_image = pad_image(image.copy(), self.llava_size)
        uncond_llava_item_image = np.ones_like(llava_item_image) * 255
        cat_item_image = pad_image(image.copy(), size)

        llava_item_tensor = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
        uncond_llava_item_tensor = self.llava_transform(Image.fromarray(uncond_llava_item_image))
        cat_item_tensor = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0
        batch = {
            "pixel_value_llava": llava_item_tensor.unsqueeze(0),
            "uncond_pixel_value_llava": uncond_llava_item_tensor.unsqueeze(0),
            'pixel_value_ref': cat_item_tensor.unsqueeze(0), 
        }
        return batch


class JsonDataset(object):
    def __init__(self, args):
        self.args = args
        self.data_list = args.input
        self.pos_prompt = args.pos_prompt
        self.input_video = args.input_video
        self.input_scene_video = args.input_scene_video
        self.pad_color = (255, 255, 255)
        self.llava_size = (336, 336)
        self.ref_size = (args.video_size[1], args.video_size[0])
        if self.data_list.endswith('.list'):
            self.data_paths = [line.strip() for line in open(self.data_list, 'r')] if self.data_list is not None else []
        else:
            self.data_paths = [self.data_list]
        
        if self.pos_prompt.endswith('.list'):
            self.pos_prompts = [line.strip() for line in open(self.pos_prompt, 'r')] if self.pos_prompt is not None else []
        else:
            self.pos_prompts = [self.pos_prompt]

        if self.input_video is not None:
            if self.input_video.endswith('.list'):
                self.input_video = [line.strip() for line in open(self.input_video, 'r')] if self.data_list is not None else []
            else:
                self.input_video = [self.input_video]

        if self.input_scene_video is not None:
            if self.input_scene_video.endswith('.list'):
                self.input_scene_video = [line.strip() for line in open(self.input_scene_video, 'r')] if self.data_list is not None else []
            else:
                self.input_scene_video = [self.input_scene_video]

        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(self.llava_size, interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        self.sample_size = [480, 480] # hard-code now
        assert self.sample_size[0] == self.sample_size[1]
        if self.sample_size[0] < 540:
            self.buckets = self.generate_crop_size_list(base_size=self.sample_size[0])
        else:
            self.buckets = self.generate_crop_size_list(base_size=self.sample_size[0], patch_size=32)
        self.aspect_ratios = np.array([float(w) / float(h) for w, h in self.buckets])
        print(f"Multi-aspect-ratio bucket num: {len(self.buckets)}")
        
    def __len__(self):
        return len(self.data_paths)
    
    # def read_image(self, image_path):
    #     if isinstance(image_path, dict):
    #         image_path = image_path['seg_item_image_path']

    #     try:
    #         face_image_masked = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    #     except:
    #         face_image_masked = Image.open(image_path).convert('RGB')

    #     cat_face_image = pad_image(face_image_masked.copy(), self.ref_size)
    #     llava_face_image = pad_image(face_image_masked.copy(), self.llava_size)
    #     return llava_face_image, cat_face_image

    def read_image(self, image_path, bg_color=128):
        if isinstance(image_path, dict):
            image_path = image_path['seg_item_image_path']

        tmp_image = Image.open(image_path)

        # random image background augmentation
        if tmp_image.mode == 'RGBA':
            tmp_image = np.array(tmp_image, dtype=np.float32)
            tmp_image, mask = tmp_image[..., 0:3], tmp_image[..., 3:4]
            mask = np.where(mask > 0, 1.0, 0.0)
            tmp_image = (tmp_image * mask + bg_color * (1 - mask)).astype(np.uint8)
        elif tmp_image.mode == 'RGB':
            tmp_image = np.array(tmp_image)
        else:
            raise Exception(f"Unsupported image mode: {tmp_image.mode}. RGBA or RGB modes are required.")
        
        cat_tmp_image = pad_image(tmp_image.copy(), self.ref_size)
        llava_tmp_image = pad_image(tmp_image.copy(), self.llava_size)

        return llava_tmp_image, cat_tmp_image

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        # data_name = os.path.basename(os.path.splitext(data_path)[0])
        data_name = data_path.split('/')[-2]
        if os.path.isfile(data_path): # single image file
            if data_path.endswith('.json'):
                data = json.load(open(data_path, 'r'))
                llava_item_image, cat_item_image = self.read_image(data)
                item_prompt = data['item_prompt']
                seed = data['seed']
                prompt = data['prompt']
                if 'negative_prompt' in data:
                    negative_prompt = data['negative_prompt']
                else:
                    negative_prompt = ''
            else: # single image file
                llava_item_image, cat_item_image = self.read_image(data_path)
                #item_prompt = 'object'
                item_prompt = 'character'
                seed = self.args.seed
                # prompt = self.args.pos_prompt
                # negative_prompt = self.args.neg_prompt
                prompt = self.pos_prompts[idx] # 250809, for batch inference
                negative_prompt = self.args.neg_prompt
                
            llava_item_tensor = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
            cat_item_tensor = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0

            uncond_llava_item_image = np.ones_like(llava_item_image) * 255
            uncond_llava_item_tensor = self.llava_transform(Image.fromarray(uncond_llava_item_image))
        elif os.path.isdir(data_path): # multi-view images folder, directly load images from the original rendered folder
            item_prompt = 'character'
            seed = self.args.seed
            # prompt = self.args.pos_prompt
            # negative_prompt = self.args.neg_prompt
            prompt = self.pos_prompts[idx] # 250809, for batch inference
            negative_prompt = self.args.neg_prompt
            for v in self.args.multi_view_ids:
                llava_item_image, cat_item_image = self.read_image(f"{data_path}/{v:03d}_0001.png")

                llava_item_tensor_tmp = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
                cat_item_tensor_tmp = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0

                uncond_llava_item_tensor_tmp = self.llava_transform(Image.fromarray(np.ones_like(llava_item_image) * 255))

                try:
                    cat_item_tensor = torch.cat([cat_item_tensor, cat_item_tensor_tmp.unsqueeze(0)], dim=0)
                    llava_item_tensor = torch.cat([llava_item_tensor, llava_item_tensor_tmp.unsqueeze(0)], dim=0)
                    uncond_llava_item_tensor = torch.cat([uncond_llava_item_tensor, uncond_llava_item_tensor_tmp.unsqueeze(0)], dim=0)
                except:
                    cat_item_tensor = cat_item_tensor_tmp.unsqueeze(0)
                    llava_item_tensor = llava_item_tensor_tmp.unsqueeze(0)
                    uncond_llava_item_tensor = uncond_llava_item_tensor_tmp.unsqueeze(0)

        # # 250625 preprocess input video condition
        # if self.input_video is not None:
        #     input_video_path = self.input_video[idx]
        #     video_reader = self.request_ceph_data(input_video_path)

        #     fps = video_reader.get_avg_fps()

        #     stride = 1
        #     self.use_stride = True # HunyuanVideo-I2V hyvideo/hyvae_extract/vae.yaml default is True
        #     if self.use_stride: # HunyuanVideo-I2V hyvideo/hyvae_extract/vae.yaml default is True
        #         if int(fps) >= 50:
        #             stride = 2
        #         else:
        #             stride = 1
        #     else:
        #         stride = 1
            
        #     video_length = len(video_reader)
        #     vae_time_compression_ratio = 4# models/vae_3d/hyvae_v1_0801/config.json default vae_time_compression_ratio=4
        #     if video_length < self.args.sample_n_frames*stride:
        #         sample_n_frames = video_length - (video_length - 1) % (vae_time_compression_ratio*stride)  # 4n+1/8n+1
        #     else:
        #         sample_n_frames = self.args.sample_n_frames*stride
            
        #     start_idx = 0
        #     batch_index = list(range(start_idx, start_idx + sample_n_frames, stride))

        #     # 20250322 pftq: fixed to return 5 values for consistency and "not enough values to unpack" error
        #     # copy from https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/blob/1481c1d5ae88e9905f54f2a3c6a1b68ef2a10528/hyvideo/hyvae_extract/dataset.py#L176
        #     if len(batch_index) == 0:
        #         print(f"get video len=0, skip for {input_video_path}")
        #         raise Exception("Invalid input video.")

        #     # Read frames
        #     try:
        #         video_images = video_reader.get_batch(batch_index).asnumpy()
        #     except Exception as e:
        #         print(f'Error: {e}, video_path: {input_video_path}')
        #         raise
        #     pixel_value_input_video = torch.from_numpy(video_images).permute(0, 3, 1, 2).contiguous()
        #     del video_reader

        #     # print(llava_item_tensor.shape, cat_item_tensor.shape)
        #     # raise ValueError
        #     batch = {
        #         "pixel_value_llava": llava_item_tensor,
        #         "uncond_pixel_value_llava": uncond_llava_item_tensor,
        #         "pixel_value_ref": cat_item_tensor,
        #         "prompt": prompt,
        #         "negative_prompt": negative_prompt,
        #         "seed": seed,
        #         "name": item_prompt,
        #         'data_name': data_name,
        #         "pixel_value_input_video": pixel_value_input_video,
        #     }
        #     return batch

        # print(llava_item_tensor.shape, cat_item_tensor.shape)
        # raise ValueError
        batch = {
            "pixel_value_llava": llava_item_tensor,
            "uncond_pixel_value_llava": uncond_llava_item_tensor,
            "pixel_value_ref": cat_item_tensor,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "name": item_prompt,
            'data_name': data_name
        }

        if self.input_video is not None:
            input_video_path = self.input_video[idx]
            video_reader = self.request_ceph_data(input_video_path)

            fps = video_reader.get_avg_fps()

            stride = 1
            self.use_stride = True # HunyuanVideo-I2V hyvideo/hyvae_extract/vae.yaml default is True
            if self.use_stride: # HunyuanVideo-I2V hyvideo/hyvae_extract/vae.yaml default is True
                if int(fps) >= 50:
                    stride = 2
                else:
                    stride = 1
            else:
                stride = 1
            
            video_length = len(video_reader)
            vae_time_compression_ratio = 4# models/vae_3d/hyvae_v1_0801/config.json default vae_time_compression_ratio=4
            if video_length < self.args.sample_n_frames*stride:
                sample_n_frames = video_length - (video_length - 1) % (vae_time_compression_ratio*stride)  # 4n+1/8n+1
            else:
                sample_n_frames = self.args.sample_n_frames*stride
            
            start_idx = 0
            batch_index = list(range(start_idx, start_idx + sample_n_frames, stride))

            # 20250322 pftq: fixed to return 5 values for consistency and "not enough values to unpack" error
            # copy from https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/blob/1481c1d5ae88e9905f54f2a3c6a1b68ef2a10528/hyvideo/hyvae_extract/dataset.py#L176
            if len(batch_index) == 0:
                print(f"get video len=0, skip for {input_video_path}")
                raise Exception("Invalid input video.")

            # Read frames
            try:
                video_images = video_reader.get_batch(batch_index).asnumpy()
            except Exception as e:
                print(f'Error: {e}, video_path: {input_video_path}')
                raise
            pixel_value_input_video = torch.from_numpy(video_images).permute(0, 3, 1, 2).contiguous()
            del video_reader

            # print(llava_item_tensor.shape, cat_item_tensor.shape)
            # raise ValueError
            batch["pixel_value_input_video"] = pixel_value_input_video
        
        # 250723 preprocess input scene video condition
        if self.input_scene_video is not None:
            input_scene_video_path = self.input_scene_video[idx]
            video_reader = self.request_ceph_data(input_scene_video_path)

            fps = video_reader.get_avg_fps()

            stride = 1
            self.use_stride = True # HunyuanVideo-I2V hyvideo/hyvae_extract/vae.yaml default is True
            if self.use_stride: # HunyuanVideo-I2V hyvideo/hyvae_extract/vae.yaml default is True
                if int(fps) >= 50:
                    stride = 2
                else:
                    stride = 1
            else:
                stride = 1
            
            video_length = len(video_reader)
            vae_time_compression_ratio = 4# models/vae_3d/hyvae_v1_0801/config.json default vae_time_compression_ratio=4
            if video_length < self.args.sample_n_frames*stride:
                sample_n_frames = video_length - (video_length - 1) % (vae_time_compression_ratio*stride)  # 4n+1/8n+1
            else:
                sample_n_frames = self.args.sample_n_frames*stride
            
            start_idx = 0
            batch_index = list(range(start_idx, start_idx + sample_n_frames, stride))

            # 20250322 pftq: fixed to return 5 values for consistency and "not enough values to unpack" error
            # copy from https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/blob/1481c1d5ae88e9905f54f2a3c6a1b68ef2a10528/hyvideo/hyvae_extract/dataset.py#L176
            if len(batch_index) == 0:
                print(f"get video len=0, skip for {input_video_path}")
                raise Exception("Invalid input video.")

            # Read frames
            try:
                video_images = video_reader.get_batch(batch_index).asnumpy()
            except Exception as e:
                print(f'Error: {e}, video_path: {input_video_path}')
                raise
            pixel_value_input_video = torch.from_numpy(video_images).permute(0, 3, 1, 2).contiguous()
            del video_reader

            height, width = pixel_value_input_video.shape[-2:]
            bw, bh = self.get_closest_ratio(width=width, 
                                            height=height, ratios=self.aspect_ratios, buckets=self.buckets)
            sample_size = bh, bw
            target_size = self.get_target_size(pixel_value_input_video, sample_size)
            train_crop = transforms.CenterCrop(sample_size) # if self.is_center_crop else transforms.RandomCrop(sample_size)

            pixel_value_input_video = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(pixel_value_input_video)

            y1 = max(0, int(round((height - sample_size[0]) / 2.0)))
            x1 = max(0, int(round((width - sample_size[1]) / 2.0)))
            pixel_value_input_video = train_crop(pixel_value_input_video)

            # print(llava_item_tensor.shape, cat_item_tensor.shape)
            # raise ValueError
            batch["pixel_value_input_scene_video"] = pixel_value_input_video
        
        return batch
    
    def request_ceph_data(self, path):
        try:
            video_reader = VideoReader(path)
        except Exception as e:
            print(f"Error: {e}")
            raise
        return video_reader

    @staticmethod
    def get_closest_ratio(width: float, height: float, ratios: list, buckets: list):
        aspect_ratio = float(width) / float(height)
        closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
        return buckets[closest_ratio_id]

    @staticmethod
    def generate_crop_size_list(base_size=256, patch_size=16, max_ratio=4.0):
        num_patches = round((base_size / patch_size) ** 2)
        assert max_ratio >= 1.
        crop_size_list = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                crop_size_list.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return crop_size_list

    def get_target_size(self, frames, target_size):
        T, C, H, W = frames.shape
        th, tw = target_size
        r = max(th / H, tw / W)
        target_size = int(H * r), int(W * r)
        return target_size


# 250724, copy from HYCustom-V2V
from hymm_sp.data_kits.dwpose import DWposeDetector, draw_pose
from hymm_sp.constants_train import ANNOTATOR_PATH
import pdb

class BaseDataset(object):
    def __init__(self, args):
        self.args = args
        self.data_list = args.input
        self.input_video = args.input_video
        self.input_scene_video = args.input_scene_video 
        self.input_scene_mask_video = args.input_scene_mask_video 
        self.pos_prompt = args.pos_prompt 
        self.output_video = args.output_video 
        self.pad_color = (255, 255, 255)
        self.llava_size = (336, 336)
        self.ref_size = (args.video_size[1], args.video_size[0])
        if self.data_list.endswith('.list'):
            self.data_paths = [line.strip() for line in open(self.data_list, 'r')] if self.data_list is not None else []
        else:
            self.data_paths = [self.data_list]

        if self.input_video is not None:
            if self.input_video.endswith('.list'):
                self.input_video = [line.strip() for line in open(self.input_video, 'r')] if self.data_list is not None else []
            else:
                self.input_video = [self.input_video]

        if self.input_scene_video is not None:
            if self.input_scene_video.endswith('.list'):
                self.input_scene_video = [line.strip() for line in open(self.input_scene_video, 'r')] if self.data_list is not None else []
            else:
                self.input_scene_video = [self.input_scene_video]
        
        if self.input_scene_mask_video is not None:
            if self.input_scene_mask_video.endswith('.list'):
                self.input_scene_mask_video = [line.strip() for line in open(self.input_scene_mask_video, 'r')] if self.data_list is not None else []
            else:
                self.input_scene_mask_video = [self.input_scene_mask_video]
        
        if self.pos_prompt.endswith('.list'):
            self.pos_prompts = [line.strip() for line in open(self.pos_prompt, 'r')] if self.pos_prompt is not None else []
        else:
            self.pos_prompts = [self.pos_prompt]
        
        if self.output_video is not None:
            if self.output_video.endswith('.list'):
                self.output_video = [line.strip() for line in open(self.output_video, 'r')] if self.output_video is not None else []
            else:
                self.output_video = [self.output_video]

        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(self.llava_size, interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        self.is_center_crop = True
        self.sample_size = [480, 480] # hard-code now
        if self.args.bucket_sample_size is not None:
            self.sample_size = [self.args.bucket_sample_size[0], self.args.bucket_sample_size[1]]
        assert self.sample_size[0] == self.sample_size[1]
        if self.sample_size[0] < 540:
            self.buckets = self.generate_crop_size_list(base_size=self.sample_size[0])
        else:
            self.buckets = self.generate_crop_size_list(base_size=self.sample_size[0], patch_size=32)
        self.aspect_ratios = np.array([float(w) / float(h) for w, h in self.buckets])
        print(f"Multi-aspect-ratio bucket num: {len(self.buckets)}")
        
    def __len__(self):
        return len(self.data_paths)

    def read_image(self, image_path, bg_color=128):
        if isinstance(image_path, dict):
            image_path = image_path['seg_item_image_path']
        
        tmp_image = Image.open(image_path)

        # random image background augmentation
        if tmp_image.mode == 'RGBA':
            tmp_image = np.array(tmp_image, dtype=np.float32)
            tmp_image, mask = tmp_image[..., 0:3], tmp_image[..., 3:4]
            mask = np.where(mask > 0, 1.0, 0.0)
            tmp_image = (tmp_image * mask + bg_color * (1 - mask)).astype(np.uint8)
        elif tmp_image.mode == 'RGB':
            tmp_image = np.array(tmp_image)
        else:
            raise Exception(f"Unsupported image mode: {tmp_image.mode}. RGBA or RGB modes are required.")
        
        cat_tmp_image = pad_image(tmp_image.copy(), self.ref_size)
        llava_tmp_image = pad_image(tmp_image.copy(), self.llava_size)

        return llava_tmp_image, cat_tmp_image
    
    def get_batch(self, idx):
        data_path = self.data_paths[idx]
        
        # data_name = os.path.basename(os.path.splitext(data_path)[0])
        data_name = data_path.split("/")[-2]
        if os.path.isfile(data_path): # single image file
            if data_path.endswith('.json'):
                data = json.load(open(data_path, 'r'))
                llava_item_image, cat_item_image = self.read_image(data)
                item_prompt = data['item_prompt']
                seed = data['seed']
                prompt = data['prompt']
                if 'negative_prompt' in data:
                    negative_prompt = data['negative_prompt']
                else:
                    negative_prompt = ''
            else: # single image file
                llava_item_image, cat_item_image = self.read_image(data_path)
                #item_prompt = 'object'
                item_prompt = 'character'
                seed = self.args.seed
                # prompt = self.args.pos_prompt
                prompt = self.pos_prompts[idx] # 251009, for batch inference
                negative_prompt = self.args.neg_prompt
                
            llava_item_tensor = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
            cat_item_tensor = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0

            uncond_llava_item_image = np.ones_like(llava_item_image) * 255
            uncond_llava_item_tensor = self.llava_transform(Image.fromarray(uncond_llava_item_image))
        elif os.path.isdir(data_path): # multi-view images folder, directly load images from the original rendered folder
            item_prompt = 'character'
            seed = self.args.seed
            # prompt = self.args.pos_prompt
            prompt = self.pos_prompts[idx]
            negative_prompt = self.args.neg_prompt
            for v in self.args.multi_view_ids:
                llava_item_image, cat_item_image = self.read_image(f"{data_path}/{v:03d}_0001.png")

                llava_item_tensor_tmp = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
                cat_item_tensor_tmp = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0

                uncond_llava_item_tensor_tmp = self.llava_transform(Image.fromarray(np.ones_like(llava_item_image) * 255))

                try:
                    cat_item_tensor = torch.cat([cat_item_tensor, cat_item_tensor_tmp.unsqueeze(0)], dim=0)
                    llava_item_tensor = torch.cat([llava_item_tensor, llava_item_tensor_tmp.unsqueeze(0)], dim=0)
                    uncond_llava_item_tensor = torch.cat([uncond_llava_item_tensor, uncond_llava_item_tensor_tmp.unsqueeze(0)], dim=0)
                except:
                    cat_item_tensor = cat_item_tensor_tmp.unsqueeze(0)
                    llava_item_tensor = llava_item_tensor_tmp.unsqueeze(0)
                    uncond_llava_item_tensor = uncond_llava_item_tensor_tmp.unsqueeze(0)

        # print(llava_item_tensor.shape, cat_item_tensor.shape)
        # raise ValueError
        batch = {
            "pixel_value_llava": llava_item_tensor,
            "uncond_pixel_value_llava": uncond_llava_item_tensor,
            "pixel_value_ref": cat_item_tensor,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "name": item_prompt,
            'data_name': data_name
        }
        if self.output_video is not None:
            batch["output_video"] = self.output_video[idx]

        if self.input_video is not None and os.path.basename(os.path.splitext(self.input_video[idx])[0]) != "null":
            input_video_path = self.input_video[idx]
            video_reader = self.request_ceph_data(input_video_path)

            fps = video_reader.get_avg_fps()

            stride = 1
            self.use_stride = True # HunyuanVideo-I2V hyvideo/hyvae_extract/vae.yaml default is True
            if self.use_stride: # HunyuanVideo-I2V hyvideo/hyvae_extract/vae.yaml default is True
                if int(fps) >= 50:
                    stride = 2
                else:
                    stride = 1
            else:
                stride = 1
            
            video_length = len(video_reader)
            vae_time_compression_ratio = 4# models/vae_3d/hyvae_v1_0801/config.json default vae_time_compression_ratio=4
            if video_length < self.args.sample_n_frames*stride:
                sample_n_frames = video_length - (video_length - 1) % (vae_time_compression_ratio*stride)  # 4n+1/8n+1
            else:
                sample_n_frames = self.args.sample_n_frames*stride
            
            start_idx = 0
            batch_index = list(range(start_idx, start_idx + sample_n_frames, stride))

            # 20250322 pftq: fixed to return 5 values for consistency and "not enough values to unpack" error
            # copy from https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/blob/1481c1d5ae88e9905f54f2a3c6a1b68ef2a10528/hyvideo/hyvae_extract/dataset.py#L176
            if len(batch_index) == 0:
                print(f"get video len=0, skip for {input_video_path}")
                raise Exception("Invalid input video.")

            # Read frames
            try:
                video_images = video_reader.get_batch(batch_index).asnumpy()
            except Exception as e:
                print(f'Error: {e}, video_path: {input_video_path}')
                raise
            pixel_value_input_video = torch.from_numpy(video_images).permute(0, 3, 1, 2).contiguous()
            del video_reader

            # print(llava_item_tensor.shape, cat_item_tensor.shape)
            # raise ValueError
            batch["pixel_value_input_video"] = pixel_value_input_video
        
        return batch

    def __getitem__(self, idx):
        return self.get_batch(idx)
    
    def request_ceph_data(self, path):
        try:
            video_reader = VideoReader(path)
        except Exception as e:
            print(f"Error: {e}")
            raise
        return video_reader

    @staticmethod
    def get_closest_ratio(width: float, height: float, ratios: list, buckets: list):
        aspect_ratio = float(width) / float(height)
        closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
        return buckets[closest_ratio_id]

    @staticmethod
    def generate_crop_size_list(base_size=256, patch_size=16, max_ratio=4.0):
        num_patches = round((base_size / patch_size) ** 2)
        assert max_ratio >= 1.
        crop_size_list = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                crop_size_list.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return crop_size_list

    def get_target_size(self, frames, target_size):
        T, C, H, W = frames.shape
        th, tw = target_size
        r = max(th / H, tw / W)
        target_size = int(H * r), int(W * r)
        return target_size

class VideoDataset(BaseDataset):
    def __init__(self, args, device='cuda'):
        # super().__init__(args, device)
        super().__init__(args)
        self.expand_scale = int(args.expand_scale)
        self.pose_enhance = args.pose_enhance
        if self.pose_enhance:
            self.dwpose_detector = DWposeDetector(
                model_det=os.path.join(ANNOTATOR_PATH['dwpose'], "yolox_l.onnx"),
                model_pose=os.path.join(ANNOTATOR_PATH['dwpose'], "dw-ll_ucoco_384.onnx"),
                device=device
            )
        else:
            self.dwpose_detector = None

    def __getitem__(self, idx):
        if self.input_scene_video is not None:
            video_path = self.input_scene_video[idx]
            input_video = VideoReader(video_path)
        else:
            raise NotImplementedError
        if self.input_scene_mask_video is not None:
            mask_path = self.input_scene_mask_video[idx]
            mask_video = VideoReader(mask_path)
            num_frames = min(len(input_video), len(mask_video))
        else:
            mask_video = None
            num_frames = len(input_video)
        height, width = input_video[0].asnumpy().shape[:2]
        
        masked_frames = []
        masks = []
        for frame_idx in range(num_frames):
            frame = input_video[frame_idx].asnumpy()
            if mask_video is not None:
                mask = mask_video[frame_idx].asnumpy()
            else:
                mask = np.ones_like(frame) * 255
            mask = cv2.resize(mask, (width, height))
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if self.expand_scale != 0:
                kernel_size = abs(self.expand_scale)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                op_expand = cv2.dilate if self.expand_scale > 0 else cv2.erode
                mask = op_expand(mask, kernel, iterations=3)
            mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
            masks.append(mask)

            if self.args.scene_mask_type == "MASKED":
                # default in HYCustom-V2V, for masked scene latents
                inverse_mask = mask == 0
                if self.dwpose_detector:
                    pose_img = draw_pose(self.dwpose_detector(frame), height, width, ref_w=576)
                    masked_frame = np.where(inverse_mask[..., None], frame, pose_img)
                else:
                    masked_frame = frame * (inverse_mask[..., None].astype(frame.dtype))
                masked_frames.append(masked_frame)
                if frame_idx == 0:
                    print("Use MASKED scene.")

            elif self.args.scene_mask_type == "INPAINTED":
                # DO NOT mask corresponding area, for inpainted scene latents
                masked_frames.append(frame)
                if frame_idx == 0:
                    print("Use INPAINTED scene.")
            
            else:
                # default in HYCustom-V2V, for masked scene latents
                inverse_mask = mask == 0
                if self.dwpose_detector:
                    pose_img = draw_pose(self.dwpose_detector(frame), height, width, ref_w=576)
                    masked_frame = np.where(inverse_mask[..., None], frame, pose_img)
                else:
                    masked_frame = frame * (inverse_mask[..., None].astype(frame.dtype))
                masked_frames.append(masked_frame)
                if frame_idx == 0:
                    print("DO NOT set scene_mask_type, default to MASKED.")

        masks_tensor = torch.from_numpy(np.asarray(masks)).unsqueeze(0).repeat_interleave(3, dim=0) / 255.0
        masked_frames_tensor = torch.from_numpy(np.asarray(masked_frames)).permute((3, 0, 1, 2)) / 255.0
        
        for i, frames in enumerate([masks_tensor, masked_frames_tensor]):
            frames = frames.permute((1, 0, 2, 3))
            bw, bh = self.get_closest_ratio(width=width, height=height, ratios=self.aspect_ratios, buckets=self.buckets)
            sample_size = bh, bw
            target_size = self.get_target_size(frames, sample_size)
            train_crop = transforms.CenterCrop(sample_size) if self.is_center_crop else transforms.RandomCrop(sample_size)

            frames = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(frames)

            y1 = max(0, int(round((height - sample_size[0]) / 2.0)))
            x1 = max(0, int(round((width - sample_size[1]) / 2.0)))
            frames = train_crop(frames)

            if i == 0:
                masks_tensor = frames.permute((1, 0, 2, 3))
            elif i == 1:
                masked_frames_tensor = frames.permute((1, 0, 2, 3))
        self.ref_size = (masked_frames_tensor.shape[-1], masked_frames_tensor.shape[-2])

        base_batch = self.get_batch(idx)
        batch = {
            **base_batch,
            "pixel_value_bg": masked_frames_tensor,
            "pixel_value_mask": masks_tensor,
        }
        return batch
    
    @staticmethod
    def get_closest_ratio(width: float, height: float, ratios: list, buckets: list):
        aspect_ratio = float(width) / float(height)
        closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
        return buckets[closest_ratio_id]
    
    @staticmethod
    def generate_crop_size_list(base_size=256, patch_size=16, max_ratio=4.0):
        num_patches = round((base_size / patch_size) ** 2)
        assert max_ratio >= 1.
        crop_size_list = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                crop_size_list.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return crop_size_list

    def get_target_size(self, frames, target_size):
        T, C, H, W = frames.shape
        th, tw = target_size
        r = max(th / H, tw / W)
        target_size = int(H * r), int(W * r)
        return target_size