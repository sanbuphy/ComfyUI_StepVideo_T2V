import os
import torch
import numpy as np
import folder_paths
from PIL import Image
import cv2
import tempfile
import shutil

# Try to import required libraries
try:
    from modelscope import snapshot_download
    from diffsynth import ModelManager, StepVideoPipeline, save_video
except ImportError:
    raise ImportError("Please install required dependencies: pip install modelscope diffsynth")

# Get ComfyUI model directories
CHECKPOINT_DIR = folder_paths.get_folder_paths("checkpoints")[0]
DEFAULT_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "stepfun-ai", "stepvideo-t2v-turbo")

# Register the model path with ComfyUI
if not os.path.exists(DEFAULT_MODEL_PATH):
    os.makedirs(DEFAULT_MODEL_PATH, exist_ok=True)
folder_paths.add_model_folder_path("stepvideo", DEFAULT_MODEL_PATH)

class StepVideoModelLoaderNode:
    """
    StepVideo Model Downloader and Loader - Download (if needed) and load the StepVideo T2V model
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_path": ("STRING", {"default": DEFAULT_MODEL_PATH}),
            }
        }
    
    RETURN_TYPES = ("STEPVIDEO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "download_and_load_model"
    CATEGORY = "StepVideo"
    
    def __init__(self):
        self.model_manager = None
        self.pipe = None
        
    def download_and_load_model(self, model_path):
        """Download (if needed) and load StepVideo model"""
        print(f"attempting to download...")
        
        # Create directory (if it doesn't exist)
        os.makedirs(model_path, exist_ok=True)
        
        # 下载模型
        snapshot_download(model_id="stepfun-ai/stepvideo-t2v-turbo", cache_dir=CHECKPOINT_DIR)
        
        print("StepVideo T2V model downloaded")
        print("Loading StepVideo T2V model...")
        
        
        print(f"now loading compiled attention mechanism... {model_path}")
        # 尝试加载编译的注意力机制
        try:
            compiled_lib = os.path.join(model_path, "lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so")
            if os.path.exists(compiled_lib):
                torch.ops.load_library(compiled_lib)
        except Exception as e:
            print(f"Error loading compiled attention mechanism, file may not be downloaded: {e}")
            print("Continuing with non-compiled version...")
        
        # Load models
        model_manager = ModelManager()
        print('actual model path:', model_path)
        model_manager.load_models(
            [os.path.join(model_path, "hunyuan_clip/clip_text_encoder/pytorch_model.bin")],
            torch_dtype=torch.float32, device="cpu"
        )
        model_manager.load_models(
            [
                os.path.join(model_path, "step_llm"),
                [
                    os.path.join(model_path, "transformer/diffusion_pytorch_model-00001-of-00006.safetensors"),
                    os.path.join(model_path, "transformer/diffusion_pytorch_model-00002-of-00006.safetensors"),
                    os.path.join(model_path, "transformer/diffusion_pytorch_model-00003-of-00006.safetensors"),
                    os.path.join(model_path, "transformer/diffusion_pytorch_model-00004-of-00006.safetensors"),
                    os.path.join(model_path, "transformer/diffusion_pytorch_model-00005-of-00006.safetensors"),
                    os.path.join(model_path, "transformer/diffusion_pytorch_model-00006-of-00006.safetensors"),
                ]
            ],
            torch_dtype=torch.bfloat16, device="cpu" # You can set torch_dtype=torch.bfloat16 to reduce RAM (not VRAM) usage.
        )
        model_manager.load_models(
            [os.path.join(model_path, "vae/vae_v2.safetensors")],
            torch_dtype=torch.bfloat16, device="cpu"
        )
        
        # Create pipeline
        pipe = StepVideoPipeline.from_model_manager(
            model_manager, 
            torch_dtype=torch.bfloat16, 
            device="cuda"
        )
        
        # Enable VRAM management
        pipe.enable_vram_management(num_persistent_param_in_dit=0)
        
        print("StepVideo T2V models loaded successfully!")
        
        return (pipe,)


class StepVideoT2VNode:
    """
    StepVideo Text to Video - Generate videos from text prompts using a loaded model
    
    Inputs:
    - model: The StepVideo model loaded by StepVideoModelLoader
    - prompt: Text prompt describing what you want to see in the video (can be connected from external text nodes)
    - negative_prompt: Text prompt describing what you don't want to see (can be connected from external text nodes)
    - Other parameters: Control various aspects of the generation process
    
    Outputs:
    - video: The generated video that can be passed to SaveVideo node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STEPVIDEO_MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 9.0, "min": 1.0, "max": 20.0}),
                "num_frames": ("INT", {"default": 51, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "tiled": ("BOOLEAN", {"default": True}),
                "tile_size_x": ("INT", {"default": 34, "min": 8, "max": 128}),
                "tile_size_y": ("INT", {"default": 34, "min": 8, "max": 128}),
                "tile_stride_x": ("INT", {"default": 16, "min": 4, "max": 64}),
                "tile_stride_y": ("INT", {"default": 16, "min": 4, "max": 64}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_video"
    CATEGORY = "StepVideo"
    
    def generate_video(self, model, prompt, negative_prompt, num_inference_steps, cfg_scale, 
                      num_frames, seed, tiled, tile_size_x, tile_size_y, 
                      tile_stride_x, tile_stride_y, fps):
        """Generate video"""
        # Set random seed
        torch.manual_seed(seed)
        
        # Set tiling parameters
        tile_size = (tile_size_x, tile_size_y) if tiled else None
        tile_stride = (tile_stride_x, tile_stride_y) if tiled else None
        
        print(f"Generating video with prompt: {prompt}")
        print(f"Negative prompt: {negative_prompt}")
        
        # Generate video
        video = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            num_frames=num_frames,
            seed=seed,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride
        )
        
        print("Video generation completed!")
        
        return (video,)


class SaveVideoNode:
    """
    Save Video - Save a video to a specified location with custom settings
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename": ("STRING", {"default": "output_video"}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "quality": ("INT", {"default": 5, "min": 1, "max": 10}),
                "denoise": ("BOOLEAN", {"default": True}),
                "output_dir": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"
    CATEGORY = "StepVideo"
    
    def save_video(self, video, filename, fps, quality, denoise, output_dir=""):
        """Save video to file"""
        # Determine output directory
        if output_dir and os.path.isdir(output_dir):
            save_dir = output_dir
        else:
            save_dir = os.path.join(folder_paths.get_output_directory(), "stepvideo")
            os.makedirs(save_dir, exist_ok=True)
        
        # Ensure filename has .mp4 extension
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        
        # Create full path
        video_path = os.path.join(save_dir, filename)
        
        # Set ffmpeg parameters
        ffmpeg_params = []
        if denoise:
            ffmpeg_params = ["-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1"]
        
        # Save video
        print(f"Saving video to: {video_path}")
        save_video(
            video, 
            video_path, 
            fps=fps, 
            quality=quality,
            ffmpeg_params=ffmpeg_params
        )
        
        print(f"Video saved successfully to: {video_path}")
        
        return (video_path,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "StepVideoModelLoader": StepVideoModelLoaderNode,
    "StepVideoT2V": StepVideoT2VNode,
    "SaveVideo": SaveVideoNode
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "StepVideoModelLoader": "StepVideo Model Downloader & Loader",
    "StepVideoT2V": "StepVideo Text to Video",
    "SaveVideo": "Save Video"
}