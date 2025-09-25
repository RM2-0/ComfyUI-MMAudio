import os
import torch
import json
from torchvision.transforms import v2
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

script_directory = os.path.dirname(os.path.abspath(__file__))

if not "mmaudio" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("mmaudio", os.path.join(folder_paths.models_dir, "mmaudio"))

from .mmaudio.eval_utils import generate
from .mmaudio.model.flow_matching import FlowMatching
from .mmaudio.model.networks import MMAudio
from .mmaudio.model.utils.features_utils import FeaturesUtils
from .mmaudio.model.sequence_config import (CONFIG_16K, CONFIG_44K, SequenceConfig)
from .mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from .mmaudio.ext.synchformer import Synchformer
from .mmaudio.ext.autoencoder import AutoEncoderModule
from open_clip import CLIP

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def process_video_tensor(video_tensor: torch.Tensor, duration_sec: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Process video tensor for MMAudio inference following official implementation logic.
    
    Key improvement: When input frames are insufficient, duplicate frames instead of truncating duration.
    This ensures the output audio matches the requested duration, not the video length.
    
    Args:
        video_tensor: Input video tensor with shape (frames, height, width, channels)
        duration_sec: Requested audio duration in seconds
        
    Returns:
        tuple: (clip_frames, sync_frames, preserved_duration_sec)
    """
    # MMAudio model constants - matching official implementation
    _CLIP_SIZE = 384      # CLIP model input resolution
    _CLIP_FPS = 8.0       # CLIP model frame rate
    _SYNC_SIZE = 224      # Synchformer input resolution  
    _SYNC_FPS = 25.0      # Synchformer frame rate

    # Transform pipeline for CLIP frames (384x384 resolution)
    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToPILImage(),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float32),
    ])

    # Transform pipeline for Sync frames (224x224 with normalization)
    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToPILImage(),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    total_frames = video_tensor.shape[0]
    
    # Calculate required frame counts based on target duration
    target_clip_frames = int(_CLIP_FPS * duration_sec)
    target_sync_frames = int(_SYNC_FPS * duration_sec)
    
    log.info(f'Video processing: {total_frames} input frames -> CLIP needs {target_clip_frames}, Sync needs {target_sync_frames}')
    
    def duplicate_frames_to_target(frames: torch.Tensor, target_count: int, fps_name: str) -> torch.Tensor:
        """
        Duplicate frames to reach target count, mimicking official frame rate conversion logic.
        
        Official MMAudio behavior: "For input videos with a frame rate below 25 FPS, 
        frames will be duplicated to match the required rate."
        
        Args:
            frames: Input frames tensor
            target_count: Required number of frames
            fps_name: Name for logging (CLIP/Sync)
            
        Returns:
            Processed frames tensor with target_count frames
        """
        current_count = frames.shape[0]
        
        if current_count >= target_count:
            # Sufficient frames, take first N frames
            return frames[:target_count]
        
        # Insufficient frames - duplicate following official logic
        log.info(f'{fps_name}: Duplicating frames from {current_count} to {target_count}')
        
        # Calculate repetition ratio for frame mapping
        repeat_ratio = target_count / current_count
        
        # Create new frame sequence by intelligent frame selection
        new_frames = []
        for i in range(target_count):
            # Determine which source frame to use for this position
            source_frame_idx = min(int(i / repeat_ratio), current_count - 1)
            new_frames.append(frames[source_frame_idx])
        
        return torch.stack(new_frames)
    
    # Process frames for both CLIP and Sync following official logic
    processed_clip_frames = duplicate_frames_to_target(video_tensor, target_clip_frames, "CLIP")
    processed_sync_frames = duplicate_frames_to_target(video_tensor, target_sync_frames, "Sync")
    
    # Convert tensor dimensions: (frames, height, width, channels) -> (frames, channels, height, width)
    processed_clip_frames = processed_clip_frames.permute(0, 3, 1, 2)
    processed_sync_frames = processed_sync_frames.permute(0, 3, 1, 2)

    # Apply transforms to prepare frames for model input
    clip_frames = torch.stack([clip_transform(frame) for frame in processed_clip_frames])
    sync_frames = torch.stack([sync_transform(frame) for frame in processed_sync_frames])

    log.info(f'Final output: CLIP {clip_frames.shape}, Sync {sync_frames.shape}')
    log.info(f'Duration preserved: {duration_sec}s (unchanged)')

    # CRITICAL: Always return original duration_sec, never modify it!
    # This ensures audio duration matches user request, not video length
    return clip_frames, sync_frames, duration_sec

#region Model loading
class MMAudioModelLoader:
    """
    Load MMAudio model weights and configure model architecture.
    
    Supports different model sizes (small/large) and sampling rates (16k/44k).
    Automatically detects model configuration from filename.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mmaudio_model": (folder_paths.get_filename_list("mmaudio"), 
                    {"tooltip": "These models are loaded from the 'ComfyUI/models/mmaudio' folder"}),
                "base_precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("MMAUDIO_MODEL",)
    RETURN_NAMES = ("mmaudio_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, mmaudio_model, base_precision):
        """Load MMAudio model with specified precision."""
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()

        # Configure data type based on precision setting
        base_dtype = {
            "fp8_e4m3fn": torch.float8_e4m3fn, 
            "fp8_e4m3fn_fast": torch.float8_e4m3fn, 
            "bf16": torch.bfloat16, 
            "fp16": torch.float16, 
            "fp32": torch.float32
        }[base_precision]

        # Load model weights
        mmaudio_model_path = folder_paths.get_full_path_or_raise("mmaudio", mmaudio_model)
        mmaudio_sd = load_torch_file(mmaudio_model_path, device=offload_device)

        # Configure model architecture based on model size
        if "small" in mmaudio_model:
            num_heads = 7
            model = MMAudio(
                latent_dim=40,
                clip_dim=1024,
                sync_dim=768,
                text_dim=1024,
                hidden_dim=64 * num_heads,
                depth=12,
                fused_depth=8,
                num_heads=num_heads,
                latent_seq_len=345,
                clip_seq_len=64,
                sync_seq_len=192
            )
        elif "large" in mmaudio_model:
            num_heads = 14
            model = MMAudio(
                latent_dim=40,
                clip_dim=1024,
                sync_dim=768,
                text_dim=1024,
                hidden_dim=64 * num_heads,
                depth=21,
                fused_depth=14,
                num_heads=num_heads,
                latent_seq_len=345,
                clip_seq_len=64,
                sync_seq_len=192,
                v2=True
            )
        
        # Initialize model and load weights
        model = model.eval().to(device=device, dtype=base_dtype)
        model.load_weights(mmaudio_sd)
        log.info(f'Loaded MMAudio model weights from {mmaudio_model_path}')
        
        # Set sequence configuration based on sampling rate
        if "44" in mmaudio_model:
            model.seq_cfg = CONFIG_44K
        elif "16" in mmaudio_model:
            model.seq_cfg = CONFIG_16K

        return (model,)

#region Features Utils
class MMAudioVoCoderLoader:
    """
    Load BigVGAN vocoder model for 16kHz audio generation.
    Only needed when using 16k models.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vocoder_model": (folder_paths.get_filename_list("mmaudio"), 
                    {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
            },
        }

    RETURN_TYPES = ("VOCODER_MODEL",)
    RETURN_NAMES = ("mmaudio_vocoder", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, vocoder_model):
        """Load BigVGAN vocoder model."""
        from .mmaudio.ext.bigvgan import BigVGAN
        vocoder_model_path = folder_paths.get_full_path_or_raise("mmaudio", vocoder_model)
        vocoder_model = BigVGAN.from_pretrained(vocoder_model_path).eval()
        return (vocoder_model_path,)

class MMAudioFeatureUtilsLoader:
    """
    Load feature extraction models: VAE, Synchformer, and CLIP.
    
    These models are responsible for:
    - VAE: Audio encoding/decoding
    - Synchformer: Video synchronization features
    - CLIP: Visual semantic features
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_model": (folder_paths.get_filename_list("mmaudio"), 
                    {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "synchformer_model": (folder_paths.get_filename_list("mmaudio"), 
                    {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "clip_model": (folder_paths.get_filename_list("mmaudio"), 
                    {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
            },
            "optional": {
                "bigvgan_vocoder_model": ("VOCODER_MODEL", 
                    {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "mode": (["16k", "44k"], {"default": "44k"}),
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("MMAUDIO_FEATUREUTILS",)
    RETURN_NAMES = ("mmaudio_featureutils", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, vae_model, precision, synchformer_model, clip_model, mode, bigvgan_vocoder_model=None):
        """Load all feature extraction models."""
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Load Synchformer for video synchronization features
        synchformer_path = folder_paths.get_full_path_or_raise("mmaudio", synchformer_model)
        synchformer_sd = load_torch_file(synchformer_path, device=offload_device)
        synchformer = Synchformer()
        synchformer.load_state_dict(synchformer_sd)
        synchformer = synchformer.eval().to(device=device, dtype=dtype)

        # Load BigVGAN vocoder (handles both 16k and 44k modes)
        download_path = folder_paths.get_folder_paths("mmaudio")[0]
        nvidia_bigvgan_vocoder_path = os.path.join(download_path, "nvidia", "bigvgan_v2_44khz_128band_512x")
        
        if mode == "44k":
            # Auto-download NVIDIA BigVGAN for 44kHz if not exists
            if not os.path.exists(nvidia_bigvgan_vocoder_path):
                log.info(f"Downloading nvidia bigvgan vocoder model to: {nvidia_bigvgan_vocoder_path}")
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
                    ignore_patterns=["*3m*",],
                    local_dir=nvidia_bigvgan_vocoder_path,
                    local_dir_use_symlinks=False,
                )

            bigvgan_vocoder = BigVGANv2.from_pretrained(nvidia_bigvgan_vocoder_path).eval().to(device=device, dtype=dtype)
        else:
            # Use provided vocoder for 16k mode
            assert bigvgan_vocoder_model is not None, "bigvgan_vocoder_model must be provided for 16k mode"
            bigvgan_vocoder = bigvgan_vocoder_model

        # Load VAE for audio encoding/decoding
        vae_path = folder_paths.get_full_path_or_raise("mmaudio", vae_model)
        vae_sd = load_torch_file(vae_path, device=offload_device)
        vae = AutoEncoderModule(
            vae_state_dict=vae_sd,
            bigvgan_vocoder=bigvgan_vocoder,
            mode=mode
        )
        vae = vae.eval().to(device=device, dtype=dtype)

        # Load CLIP model for visual semantic features
        clip_model_path = folder_paths.get_full_path_or_raise("mmaudio", clip_model)
        clip_config_path = os.path.join(script_directory, "configs", "DFN5B-CLIP-ViT-H-14-384.json")
        
        with open(clip_config_path) as f:
            clip_config = json.load(f)

        # Initialize CLIP model with empty weights first
        with init_empty_weights():
            clip_model = CLIP(**clip_config["model_cfg"]).eval()
        
        # Load and assign weights
        clip_sd = load_torch_file(os.path.join(clip_model_path), device=offload_device)
        for name, param in clip_model.named_parameters():
            set_module_tensor_to_device(clip_model, name, device=device, dtype=dtype, value=clip_sd[name])
        clip_model.to(device=device, dtype=dtype)

        # Create feature utils with all loaded models
        feature_utils = FeaturesUtils(
            vae=vae,
            synchformer=synchformer,
            enable_conditions=True,
            clip_model=clip_model
        )
        return (feature_utils,)

#region sampling
class MMAudioSampler:
    """
    Generate audio from video and/or text using MMAudio model.
    
    This is the main inference node that combines all models to generate
    synchronized audio that matches video content and text prompts.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mmaudio_model": ("MMAUDIO_MODEL",),
                "feature_utils": ("MMAUDIO_FEATUREUTILS",),
                "duration": ("FLOAT", {"default": 8, "step": 0.01, 
                    "tooltip": "Duration of the audio in seconds"}),
                "steps": ("INT", {"default": 25, "step": 1, 
                    "tooltip": "Number of steps to interpolate"}),
                "cfg": ("FLOAT", {"default": 4.5, "step": 0.1, 
                    "tooltip": "Strength of the conditioning"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "mask_away_clip": ("BOOLEAN", {"default": False, 
                    "tooltip": "If true, the clip video will be masked away"}),
                "force_offload": ("BOOLEAN", {"default": True, 
                    "tooltip": "If true, the model will be offloaded to the offload device"}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sample"
    CATEGORY = "MMAudio"

    def sample(self, mmaudio_model, seed, feature_utils, duration, steps, cfg, prompt, negative_prompt, mask_away_clip, force_offload, images=None):
        """
        Generate audio using MMAudio model.
        
        Args:
            mmaudio_model: Loaded MMAudio model
            seed: Random seed for reproducible generation
            feature_utils: Feature extraction utilities
            duration: Target audio duration in seconds
            steps: Number of diffusion steps
            cfg: Classifier-free guidance strength
            prompt: Text description of desired audio
            negative_prompt: Text description of undesired audio
            mask_away_clip: Whether to ignore visual features (text-only mode)
            force_offload: Whether to offload models after generation
            images: Optional video frames as IMAGE tensor
            
        Returns:
            Generated audio in ComfyUI audio format
        """
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Initialize random number generator
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        seq_cfg = mmaudio_model.seq_cfg

        # Process input video if provided
        if images is not None:
            images = images.to(device=device)
            # Use our fixed video processing function
            clip_frames, sync_frames, duration = process_video_tensor(images, duration)
            
            log.info(f"Processed video: clip_frames {clip_frames.shape}, sync_frames {sync_frames.shape}, duration {duration}s")
            
            if mask_away_clip:
                # Text-only mode: ignore visual features
                clip_frames = None
            else:
                # Add batch dimension for CLIP frames
                clip_frames = clip_frames.unsqueeze(0)
            
            # Add batch dimension for sync frames
            sync_frames = sync_frames.unsqueeze(0)
        else:
            # Text-only mode: no video input
            clip_frames = None
            sync_frames = None

        # Update model sequence configuration
        seq_cfg.duration = duration
        mmaudio_model.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        # Initialize flow matching scheduler
        scheduler = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=steps)
        
        # Move models to device for inference
        feature_utils.to(device)
        mmaudio_model.to(device)
        
        # Generate audio using MMAudio
        audios = generate(
            clip_frames,
            sync_frames, 
            [prompt],
            negative_text=[negative_prompt],
            feature_utils=feature_utils,
            net=mmaudio_model,
            fm=scheduler,
            rng=rng,
            cfg_strength=cfg
        )
        
        # Offload models to save memory if requested
        if force_offload:
            mmaudio_model.to(offload_device)
            feature_utils.to(offload_device)
            mm.soft_empty_cache()
        
        # Prepare output in ComfyUI audio format
        waveform = audios.float().cpu()
        audio = {
            "waveform": waveform,
            "sample_rate": 44100  # MMAudio generates 44.1kHz audio
        }

        return (audio,)

# Register all nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "MMAudioModelLoader": MMAudioModelLoader,
    "MMAudioFeatureUtilsLoader": MMAudioFeatureUtilsLoader,
    "MMAudioSampler": MMAudioSampler,
    "MMAudioVoCoderLoader": MMAudioVoCoderLoader,
}

# Define display names for nodes in ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMAudioModelLoader": "MMAudio ModelLoader",
    "MMAudioFeatureUtilsLoader": "MMAudio FeatureUtilsLoader",
    "MMAudioSampler": "MMAudio Sampler",
    "MMAudioVoCoderLoader": "MMAudio VoCoderLoader",
}
