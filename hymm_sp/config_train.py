import argparse
from hymm_sp.constants_train import *
import re
import collections.abc
from .modules.models import HUNYUAN_VIDEO_CONFIG

def as_tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")

def parse_args(mode="eval", namespace=None):
    parser = argparse.ArgumentParser(description="Hunyuan Multimodal training/inference script")
    #parser = add_extra_args(parser)
    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_evaluation_args(parser)

    # new
    parser = add_lora_args(parser)
    #parser = add_inference_args(parser)
    #parser = add_parallel_args(parser)
    if mode == "train":
        parser = add_training_args(parser)
        parser = add_optimizer_args(parser)
        parser = add_deepspeed_args(parser)
        parser = add_data_args(parser)
        parser = add_train_denoise_schedule_args(parser)
    args = parser.parse_args(namespace=namespace)
    args = sanity_check_args(args)
    return args

def add_extra_args(parser: argparse.ArgumentParser):
    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_evaluation_args(parser)
    return parser

def add_train_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule")

    group.add_argument("--flow-path-type", type=str, default="linear", choices=FLOW_PATH_TYPE,
                       help="Path type for flow matching schedulers.")
    group.add_argument("--flow-predict-type", type=str, default="velocity", choices=FLOW_PREDICT_TYPE,
                       help="Prediction type for flow matching schedulers.")
    group.add_argument("--flow-loss-weight", type=str, default=None, choices=FLOW_LOSS_WEIGHT,
                       help="Loss weight type for flow matching schedulers.")
    group.add_argument("--flow-train-eps", type=float, default=None,
                       help="Small epsilon for avoiding instability during training.")
    group.add_argument("--flow-sample-eps", type=float, default=None,
                       help="Small epsilon for avoiding instability during sampling.")
    group.add_argument("--flow-snr-type", type=str, default="lognorm", choices=FLOW_SNR_TYPE,
                       help="Type of SNR to use for flow matching schedulers.")

    return parser

def add_deepspeed_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="DeepSpeed")

    group.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    group.add_argument("--zero-stage", type=int, default=0, choices=[0, 1, 2, 3],
                       help="DeepSpeed ZeRO stage. 0: off, 1: offload optimizer, 2: offload parameters, "
                            "3: offload optimizer and parameters.")
    group.add_argument("--optimizer-cpu-offload", action="store_true", default=False)
    group.add_argument("--param-cpu-offload", action="store_true", default=False)
    return parser

def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Data")

    group.add_argument("--data-type", type=str, default="image", choices=DATA_TYPE, help="Type of the dataset.")
    # group.add_argument("--data-jsons-path", type=str, default=None, help="Dataset path for training.")
    group.add_argument("--data-jsons-path", type=str, nargs='+', default=None,
                        help="Dataset path for training. multiple paths can be provided.")
    # group.add_argument("--sample-n-frames", type=int, default=65,
    #                    help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1")
    group.add_argument("--sample-stride", type=int, default=1,
                       help="How many frames to skip when sampling from a video.")
    group.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading.")
    group.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor for data loading.")
    group.add_argument("--same-data-batch", action="store_true", help="Use same data type for all rank in a batch for training.")
    group.add_argument("--uncond-p", type=float, default=0.1,
                       help="Probability of randomly dropping video description.")
    group.add_argument("--sematic-cond-drop-p", type=float, default=0.1,
                       help="Probability of randomly dropping img condition description.")

    return parser

def add_training_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Training")

    group.add_argument("--task-flag", type=str, required=True,
                       help="Task flag for training/inference. It is used to determine the experiment directory.")
    group.add_argument("--output-dir", type=str, required=True, help="Directory to save logs and models")
    group.add_argument("--sample-dir", type=str, default=None, required=False, help="Directory to save samples")
    group.add_argument("--micro-batch-size", type=int, default=1, nargs='*',
                       help="Batch size per model instance (local batch size).")
    group.add_argument("--video-micro-batch-size", type=int, default=None, nargs='*',
                       help="Batch size per model instance (local batch size).")
    group.add_argument("--global-batch-size", type=int, default=None, nargs='*',
                       help="Global batch size (across all model instances). "
                            "global-batch-size = micro-batch-size * world-size * gradient-accumulation-steps")
    group.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Number of steps to accumulate gradients over before performing an update.")
    group.add_argument("--global-seed", type=int, default=42, help="Global seed for reproducibility.")

    group.add_argument("--resume", type=str, default=None,
                       help="Path to the checkpoint to resume training. It can be an experiment index to resume from "
                            "the latest checkpoint in the output directory.")
    group.add_argument("--init-from", type=str, default=None,
                       help="Path to the checkpoint to load from init ckpt for training. ")
    group.add_argument("--training-parts", type=str, default=None, help="Training a subset of the model parameters.")
    group.add_argument("--init-save", action="store_true", help="Save the initial model before training.")
    group.set_defaults(final_save=True)
    group.add_argument("--final-save", action="store_true", help="Save the final model after training.")
    group.add_argument("--no-final-save", dest="final_save", action="store_false", help="Do not save the final model.")

    group.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    group.add_argument("--max-training-steps", type=int, default=10_000_000, help="Maximum number of training steps.")
    group.add_argument("--ckpt-every", type=int, default=5000, help="Save checkpoint every N steps.")

    group.add_argument("--rope-theta-rescale-factor", type=float, default=1.0, nargs='+',
                       help="Rope interpolation factor.")
    group.add_argument("--rope-interpolation-factor", type=float, default=1.0, nargs='+',
                       help="Rope interpolation factor.")

    group.add_argument("--log-every", type=int, default=10, help="Log every N update steps.")
    group.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
    group.add_argument("--profile", action="store_true", help="Enable PyTorch profiler.")
    return parser

def add_optimizer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Optimizer")

    # Learning rate
    group.add_argument("--lr", type=float, default=1e-4,
                       help="Basic learning rate, varies depending on learning rate schedule and warmup.")
    group.add_argument("--fake-score-lr", type=float, default=4e-7,
                       help="Basic learning rate, varies depending on learning rate schedule and warmup.")
    group.add_argument("--warmup-min-lr", type=float, default=1e-6, help="Minimum learning rate for warmup.")
    group.add_argument("--fake-score-warmup-min-lr", type=float, default=1e-7, help="Minimum learning rate for warmup.")
    group.add_argument("--warmup-num-steps", type=int, default=0, help="Number of warmup steps for learning rate.")

    # Optimizer
    group.add_argument("--adam-beta1", type=float, default=0.9,
                       help="[AdamW] First coefficient for computing running averages of gradient.")
    group.add_argument("--adam-beta2", type=float, default=0.999,
                       help="[AdamW] Second coefficient for computing running averages of gradient square.")
    group.add_argument("--adam-eps", type=float, default=1e-8,
                       help="[AdamW] Term added to the denominator to improve numerical stability.")
    group.add_argument("--weight-decay", type=float, default=0,
                       help="Weight decay coefficient for L2 regularization.")
    return parser

def add_train_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="HunyuanVideo train args")


    return parser

def add_network_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Network")
    # group.add_argument("--model", type=str, default="HYVideo-T/2",
    #                    help="Model architecture to use. It it also used to determine the experiment directory.")
    group.add_argument(
        "--model",
        type=str,
        choices=list(HUNYUAN_VIDEO_CONFIG.keys()),
        default="HYVideo-T/2",
        help="Model architecture to use. It it also used to determine the experiment directory."
    )
    group.add_argument("--latent-channels", type=str, default=None,
                       help="Number of latent channels of DiT. If None, it will be determined by `vae`. If provided, "
                            "it still needs to match the latent channels of the VAE model.")
    group.add_argument("--video-condition", action="store_true", default=False, help="Load Video Editing Model.") # 250723, copy from HYCustom-V2V
    group.add_argument("--audio-condition", action="store_true", default=False, help="Load Audio Editing Model.") # 250723, copy from HYCustom-V2V
    group.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")

    # group.add_argument(
    #     "--precision",
    #     type=str,
    #     default="bf16",
    #     choices=PRECISIONS,
    #     help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.",
    # )
    group.add_argument("--gradient-checkpoint", action="store_true",
                       help="Enable gradient checkpointing to reduce memory usage.")

    group.add_argument("--gradient-checkpoint-layers", type=int, default=-1,
                       help="Number of layers to checkpoint. -1 for all layers. `n` for the first n layers.")
    return parser

def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Extra Models (VAE, Text Encoder, Tokenizer)")

    # VAE
    group.add_argument("--vae", type=str, default="884-16c-hy0801",  help="Name of the VAE model.")
    group.add_argument("--vae-precision", type=str, default="fp16", 
                       help="Precision mode for the VAE model.")
    group.add_argument("--vae-tiling", action="store_true", default=True, help="Enable tiling for the VAE model.")
    group.add_argument("--text-encoder", type=str, default="llava-llama-3-8b", choices=list(TEXT_ENCODER_PATH),
                       help="Name of the text encoder model.")
    group.add_argument("--text-encoder-precision", type=str, default="fp16", choices=PRECISIONS,
                       help="Precision mode for the text encoder model.")
    group.add_argument("--text-states-dim", type=int, default=4096, help="Dimension of the text encoder hidden states.")
    group.add_argument("--text-len", type=int, default=256, help="Maximum length of the text input.")
    group.add_argument("--tokenizer", type=str, default="llava-llama-3-8b", choices=list(TOKENIZER_PATH),
                       help="Name of the tokenizer model.")
    group.add_argument("--text-encoder-infer-mode", type=str, default="encoder", choices=["encoder", "decoder"],
                       help="Inference mode for the text encoder model. It should match the text encoder type. T5 and "
                            "CLIP can only work in 'encoder' mode, while Llava/GLM can work in both modes.")
    
    group.add_argument(
        "--prompt-template",
        type=str,
        default="li-dit-encode-video",
        choices=PROMPT_TEMPLATE,
        help="Image prompt template for the decoder-only text encoder model.",
    )

    group.add_argument("--prompt-template-video", type=str, default='li-dit-encode-video', choices=PROMPT_TEMPLATE,
                       help="Video prompt template for the decoder-only text encoder model.")
    group.add_argument("--hidden-state-skip-layer", type=int, default=2,
                       help="Skip layer for hidden states.")
    group.add_argument("--apply-final-norm", action="store_true",
                       help="Apply final normalization to the used text encoder hidden states.")

    # - CLIP
    group.add_argument("--text-encoder-2", type=str, default='clipL', choices=list(TEXT_ENCODER_PATH),
                       help="Name of the second text encoder model.")
    group.add_argument("--text-encoder-precision-2", type=str, default="fp16", choices=PRECISIONS,
                       help="Precision mode for the second text encoder model.")
    group.add_argument("--text-states-dim-2", type=int, default=768,
                       help="Dimension of the second text encoder hidden states.")
    group.add_argument("--tokenizer-2", type=str, default='clipL', choices=list(TOKENIZER_PATH),
                       help="Name of the second tokenizer model.")
    group.add_argument("--text-len-2", type=int, default=77, help="Maximum length of the second text input.")
    group.set_defaults(use_attention_mask=True)
    group.add_argument("--text-projection", type=str, default="single_refiner", choices=TEXT_PROJECTION,
                       help="A projection layer for bridging the text encoder hidden states and the diffusion model "
                            "conditions.")
    # teacher model when using distillation
    group.add_argument("--teacher-precision", type=str, default="fp16", 
                       help="Precision mode for the teacher model.")
    return parser


def add_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule")

    group.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )

    group.add_argument("--flow-shift-eval-video", type=float, default=None, help="Shift factor for flow matching schedulers when using video data.")
    group.add_argument("--flow-reverse", action="store_true", default=True, help="If reverse, learning/sampling from t=1 -> t=0.")
    group.add_argument("--flow-solver", type=str, default="euler", help="Solver for flow matching.")
    group.add_argument("--use-linear-quadratic-schedule", action="store_true", help="Use linear quadratic schedule for flow matching."
                                                    "Follow MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)")
    group.add_argument("--linear-schedule-end", type=int, default=25, help="End step for linear quadratic schedule for flow matching.")

    group.add_argument("--num-clean-latent-frames", type=int, default=0, help="add noise to only a part of video latents")
    group.add_argument(
        "--use-noise-augmentation", action="store_true", help="Whether to add small noise to clean latent frames."
    )
    group.add_argument("--clean-latent-frames-drop-p", type=float, default=0, help="Whether to drop conditional clean latent frames during training.")
    group.add_argument("--scene-video-drop-p", type=float, default=0, help="Whether to drop input scene video during training.")
    group.add_argument(
        "--use-diffusion-forcing", action="store_true", help="Whether to use diffusion forcing training."
    )
    group.add_argument(
        "--use-scene-condition", action="store_true", help="Whether to use background scene video latents."
    )
    group.add_argument(
        "--use-scene-first-frame-mask", action="store_true", help="Whether to use the first frame mask of scene video latents."
    )
    group.add_argument(
        "--use-scene-all-frame-mask", action="store_true", help="Whether to use the all frame mask of background scene video latents."
    )
    group.add_argument("--update-generator-every", type=int, default=1, help="DMD STUDENT model update frequency")
    group.add_argument("--update-fake-every", type=int, default=1, help="DMD FAKE SCORE model update frequency")
    group.add_argument("--num-few-steps", type=int, default=4, help="number of distilled few-step model inference")
    group.add_argument(
        "--use-regression-loss", action="store_true", help="Whether to use regression loss when distillation."
    )
    return parser

def add_lora_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="lora args")

    group.add_argument(
        "--use-lora", action="store_true", help="Whether to open lora mode."
    )

    group.add_argument(
        "--lora-path", type=str, default="", help="Weight path for lora model."
    )

    group.add_argument(
        "--lora-scale", type=float, default=1.0, help="Fusion scale for lora model."
    )

    group.add_argument(
        "--lora-rank", type=int, default=64, help="Rank for lora model."
    )

    return parser

def add_evaluation_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Validation Loss Evaluation")
    parser.add_argument("--precision", type=str, default="bf16", choices=PRECISIONS,
                    help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.")
    parser.add_argument("--reproduce", action="store_true",
                       help="Enable reproducibility by setting random seeds and deterministic algorithms.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint to evaluate.")

    parser.add_argument("--model-base", type=str, default="models", help="Path to the checkpoint to evaluate.")
    parser.add_argument(
        "--dit-weight",
        type=str,
        default="models/hunyuancustom_720P/mp_rank_00_model_states.pt",
        help="Path to the HunyuanVideo model. If None, search the model in the args.model_root."
        "1. If it is a file, load the model directly."
        "2. If it is a directory, search the model in the directory. Support two types of models: "
        "1) named `pytorch_model_*.pt`"
        "2) named `*_model_states.pt`, where * can be `mp_rank_00`.",
    )

    parser.add_argument("--load-key", type=str, default="module", choices=["module", "ema"],
                       help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.")
    parser.add_argument("--cpu-offload", action="store_true", help="Use CPU offload for the model load.")
    group.add_argument( "--use-fp8", action="store_true", help="Enable use fp8 for inference acceleration.")
    group.add_argument("--video-size", type=int, nargs='+', default=512,
                        help="Video size for training. If a single value is provided, it will be used for both width "
                            "and height. If two values are provided, they will be used for width and height "
                            "respectively.")
    group.add_argument("--sample-n-frames", type=int, default=1,
                       help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1")
    group.add_argument("--infer-steps", type=int, default=100, help="Number of denoising steps for inference.")
    group.add_argument("--val-disable-autocast", action="store_true",
                       help="Disable autocast for denoising loop and vae decoding in pipeline sampling.")
    group.add_argument("--num-images", type=int, default=1, help="Number of images to generate for each prompt.")
    group.add_argument("--seed", type=int, default=1024, help="Seed for evaluation.")
    group.add_argument("--save-path-suffix", type=str, default="", help="Suffix for the directory of saved samples.")
    group.add_argument("--pos-prompt", type=str, default='', help="Prompt for sampling during evaluation.")
    group.add_argument("--neg-prompt", type=str, default='', help="Negative prompt for sampling during evaluation.")
    group.add_argument("--add-pos-prompt", type=str, default='', help="Addition prompt for sampling during evaluation.")
    group.add_argument("--add-neg-prompt", type=str, default='', help="Addition negative prompt for sampling during evaluation.")
    group.add_argument("--pad-face-size", type=float, default=0.7, help="Pad bbox for face align.")
    group.add_argument("--image-path", type=str, default="",  help="")
    group.add_argument("--save-path", type=str, default=None, help="Path to save the generated samples.")
    group.add_argument("--input", type=str, default=None, help="test data.")
    group.add_argument("--input-video", type=str, default=None, help="conditional input video for autoregressive video generation.")
    group.add_argument("--input-scene-video", type=str, default=None, help="scene video for background scene condition.")
    group.add_argument("--input-scene-mask-video", type=str, default=None, help="scene video for background scene condition.")
    group.add_argument("--output-video", type=str, default=None, help="output video path list.")
    group.add_argument("--local-rank-split", type=int, default=0, help="for batch inference.")
    group.add_argument("--scene-mask-type", type=str, default=None, help="MASKED or INPAINTED")
    group.add_argument("--bucket-sample-size", type=int, nargs='+', default=None,
                        help="used by self.generate_crop_size_list in hymm_sp/data_kits/video_dataset_lora.py VideoDataset.")
    group.add_argument(
        "--only-text-character", action="store_true", help="Whether to do not use multi-view reference images."
    )
    group.add_argument(
        "--only-text-scene", action="store_true", help="Whether to do not use scene videos."
    )
    group.add_argument(
        "--only-front-back-view", action="store_true", help="Whether to only use front and back view reference images."
    )
    group.add_argument("--pose-enhance", action="store_true", help="Utilize DWpose to improve pose control during video editing.")
    group.add_argument("--expand-scale", type=int, default=0, help="Expand mask")
    group.add_argument("--audio-strength", type=float, default=1.0, help="Control the influence of audio.")
    group.add_argument("--item-name", type=str, default=None, help="")
    group.add_argument("--cfg-scale", type=float, default=7.5, help="Classifier free guidance scale.")
    group.add_argument("--ip-cfg-scale", type=float, default=0, help="Classifier free guidance scale.")
    group.add_argument("--use-deepcache", type=int, default=1)

    group.add_argument("--multi-view-ids", type=int, nargs='+', default=0,
                       help="multi-view image IDs for reference character/object")
    return parser

def sanity_check_args(args):
    # VAE channels
    vae_pattern = r"\d{2,3}-\d{1,2}c-\w+"
    if not re.match(vae_pattern, args.vae):
        raise ValueError(
            f"Invalid VAE model: {args.vae}. Must be in the format of '{vae_pattern}'."
        )
    vae_channels = int(args.vae.split("-")[1][:-1])
    if args.latent_channels is None:
        args.latent_channels = vae_channels
    if vae_channels != args.latent_channels:
        raise ValueError(
            f"Latent channels ({args.latent_channels}) must match the VAE channels ({vae_channels})."
        )
    return args
