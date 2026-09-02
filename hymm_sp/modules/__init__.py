from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
from .models_ar import HYVideoARDiffusionTransformer
from .models_ar_v2v import HYVideoARV2VDiffusionTransformer

def load_model(args, in_channels, out_channels, factor_kwargs):
    model = HYVideoDiffusionTransformer(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        **HUNYUAN_VIDEO_CONFIG[args.model],
        **factor_kwargs,
    )
    return model

def load_model_ar(args, in_channels, out_channels, factor_kwargs):
    model = HYVideoARDiffusionTransformer(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        **HUNYUAN_VIDEO_CONFIG[args.model],
        **factor_kwargs,
    )
    return model

def load_model_ar_v2v(args, in_channels, out_channels, video_condition, audio_condition, factor_kwargs):
    model = HYVideoARV2VDiffusionTransformer(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        video_condition=video_condition,
        audio_condition=audio_condition,
        **HUNYUAN_VIDEO_CONFIG[args.model],
        **factor_kwargs,
    )
    return model