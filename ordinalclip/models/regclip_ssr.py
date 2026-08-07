import os.path as osp

import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from modelscope import AutoModel
from transformers import CLIPVisionModel

from clip import clip

from ordinalclip.utils import get_logger

from .builder import MODELS
from .prompt_leaners import PROMPT_LEARNERS
from .prompt_leaners.plain_prompt_learner import PlainPromptLearner

logger = get_logger(__name__)


# for age estimation
bin_list_a = [0, 13, 19, 35, 65]
bin_list_b = [0, 13, 19, 35, 65]

bin_width_a = [13,6,16,30,36]
bin_width_b = [13,6,16,30,36]


@MODELS.register_module()
class RegCLIPSSR(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
        d=768,
        dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        clip_vision_model_name="openai/clip-vit-base-patch16",
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 model, call `clip.model.convert_weights`")

        pretrained_model_name = dino_model_name

        # Kept for compatibility with the original experiment checkpoints.
        # The released forward pass, like the experiment code, uses CLIP features.
        self.dino_encoder  = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
        self.image_encoder = CLIPVisionModel.from_pretrained(clip_vision_model_name)

        self.text_encoder = TextEncoder(clip_model)
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learner: PlainPromptLearner = PROMPT_LEARNERS.build(prompt_learner_cfg)
        self.psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens
        self.logit_scale = clip_model.logit_scale

        self.embed_dims = clip_model.text_projection.shape[1]
        self.num_ranks = self.prompt_learner.num_ranks

        match pretrained_model_name:
            case "facebook/dinov3-vitb16-pretrain-lvd1689m":
                d = 768
            case "facebook/dinov3-vitl16-pretrain-lvd1689m":
                d = 1024
            case "facebook/dinov3-vit7b16-pretrain-lvd1689m":
                d = 4096
            case _:
                d = 512

        self.d = d

        # we first adopt CLIP-adapter based adaptation method. After experiment, we found fully finetune the image encoder could get the better performance.
        self.image_adapter = Adapter(self.d, 4)
        self.down_adapter = nn.Sequential(
            nn.Linear(2*self.d, self.d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.d, self.d, bias=False),
            nn.ReLU(inplace=True)
        )
        self.align_adapter = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.d, bias=False),
            nn.ReLU(inplace=True)
        )
        self.recover = nn.Sequential(
            nn.Linear(2 * self.d, self.d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.d, self.num_ranks, bias=False)
        )
        self.drop = nn.Dropout()

        self.cls_layer = nn.Linear(self.d, self.num_ranks)
        init.kaiming_uniform_(self.cls_layer.weight, nonlinearity='relu')
        init.zeros_(self.cls_layer.bias)


        # Shallow classifiers attached to intermediate layers.
        # Produce per-layer shallow logits for deep->shallow self-distillation.
        self.layers = list(range(12))
        self.shallow_logit_layers = self.layers[:-1] if len(self.layers) > 1 else list(self.layers)
        self.shallow_classifiers = nn.ModuleList([nn.Linear(self.d, self.num_ranks) for _ in self.shallow_logit_layers])
        for m in self.shallow_classifiers:
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            init.zeros_(m.bias)
        self.regressor = SSRModule()

        self.fuse = LLNLayerScale(num_layers=len(self.layers), dim=self.d)

        # Shallow-to-deep feature correction (for deep-to-shallow distillation pipeline).
        # Use early-layer CLS features to generate a residual that corrects deep features,
        # thus modifying the final classification logits.
        self.shallow_to_deep = nn.Linear(self.d, self.d, bias=False)
        nn.init.zeros_(self.shallow_to_deep.weight)
        # Scalar gate in (0, 1) via sigmoid; starts at ~0.5 but proj is zero-initialized.
        self.shallow_gate = nn.Parameter(torch.tensor(0.0))

        if pretrained_model_name == "facebook/dinov3-vit7b16-pretrain-lvd1689m":
            self.apply_frozen_image_encoder()

        # EMA衰减率
        self.ema_decay = 0.995

        self.project = Linear_Projection(self.d, self.num_ranks)

    def apply_frozen_image_encoder(self):
        """冻结图像编码器的参数"""
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.image_encoder.eval()

        trainable_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.image_encoder.parameters())
        print(f"图像编码器可训练参数: {trainable_params}/{total_params} ({100*trainable_params/total_params:.2f}%)")

    def topk_max_pooling(self, similarity_matrix, K):
        # Get top-K similarities for each class (shape: [N, K, C])
        topk_values, _ = torch.topk(similarity_matrix, K, dim=1)

        # Average the top-K similarities per class (shape: [N, C])
        pooled_output = topk_values.mean(dim=1)

        return pooled_output

    def forward(self, images, mode='student'):
        # NOTE: `mode` is kept for backward compatibility; EMA teacher is managed by Runner.

        output = self.image_encoder(pixel_values=images, return_dict=True, output_hidden_states=True)

        hs = output.hidden_states # (b, 13, 197, d)  前12层是transformer的输出，最后一层是post_layernorm的输出(pooler_output是对[CLS] token做了线性变换和tanh激活)

        # Multi-level CLS features (B, L, D) used by runner-side distillation.
        layer_cls = torch.stack([hs[i][:, 0, :] for i in self.layers], dim=1)

        sentence_embeds = self.prompt_learner() # [num_class, 77, 512]
        psudo_sentence_tokens = self.psudo_sentence_tokens # [num_class, 77]
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)
        text_features = self.align_adapter(text_features)
        patch_features = hs[-1][:,1:,:] # (B, N, D)

        # Deep feature (default last layer CLS).
        deep_feat = hs[-1][:, 0, :]

        # Shallow feature: mean CLS over intermediate layers.
        shallow_feat = torch.stack([hs[i][:, 0, :] for i in self.shallow_logit_layers], dim=1).mean(dim=1)
        # Multi-depth shallow logits from intermediate layers (trainable shallow classifiers).
        shallow_logits_multi = None
        try:
            shallow_feats_multi = [hs[i][:, 0, :] for i in self.shallow_logit_layers]
            shallow_logits_multi = torch.stack(
                [clf(feat) for clf, feat in zip(self.shallow_classifiers, shallow_feats_multi)], dim=1
            )
        except Exception:
            shallow_logits_multi = None

        # Use shallow feature to correct deep feature (residual).
        gate = torch.sigmoid(self.shallow_gate)
        deep_feat_corr = deep_feat + gate * self.shallow_to_deep(shallow_feat)

        image_features = deep_feat_corr
        patch_features = patch_features.reshape(-1, self.d) # (B, P, d) -> (B*P, d)

        logit_scale = self.logit_scale.exp()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_cos = logit_scale * image_features @ text_features.t()

        image_features, text_features = self.project(image_features, text_features)
        # Also expose shallow/deep logits for optional distillation losses.、
        logits_base = logits_cos
        deep_logits = logits_cos

        attn_scores = (image_features @ text_features.t()) / math.sqrt(self.d)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_text = attn_weights @ text_features
        attn_fused = torch.cat([image_features, attn_text], dim=-1)
        enhance_logits = self.recover(attn_fused)

        shallow_logits = shallow_logits_multi.mean(dim=1)
        regress_age = 0

        aux = {
            "layer_cls": layer_cls,
            "shallow_feat": shallow_feat,
            "deep_feat": deep_feat,
            "deep_feat_corr": deep_feat_corr,
            "shallow_gate": gate.detach(),
            'enhance_logits': enhance_logits,
            "deep_logits": deep_logits,
            "shallow_logits": shallow_logits,
            "shallow_logits_multi": shallow_logits_multi,
            "shallow_logit_layers": list(self.shallow_logit_layers),
        }
        return logits_base, regress_age, image_features, text_features, patch_features, aux

    def forward_text_only(self):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)

        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)

class Linear_Projection(nn.Module):
    """Minimal projection head used by RegCLIPSSR.

    Contract (as used in RegCLIPSSR.forward):
    - forward(img_features, txt_features) -> (img_features, txt_features)
    - get_logits(img_features, txt_features) -> (B, num_ranks)
    """

    def __init__(
        self,
        d: int,
        num_ranks: int,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.num_ranks = int(num_ranks)
        self.normalize = bool(normalize)

        self.img_proj = Adapter(self.d)
        self.txt_proj = Adapter(self.d)

    def forward(self, img_features: torch.Tensor, txt_features: torch.Tensor):
        img_features = self.img_proj(img_features)
        txt_features = self.txt_proj(txt_features)
        if self.normalize:
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        return img_features, txt_features

    def get_logits(self, img_features: torch.Tensor, txt_features: torch.Tensor) -> torch.Tensor:
        """Return stabilized dot-product logits (B, num_ranks)."""
        logits = img_features @ txt_features.t()

        return logits

class LLNLayerScale(nn.Module):
    def __init__(self, num_layers, dim):
        super().__init__()
        self.num_layers = num_layers
        # 为每层特征定义 LLN 模块
        self.llns = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        ) for _ in range(num_layers)])

        # Layerscale 可学习权重
        self.w = nn.Parameter(torch.ones(num_layers))

    def forward(self, features):
        # features: list of tensors [B, D], length = num_layers
        fused = 0
        for i, f in enumerate(features):
            fused += self.w[i] * self.llns[i](f)
        return fused

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) # [5, 77, 512]
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x



class SSRModule(nn.Module):
    def __init__(self, stage_num=[5, 3], d=512,
                 class_range=101, lambda_index=1., lambda_delta=1.):
        super(SSRModule, self).__init__()

        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range
        self.d = d

        self.stream1_stage2 = Adapter(self.d, 4)
        self.funsion_block_stream1_stage_2_prediction_block = nn.Linear(d, self.stage_num[1])
        self.funsion_block_stream1_stage_1_prediction_block = nn.Linear(d, self.stage_num[0])

        self.stream2_stage2 = Adapter(self.d, 4)
        self.funsion_block_stream2_stage_2_prediction_block = nn.Linear(d, self.stage_num[1])
        self.funsion_block_stream2_stage_1_prediction_block = nn.Linear(d, self.stage_num[0])

        self.stage2_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[1], 2 * self.stage_num[1]),
            nn.ReLU()
        )
        self.stage2_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[1], self.stage_num[1]),
            nn.ReLU()
        )
        self.stage2_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[1], self.stage_num[1]),
            nn.Tanh()
        )
        self.stage2_delta_k = nn.Sequential(
            nn.Linear(2 * self.stage_num[1], 1),
            nn.Tanh()
        )
        self.stage1_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, logits):

        prob_stage_1 = F.softmax(logits, dim=1)
        embedding_stage1_after_PB = self.stage1_FC_after_PB(logits)
        stage1_delta_k = self.stage1_delta_k(embedding_stage1_after_PB)

        stage1_regress_a = prob_stage_1[:, 0] * 0

        for index in range(self.stage_num[0]):
            width = (bin_list_a[index] / (1 + self.lambda_delta * stage1_delta_k[:, index]))
            stage1_regress_a = stage1_regress_a + prob_stage_1[:, index] * width
        stage1_regress_a = torch.unsqueeze(stage1_regress_a, 1)


        regress_age_a = stage1_regress_a
        regress_age_a = regress_age_a.squeeze(1)

        regress_age = regress_age_a

        return regress_age


def load_clip_to_cpu(
    text_encoder_name,
    image_encoder_name,
    root=osp.join(osp.expanduser("~/.cache/clip")),
    ):
    # text backbone
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print

    print_func("Building CLIP model...")
    text_backbone_name = text_encoder_name
    print_func(f"Text backbone : {text_backbone_name}'s counterpart.")
    url = clip._MODELS[text_backbone_name]
    model_path = clip._download(url, root=root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    # image backbone
    embed_dim = model.text_projection.shape[1]
    input_resolution = model.visual.input_resolution
    image_backbone_name = image_encoder_name
    print_func(f"Image backbone: {image_backbone_name}")

    if image_backbone_name != text_backbone_name:
        raise ValueError("The Adience release uses the same CLIP text and image backbone.")
    print_func(f"CLIP Image encoder: {image_backbone_name}!")

    return model
