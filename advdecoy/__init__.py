"""Minimal demo of adversarial decoys against attention-based defenses for Vision Transformers."""

from advdecoy.models.vit import (
    load_timm_vit,
    get_timm_data_config,
    normalize_img,
    vit_forward_with_attention,
    predict_with_attention,
)
from advdecoy.attacks.patch_fool import select_tokens_by_attention, optimize_patch_fool
from advdecoy.attacks.decoy import sample_decoy_tokens, optimize_attention_decoy
from advdecoy.defenses.armro import evaluate_armro_defense
from advdecoy.utils.patches import tokens_to_mask
