import torch
import timm


def load_timm_vit(model_name, device):
    """
    Load a pretrained timm ViT in eval mode.

    Fused attention is disabled so that attention maps can be recomputed
    explicitly by vit_forward_with_attention.
    """
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)

    for block in model.blocks:
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False

    return model


def get_timm_data_config(model):
    return timm.data.resolve_model_data_config(model)


def normalize_img(x_01, mean, std):
    """Normalize an image batch in [0, 1] with the model mean/std."""
    mean = torch.tensor(mean, device=x_01.device, dtype=x_01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x_01.device, dtype=x_01.dtype).view(1, 3, 1, 1)

    return (x_01 - mean) / std


def attention_forward(attn_module, x):
    """
    Manual forward for timm ViT attention.

    Returns:
        out: [B, N, C]
        attn: [B, heads, N, N]
    """
    B, N, C = x.shape

    qkv = attn_module.qkv(x)
    qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]

    if hasattr(attn_module, "q_norm"):
        q = attn_module.q_norm(q)

    if hasattr(attn_module, "k_norm"):
        k = attn_module.k_norm(k)

    attn = (q @ k.transpose(-2, -1)) * attn_module.scale
    attn = attn.softmax(dim=-1)

    head_out = attn_module.attn_drop(attn) @ v

    out = head_out.transpose(1, 2).reshape(B, N, C)
    out = attn_module.proj(out)
    out = attn_module.proj_drop(out)

    return out, attn


def block_forward_with_attention(block, x):
    y, attn = attention_forward(block.attn, block.norm1(x))
    x = x + block.drop_path1(block.ls1(y))

    y = block.mlp(block.norm2(x))
    x = x + block.drop_path2(block.ls2(y))

    return x, attn


def vit_forward_with_attention(model, x_norm):
    """
    Forward a timm ViT and return logits and the attention maps of all layers.

    Each attention map has shape [B, heads, N, N], with token 0 = CLS and
    tokens 1..N-1 = image tokens.
    """
    B = x_norm.shape[0]

    patch_tokens = model.patch_embed(x_norm)
    cls_token = model.cls_token.expand(B, -1, -1)

    tokens = torch.cat([cls_token, patch_tokens], dim=1)
    tokens = tokens + model.pos_embed
    tokens = model.pos_drop(tokens)

    attn_maps = []

    for block in model.blocks:
        tokens, attn = block_forward_with_attention(block, tokens)
        attn_maps.append(attn)

    tokens = model.norm(tokens)
    logits = model.head(tokens[:, 0])

    return logits, attn_maps


@torch.no_grad()
def predict_with_attention(model, x_01, mean, std):
    """Predict an image batch in [0, 1] and return probabilities and attention maps."""
    logits, attn_maps = vit_forward_with_attention(model, normalize_img(x_01, mean, std))

    probs = logits.softmax(dim=1)

    return {
        "logits": logits,
        "probs": probs,
        "pred": probs.argmax(dim=1),
        "conf": probs.max(dim=1).values,
        "attn_maps": attn_maps,
    }
