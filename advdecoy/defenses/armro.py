import torch
import torch.nn.functional as F

from advdecoy.utils.patches import tokens_to_mask


def armro_token_scores(attn):
    """
    ARMRO score of every image token at one layer: attention received,
    averaged over heads and all queries. attn: [B, H, N, N] -> [B, N - 1].
    """
    return attn.mean(dim=1).mean(dim=1)[:, 1:]


def extract_suspicious_tokens_from_attention(attn, n_d=5, tau=1.2, gap=False):
    """
    Select suspicious tokens at one layer.

    Among the n_d highest-scoring image tokens, token i is suspicious if its
    score exceeds tau times the next one. With gap=True, once such a jump is
    found all stronger tokens are suspicious too.

    Returns one set of image-grid tokens per batch element.
    """
    token_scores = armro_token_scores(attn)

    selected_per_sample = []

    for b in range(token_scores.shape[0]):
        k_eff = min(n_d, token_scores.shape[1])
        top_scores, top_indices = torch.topk(token_scores[b], k=k_eff, largest=True)

        suspicious = []

        if gap:
            flag = False
            for i in range(k_eff - 2, -1, -1):
                if top_scores[i] > tau * top_scores[i + 1] or flag:
                    suspicious.append(int(top_indices[i].item()))
                    flag = True
        else:
            for i in range(k_eff - 1):
                if top_scores[i] > tau * top_scores[i + 1]:
                    suspicious.append(int(top_indices[i].item()))

        selected_per_sample.append(set(suspicious))

    return selected_per_sample


def extract_armro_tokens(attn_maps, layers_to_use, n_d=5, tau=1.2, gap=False):
    """Union over layers_to_use of the suspicious tokens, one sorted list per image."""
    B = attn_maps[0].shape[0]
    selected_tokens = [set() for _ in range(B)]

    for layer_idx in layers_to_use:
        layer_selected = extract_suspicious_tokens_from_attention(attn_maps[layer_idx], n_d=n_d, tau=tau, gap=gap)

        for b in range(B):
            selected_tokens[b].update(layer_selected[b])

    return [sorted(tokens) for tokens in selected_tokens]


def apply_armro_mask(images, selected_tokens, patch_size=16, fill_mode="mean"):
    """
    Replace the selected tokens of each image.

    fill_mode: "mean" (image mean color), "gray" (0.5), "zero", or "blur"
    (local average pooling).
    """
    image_size = images.shape[-1]

    mask = torch.cat(
        [tokens_to_mask(tokens, image_size=image_size, patch_size=patch_size, device=images.device) for tokens in selected_tokens],
        dim=0,
    )

    if fill_mode == "mean":
        replacement = images.mean(dim=(2, 3), keepdim=True).expand_as(images)
    elif fill_mode == "gray":
        replacement = torch.ones_like(images) * 0.5
    elif fill_mode == "zero":
        replacement = torch.zeros_like(images)
    elif fill_mode == "blur":
        replacement = F.avg_pool2d(images, kernel_size=patch_size, stride=1, padding=patch_size // 2)
        replacement = replacement[:, :, :image_size, :image_size]
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    defended = images * (1.0 - mask) + replacement * mask

    return defended.clamp(0, 1), mask


@torch.no_grad()
def evaluate_armro_defense(
    model,
    x_input_01,
    attn_maps,
    predict_fn,
    layers_to_use,
    n_d=5,
    tau=1.2,
    gap=False,
    patch_size=16,
    fill_mode="mean",
):
    """
    Run ARMRO on x_input_01, given its attention maps.

    predict_fn(model, x_01) must return a dict with at least "pred" and "conf".
    Selected tokens are image-grid indices.
    """
    selected_tokens = extract_armro_tokens(attn_maps, layers_to_use, n_d=n_d, tau=tau, gap=gap)

    x_defended_01, defense_mask = apply_armro_mask(x_input_01, selected_tokens, patch_size=patch_size, fill_mode=fill_mode)

    return {
        "selected_tokens": selected_tokens,
        "x_defended_01": x_defended_01,
        "defense_mask": defense_mask,
        "before": predict_fn(model, x_input_01),
        "after": predict_fn(model, x_defended_01),
    }
