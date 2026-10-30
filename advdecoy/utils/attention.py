import torch

from advdecoy.defenses.armro import armro_token_scores


@torch.no_grad()
def received_attention_map(attn_maps, layers=None):
    """
    Attention received by each image token, as seen by ARMRO, averaged over
    layers. Returns a [grid_size, grid_size] numpy array for the first image.
    """
    if layers is None:
        layers = range(len(attn_maps))

    scores = torch.stack([armro_token_scores(attn_maps[l])[0] for l in layers]).mean(dim=0)
    grid_size = int(scores.numel() ** 0.5)

    return scores.reshape(grid_size, grid_size).cpu().numpy()


@torch.no_grad()
def cls_attention_map(attn_maps, layers=None):
    """
    CLS-to-patch attention (mean over heads and layers) for the first image,
    as a [grid_size, grid_size] numpy array.
    """
    if layers is None:
        layers = range(len(attn_maps))

    scores = torch.stack([attn_maps[l][0, :, 0, 1:].mean(dim=0) for l in layers]).mean(dim=0)
    grid_size = int(scores.numel() ** 0.5)

    return scores.reshape(grid_size, grid_size).cpu().numpy()
