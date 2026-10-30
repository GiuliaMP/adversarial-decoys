import torch


# Token convention used throughout the package:
#     image-grid token t = row * grid_size + col, t in [0, grid_size^2 - 1].
# In attention maps the same token sits at index t + 1 (index 0 is CLS).


def tokens_to_mask(token_indices, image_size=224, patch_size=16, device="cpu"):
    """
    Build an image-space binary mask [1, 1, H, W] covering the given
    image-grid tokens.
    """
    grid_size = image_size // patch_size

    mask = torch.zeros(1, 1, image_size, image_size, device=device)

    for token in token_indices:
        r = int(token) // grid_size
        c = int(token) % grid_size

        mask[:, :, r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size] = 1.0

    return mask


def sample_random_tokens(grid_size, num_tokens, forbidden_tokens=None, generator=None):
    """Sample num_tokens distinct image-grid tokens, excluding forbidden_tokens."""
    if forbidden_tokens is None:
        forbidden_tokens = set()
    else:
        forbidden_tokens = set(int(t) for t in forbidden_tokens)

    valid_tokens = [t for t in range(grid_size * grid_size) if t not in forbidden_tokens]

    if num_tokens > len(valid_tokens):
        raise ValueError(
            f"Cannot sample {num_tokens} tokens. "
            f"Only {len(valid_tokens)} valid non-forbidden tokens are available."
        )

    perm = torch.randperm(len(valid_tokens), generator=generator).tolist()
    return [valid_tokens[i] for i in perm[:num_tokens]]
