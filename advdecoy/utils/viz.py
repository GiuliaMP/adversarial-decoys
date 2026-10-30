import numpy as np
import matplotlib.patches as mpatches


def to_numpy_image(x_01):
    """[1, 3, H, W] tensor in [0, 1] -> [H, W, 3] numpy array."""
    return x_01[0].detach().permute(1, 2, 0).cpu().numpy()


def show_image(ax, x_01, title=None):
    ax.imshow(to_numpy_image(x_01))
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=10)


def show_attention(ax, x_01, attn_map, title=None, alpha=0.6, cmap="viridis"):
    """Overlay a [grid, grid] attention map on the image (one cell per token)."""
    image = to_numpy_image(x_01)
    size = image.shape[0]

    ax.imshow(image)
    ax.imshow(attn_map, cmap=cmap, alpha=alpha, extent=(0, size, size, 0), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=10)


def draw_tokens(ax, token_indices, grid_size=14, patch_size=16, color="lime", linewidth=2, linestyle="-"):
    """Draw a box around each image-grid token."""
    for token in token_indices:
        r, c = int(token) // grid_size, int(token) % grid_size
        ax.add_patch(
            mpatches.Rectangle(
                (c * patch_size - 0.5, r * patch_size - 0.5),
                patch_size,
                patch_size,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )
        )


def topk(probs, k=5):
    """Top-k (index, probability) pairs for the first image."""
    values, indices = probs[0].topk(k)
    return [(int(i), float(v)) for i, v in zip(indices, values)]


def show_topk(ax, probs, class_names, k=5, highlight=None, title=None):
    """Horizontal bar chart of the top-k classes; highlight is a class index."""
    pairs = topk(probs, k)[::-1]
    labels = [class_names[i] for i, _ in pairs]
    values = [v for _, v in pairs]
    colors = ["tab:green" if i == highlight else "tab:gray" for i, _ in pairs]

    ax.barh(np.arange(len(pairs)), values, color=colors)
    ax.set_yticks(np.arange(len(pairs)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.tick_params(axis="x", labelsize=8)
    if title is not None:
        ax.set_title(title, fontsize=10)
