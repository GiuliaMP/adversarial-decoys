import torch
import torch.nn.functional as F

from advdecoy.utils.patches import sample_random_tokens


def sample_decoy_tokens(grid_size, num_tokens, attack_token_indices, generator=None):
    """Random decoy placement: num_tokens image-grid tokens disjoint from the attack."""
    return sample_random_tokens(
        grid_size=grid_size,
        num_tokens=num_tokens,
        forbidden_tokens=attack_token_indices,
        generator=generator,
    )


def apply_decoy_patch(x_base_01, patch_raw, patch_mask):
    """
    Replace the masked region of x_base_01 with sigmoid(patch_raw).

    The sigmoid keeps the decoy a valid image in [0, 1] without projection.
    """
    patch = torch.sigmoid(patch_raw)
    x_final_01 = x_base_01 * (1 - patch_mask) + patch * patch_mask

    return x_final_01, patch


def defense_aligned_attention_loss(
    attn_maps,
    target_layers,
    decoy_token_indices,
    grid_size,
    beta_ratio=2.0,
    beta_layer_balance=0.1,
    temperature=0.10,
    topk=3,
    ratio_threshold=2,
    alpha_satisfied=0.05,
    alpha_unsatisfied=1.0,
):
    """
    Attention loss that makes the decoy tokens the most attended tokens, as
    measured by attention-based defenses.

    For every target layer, token scores are the attention received from the
    image-token queries (mean over heads and queries). The loss combines
      - focus:  -log(mean decoy score)
      - ratio:  -log(min of top-k decoy scores / max non-decoy score)
    weighted by alpha (small once the ratio exceeds ratio_threshold), plus a
    KL term that spreads the decoy tokens' soft layer assignment uniformly.

    The non-decoy maximum covers every other token, so the attacked tokens
    do not need to be known.
    """
    device = attn_maps[0].device
    num_img_tokens = grid_size * grid_size

    # Attention-map indices: 0 is CLS, image tokens are 1 ... num_img_tokens.
    patch_query_indices = torch.arange(1, 1 + num_img_tokens, device=device, dtype=torch.long)
    decoy_tokens = torch.tensor(decoy_token_indices, device=device, dtype=torch.long) + 1
    non_decoy_tokens = patch_query_indices[~torch.isin(patch_query_indices, decoy_tokens)]

    # Decoy score matrix S: [L, D].
    S_rows = []
    for layer_idx in target_layers:
        attn_img_to_all = attn_maps[layer_idx][:, :, patch_query_indices, :]
        S_rows.append(attn_img_to_all[:, :, :, decoy_tokens].mean(dim=(0, 1, 2)))

    S = torch.stack(S_rows, dim=0)
    num_layers, num_decoys = S.shape

    base_loss = torch.tensor(0.0, device=device, dtype=S.dtype)
    ratios = []

    for li, layer_idx in enumerate(target_layers):
        attn_img_to_all = attn_maps[layer_idx][:, :, patch_query_indices, :]

        s_decoy_mean = S[li].mean()
        s_decoy_topk_min = torch.topk(S[li], k=min(topk, max(1, num_decoys)), largest=True).values.min()

        non_decoy_scores = attn_img_to_all[:, :, :, non_decoy_tokens].mean(dim=(0, 1, 2))
        s_other_max = non_decoy_scores.max()

        loss_focus = -torch.log(s_decoy_mean + 1e-8)
        loss_ratio = -torch.log(s_decoy_topk_min / (s_other_max + 1e-8) + 1e-8)

        ratio_value = (s_decoy_topk_min / (s_other_max + 1e-8)).detach().item()
        alpha = alpha_satisfied if ratio_value > ratio_threshold else alpha_unsatisfied

        base_loss = base_loss + alpha * (loss_focus + beta_ratio * loss_ratio)
        ratios.append(ratio_value)

    if num_layers > 1 and beta_layer_balance != 0:
        P_layer_given_token = torch.softmax(S / temperature, dim=0)
        layer_mass = P_layer_given_token.mean(dim=1)
        uniform = torch.ones_like(layer_mass) / float(num_layers)

        loss_layer_balance = F.kl_div(torch.log(layer_mass + 1e-8), uniform, reduction="batchmean")
    else:
        loss_layer_balance = torch.tensor(0.0, device=device, dtype=S.dtype)

    loss = base_loss + beta_layer_balance * loss_layer_balance

    return loss, {"ratio_per_layer": ratios}


def optimize_attention_decoy(
    model,
    x_adv_01,
    normalize_fn,
    forward_fn,
    patch_mask,
    target_layers,
    decoy_token_indices,
    grid_size,
    steps=2500,
    lr=5e-2,
    beta_ratio=2.0,
    verbose=True,
):
    """
    Optimize a decoy patch on top of an already attacked image.

    The decoy is parameterized as sigmoid(patch_raw), initialized to gray
    (patch_raw = 0), and optimized with Adam on defense_aligned_attention_loss.
    The classifier output is not part of the objective.
    """
    patch_raw = torch.nn.Parameter(torch.zeros_like(x_adv_01))
    optimizer = torch.optim.Adam([patch_raw], lr=lr)

    history = {"loss": [], "pred": []}

    for step in range(steps):
        optimizer.zero_grad()

        x_final_01, _ = apply_decoy_patch(x_adv_01, patch_raw, patch_mask)
        logits, attn_maps = forward_fn(model, normalize_fn(x_final_01))

        loss, logs = defense_aligned_attention_loss(
            attn_maps=attn_maps,
            target_layers=target_layers,
            decoy_token_indices=decoy_token_indices,
            grid_size=grid_size,
            beta_ratio=beta_ratio,
        )

        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["pred"].append(int(logits.argmax(dim=1).item()))

        if verbose and (step % 100 == 0 or step == steps - 1):
            print(
                f"[decoy] step {step + 1:4d}/{steps} | loss={history['loss'][-1]:.4f} | "
                f"layers with decoy/other ratio > 1: {sum(r > 1 for r in logs['ratio_per_layer'])}/{len(target_layers)} | "
                f"pred={history['pred'][-1]}"
            )

    with torch.no_grad():
        x_final_01, patch = apply_decoy_patch(x_adv_01, patch_raw, patch_mask)

    return {
        "x_final_01": x_final_01.detach(),
        "patch": patch.detach(),
        "history": history,
    }

