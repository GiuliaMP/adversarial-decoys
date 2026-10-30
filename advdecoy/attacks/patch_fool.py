import torch
import torch.nn.functional as F

from advdecoy.utils.patches import tokens_to_mask


def select_tokens_by_attention(attn_maps, num_tokens, select_layer=4):
    """
    Patch-Fool attention-based patch selection.

    Scores each image token by the attention it receives at select_layer
    (mean over heads and queries) and returns the num_tokens highest-scoring
    image-grid tokens. Expects a single image.
    """
    attn = attn_maps[select_layer]

    if attn.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {attn.shape[0]}.")

    key_importance = attn.mean(dim=1).mean(dim=-2)[:, 1:]  # [B, K], CLS key removed

    selected = key_importance[0].argsort(descending=True)[:num_tokens]
    return [int(t.detach().cpu().item()) for t in selected]


def patchfool_attention_objective(attn_maps, attack_token_indices, target_layers=None):
    """
    Patch-Fool attention objective, to be MAXIMIZED.

    For every layer, averages log A[q, t] over heads-averaged query rows q,
    where t is an attacked token. By default it uses the first half of the
    layers, skipping layer 0, as in the official Patch-Fool code.
    """
    if target_layers is None:
        target_layers = [l for l in range(len(attn_maps) // 2) if l != 0]

    objectives = []
    eps = 1e-12

    for layer in target_layers:
        attn_mean = attn_maps[layer].mean(dim=1)
        attn_flat = attn_mean.reshape(-1, attn_mean.shape[-1]).clamp_min(eps)

        per_token_scores = []
        for token in attack_token_indices:
            target = torch.full(
                size=(attn_flat.shape[0],),
                fill_value=int(token) + 1,
                device=attn_flat.device,
                dtype=torch.long,
            )

            # F.nll_loss(-log(attn), target) = mean_q log(attn[q, target]).
            per_token_scores.append(F.nll_loss(-torch.log(attn_flat), target))

        objectives.append(torch.stack(per_token_scores).mean())

    return torch.stack(objectives).mean()


def pcgrad_project(aux_grad, main_grad, eps=1e-12):
    """
    PCGrad projection of the attention direction onto the normal plane of the
    classification direction, applied only when the two conflict.
    """
    aux_flat = aux_grad.reshape(aux_grad.shape[0], -1)
    main_flat = main_grad.reshape(main_grad.shape[0], -1)

    dot = (aux_flat * main_flat).sum(dim=1, keepdim=True)
    main_norm_sq = (main_flat * main_flat).sum(dim=1, keepdim=True).clamp_min(eps)

    projected = aux_flat.clone()
    conflict = dot.squeeze(1) <= 0

    if conflict.any():
        beta = dot[conflict] / main_norm_sq[conflict]
        projected[conflict] = aux_flat[conflict] - beta * main_flat[conflict]

    return projected.reshape_as(aux_grad)


def apply_random_token_mask(x_01, patch_size=16, mask_ratio=0.05, fill_value=0.5, exclude_token_indices=None):
    """
    EOT transformation: replace each image token with fill_value with
    probability mask_ratio, except the tokens in exclude_token_indices.
    """
    if mask_ratio <= 0:
        return x_01

    B, C, H, W = x_01.shape
    gh = H // patch_size
    gw = W // patch_size

    keep_tokens = torch.rand(B, 1, gh, gw, device=x_01.device) > mask_ratio
    keep_tokens = keep_tokens.to(x_01.dtype)

    if exclude_token_indices is not None:
        for token in exclude_token_indices:
            keep_tokens[:, :, int(token) // gw, int(token) % gw] = 1.0

    keep_mask = keep_tokens.repeat_interleave(patch_size, dim=2)
    keep_mask = keep_mask.repeat_interleave(patch_size, dim=3)
    keep_mask = keep_mask[:, :, :H, :W]

    fill = torch.full_like(x_01, float(fill_value))
    x_masked = x_01 * keep_mask + fill * (1.0 - keep_mask)

    return x_masked.clamp(0, 1)


def optimize_patch_fool(
    model,
    x_clean_01,
    clean_label,
    normalize_fn,
    forward_fn,
    attack_token_indices,
    steps=500,
    lr=5e-2,
    patch_size=16,
    gradient_mode="pcgrad",
    attention_weight=0.02,
    attention_layers=None,
    step_size=10,
    gamma=0.95,
    eot_mask_ratio=0.0,
    eot_samples=1,
    eot_mask_fill_value=0.5,
    eot_mask_attack_tokens=True,
    verbose=True,
):
    """
    Untargeted Patch-Fool attack on a single image.

    The attacked tokens are replaced by a free patch in [0, 1], initialized
    randomly and optimized with Adam + StepLR along the CE-ascent direction
    plus attention_weight times the (PCGrad-projected) attention direction.

    With eot_mask_ratio > 0 the direction is averaged over eot_samples
    random token maskings of the attacked image (Patch-Fool EOT). With
    eot_mask_ratio = 0 this is the plain Patch-Fool attack.
    """
    device = x_clean_01.device

    attack_token_indices = [int(t) for t in attack_token_indices]

    attack_mask = tokens_to_mask(
        attack_token_indices,
        image_size=x_clean_01.shape[-1],
        patch_size=patch_size,
        device=device,
    )

    adv_param = torch.rand_like(x_clean_01).detach().requires_grad_(True)

    optimizer = torch.optim.Adam([adv_param], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    label_tensor = torch.tensor([int(clean_label)], device=device, dtype=torch.long)

    eot_k = max(1, int(eot_samples))
    exclude_from_mask = None if eot_mask_attack_tokens else attack_token_indices
    use_attention = attention_weight != 0.0

    history = {"ce": [], "pred": []}

    for step in range(int(steps)):
        final_direction = torch.zeros_like(adv_param)
        ce_loss_accum = 0.0

        for _ in range(eot_k):
            x_adv_01 = x_clean_01 * (1.0 - attack_mask) + adv_param.clamp(0.0, 1.0) * attack_mask

            if eot_mask_ratio > 0:
                x_eot_01 = apply_random_token_mask(
                    x_adv_01,
                    patch_size=patch_size,
                    mask_ratio=eot_mask_ratio,
                    fill_value=eot_mask_fill_value,
                    exclude_token_indices=exclude_from_mask,
                )
            else:
                x_eot_01 = x_adv_01

            logits, attn_maps = forward_fn(model, normalize_fn(x_eot_01))

            ce_loss = F.cross_entropy(logits, label_tensor)
            direction = torch.autograd.grad(ce_loss, adv_param, retain_graph=use_attention)[0]

            if use_attention:
                attention_objective = patchfool_attention_objective(
                    attn_maps,
                    attack_token_indices,
                    target_layers=attention_layers,
                )
                attention_direction = torch.autograd.grad(attention_objective, adv_param)[0]

                if gradient_mode == "pcgrad":
                    attention_direction = pcgrad_project(attention_direction, direction)

                direction = direction + attention_weight * attention_direction

            final_direction = final_direction + direction
            ce_loss_accum += float(ce_loss.detach().cpu().item())

        final_direction = final_direction / float(eot_k)

        # Adam minimizes, final_direction is an attack-improving direction.
        optimizer.zero_grad(set_to_none=True)
        adv_param.grad = -final_direction
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            adv_param.data = adv_param.data.clamp(0.0, 1.0)

        pred = int(logits.argmax(dim=1).item())
        history["ce"].append(ce_loss_accum / float(eot_k))
        history["pred"].append(pred)

        if verbose and (step % 50 == 0 or step == steps - 1):
            print(f"[Patch-Fool] step {step + 1:4d}/{steps} | ce={history['ce'][-1]:.4f} | pred={pred}")

    with torch.no_grad():
        x_adv_01 = x_clean_01 * (1.0 - attack_mask) + adv_param.clamp(0.0, 1.0) * attack_mask

    return {
        "x_adv_01": x_adv_01.detach(),
        "attack_mask": attack_mask,
        "attack_token_indices": attack_token_indices,
        "history": history,
    }
