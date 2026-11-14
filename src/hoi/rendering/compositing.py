"""Compose neural Gaussian appearance and an opaque mesh by expected depth."""

import torch


def composite(body_premul, body_alpha, body_depth, mesh, background=None):
    if background is None:
        background = torch.ones(3, device=body_premul.device)
    ma = mesh["alpha"]
    mp = mesh["premultiplied_rgb"]
    md = mesh["depth"]
    body_first = (body_alpha > 1e-5) & ((ma <= 1e-5) | ((body_depth > 0) & (body_depth < md)))
    front = torch.where(body_first[..., None], body_premul, mp)
    back = torch.where(body_first[..., None], mp, body_premul)
    fa = torch.where(body_first, body_alpha, ma)
    ba = torch.where(body_first, ma, body_alpha)
    alpha = fa + (1 - fa) * ba
    rgb = front + (1 - fa[..., None]) * back + (1 - alpha[..., None]) * background
    return {"rgb": rgb, "alpha": alpha, "body_first": body_first}
