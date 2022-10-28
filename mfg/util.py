from typing import TYPE_CHECKING, Tuple
import torch


def assert_zero_grads(network: torch.nn.Module) -> None:
    for param in network.parameters():
        if not (param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))): debug()
        assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))

@torch.no_grad()
def proj_pca(xs_f: torch.Tensor, xs_b: torch.Tensor, reverse: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    # xs: (batch, T, nx)
    # Only use final timestep of xs_f for PCA.
    batch, T, nx = xs_f.shape

    # (batch * T, nx)
    flat_xsf = xs_f.reshape(-1, *xs_f.shape[2:])
    flat_xsb = xs_b.reshape(-1, *xs_b.shape[2:])

    # Center by subtract mean.
    # (batch, nx)
    if reverse:
        # If reverse, use xs_b[0] instead of xs_f[T]
        final_xs_f = xs_b[:, 0, :]
    else:
        final_xs_f = xs_f[:, -1, :]

    mean_pca_xs = torch.mean(final_xs_f, dim=0, keepdim=True)
    final_xs_f -= mean_pca_xs

    # if batch is too large, it will run out of memory.
    if batch > 200:
        rand_idxs = torch.randperm(batch)[:200]
        final_xs_f = final_xs_f[rand_idxs]

    # U: (batch, k)
    # S: (k, k)
    # VT: (k, nx)
    U, S, VT = torch.linalg.svd(final_xs_f)

    # log.info("Singular values of xs_f at final timestep:")
    # log.info(S)

    # Keep the first and last directions.
    VT = VT[:2, :]
    # VT = VT[[0, -1], :]

    assert VT.shape == (2, nx)
    V = VT.T

    # Project both xs_f and xs_b onto V.
    flat_xsf -= mean_pca_xs
    flat_xsb -= mean_pca_xs

    proj_xs_f = flat_xsf @ V
    proj_xs_f = proj_xs_f.reshape(batch, T, *proj_xs_f.shape[1:])

    proj_xs_b = flat_xsb @ V
    proj_xs_b = proj_xs_b.reshape(batch, T, *proj_xs_b.shape[1:])

    return proj_xs_f, proj_xs_b