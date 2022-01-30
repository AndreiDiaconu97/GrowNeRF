import torch

from .nerf_helpers import cumprod_exclusive


def volume_render_radiance_field(
        radiance_field,
        depth_values,
        ray_directions,
        radiance_field_noise_std=0.0,
        white_background=False,
):
    """
    Render pixel maps from a radiance field given a viewpoint (`ray_directions`),\n
    Returns a color map, a disparity(?) map, an accuracy(?) map, and a depth map.\n
    The return element `samples_weights` is referred to each sampled point, it is in range [0,1], and it is used outside this function for the fine_model PDF sampling.

    :param torch.Tensor radiance_field: tensor of 4D elements (RGB+alpha) predicted by a model, one element for each sample (default:64 samples per ray) of each ray (default:1024 rays per iteration)
    :param torch.Tensor depth_values: 1D position of each sampled point along its ray, for all rays
    :param torch.Tensor ray_directions:
    :param float radiance_field_noise_std: to perturb the samples weights
    :param bool white_background: if False, background is black
    :return: rgb_map, disp_map, acc_map, samples_weights, depth_map
    :rtype: tuple[torch.Tensor]
    """
    # TESTED
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :3])  # torch.tanh(radiance_field[..., :3]) # for negative pixels
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
                torch.randn(
                    radiance_field[..., 3].shape,
                    dtype=radiance_field.dtype,
                    device=radiance_field.device,
                )
                * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    samples_weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = samples_weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = samples_weights * depth_values
    depth_map = depth_map.sum(dim=-1)
    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = samples_weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, samples_weights, depth_map
