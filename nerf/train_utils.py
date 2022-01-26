import torch

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .volume_rendering_utils import volume_render_radiance_field


def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn, penultimate_features=None):
    """
    Batchify the sampled points `"pts"` and forward network `"network_fn"` for each batch.

    :param torch.nn.Module network_fn: network to forward (usually either the coarse or fine NeRF)
    :param torch.Tensor pts: point samples to query (Rays per image * samples per ray)
    :param torch.Tensor ray_batch: 11D rays used for the sampled points
    :param int chunksize: maximum number of sample points per batch
    :param None | function embed_fn: positional encoding of points positions
    :param None | function embeddirs_fn: positional encoding of rays view directions
    :param None | torch.Tensor penultimate_features: features to forward along points, only used by weak learners during training (after it is added to the ensemble, the feature propagation is done by the ensemble class)
    :return: radiance_field, radiance_field_penultimate
    :rtype: tuple[torch.Tensor]
    """

    ### Apply positional encoding ##################################
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)
    ################################################################

    batches = get_minibatches(embedded, chunksize=chunksize)  # group sampled points into smaller batches
    if penultimate_features is not None:
        penultimate_features_flat = penultimate_features.reshape(-1, penultimate_features.shape[-1])
        penultimate_features_batches = get_minibatches(penultimate_features_flat, chunksize=chunksize)
        preds = [network_fn(batch, penultimate_features_batches[i]) for i, batch in enumerate(batches)]
    else:
        preds = [network_fn(batch) for batch in batches]

    preds_penultimate = [p[0] for p in preds]
    preds = [p[1] for p in preds]

    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )

    radiance_field_penultimate = torch.cat(preds_penultimate, dim=0)
    radiance_field_penultimate = radiance_field_penultimate.reshape(
        list(pts.shape[:-1]) + [radiance_field_penultimate.shape[-1]]
    )
    return radiance_field, radiance_field_penultimate


def predict_and_render_radiance(
        ray_batch,
        model_coarse,
        model_fine,
        options,
        mode="train",
        encode_position_fn=None,
        encode_direction_fn=None,
        penultimate_features_coarse=None,
        penultimate_features_fine=None,
        pts_and_zvals=None,
):
    """
    Predict a subsample of an image from a batch of rays, one pixel for each. (default:1024 pixels).
        - this function relies on `run_network()` for inferencing a subset of the radiance field ([Rays, Points,RGBa] where Rays are from `ray_batch`),and on `volume_render_radiance_field()` for rendering the predicted subset of the radiance field
        - this function also samples the points for both coarse and fine models

    :param torch.Tensor ray_batch: batch of rays to be sampled (default: 1024)
    :param torch.nn.Module model_coarse:
    :param None | torch.nn.Module model_fine:
    :param nerf.cfgnode.CfgNode options: dictionary with all config parameters
    :param str mode: "train" or "validation"
    :param None | function encode_position_fn: positional encoding of points positions
    :param None | function encode_direction_fn: positional encoding of rays view directions
    :param None | torch.Tensor penultimate_features_coarse: is used only by weak learners, they connect to the penultimate features of the last learner of the ensemble for training
    :param None | torch.Tensor penultimate_features_fine: is used only by weak learners, they connect to the penultimate features of the last learner of the ensemble for training
    :param None | list[tuple[torch.Tensor, torch.Tensor]] pts_and_zvals: sampled points and their intervals on rays for each model, recycled by weak learners because recomputation is affected by random noise (would sample different points)
    :return: rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
    :rtype: tuple[torch.Tensor]
    """

    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    ###################################################################################################################
    ### Sample points for the coarse network (regular intervals + optional small perturbation) ########################
    if not pts_and_zvals:
        pts_and_zvals = [None, None]
    if not pts_and_zvals[0]:
        # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
        # when not enabling "ndc".
        t_vals = torch.linspace(
            0.0,
            1.0,
            getattr(options.nerf, mode).num_coarse,
            dtype=ro.dtype,
            device=ro.device,
        )
        if not getattr(options.nerf, mode).lindisp:
            z_vals = near * (1.0 - t_vals) + far * t_vals
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
        z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

        if getattr(options.nerf, mode).perturb:  # add NOISE in the intervals size
            # Get intervals between samples.
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
            lower = torch.cat((z_vals[..., :1], mids), dim=-1)
            # Stratified samples in those intervals.
            t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
            z_vals = lower + (upper - lower) * t_rand
        # pts -> (num_rays, N_samples, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
        pts_and_zvals[0] = pts, z_vals
    else:
        pts, z_vals = pts_and_zvals[0]
    ###################################################################################################################

    radiance_field_penultimate_coarse = None
    radiance_field, radiance_field_penultimate_coarse = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
        penultimate_features_coarse
    )

    (
        rgb_coarse,
        disp_coarse,
        acc_coarse,
        weights,
        depth_coarse,
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
    )

    rgb_fine, disp_fine, acc_fine, radiance_field_penultimate_fine = None, None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        ###############################################################################################################
        ### Sample points for the fine network ########################################################################
        if not pts_and_zvals[1]:
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid,
                weights[..., 1:-1],
                getattr(options.nerf, mode).num_fine,
                det=(getattr(options.nerf, mode).perturb == 0.0),
            )
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
            # pts -> (N_rays, N_samples + N_importance, 3)
            pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
            pts_and_zvals[1] = pts, z_vals
        else:
            pts, z_vals = pts_and_zvals[1]
        ###############################################################################################################

        radiance_field, radiance_field_penultimate_fine = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
            penultimate_features_fine
        )

        rgb_fine, disp_fine, acc_fine, _, _ = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
        )

    if mode == "validation":
        radiance_field_penultimate_coarse, radiance_field_penultimate_fine, pts_and_zvals = None, None, None  # We don't need these during validation

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, radiance_field_penultimate_coarse, radiance_field_penultimate_fine, pts_and_zvals


def run_one_iter_of_nerf(
        hwf,
        model_coarse,
        model_fine,
        ray_origins,
        ray_directions,
        options,
        mode="train",
        encode_position_fn=None,
        encode_direction_fn=None,
        penultimate_features_coarse=None,
        penultimate_features_fine=None,
        ray_batches=None,
        pts_and_zvals=None
):
    """
    Predict a subsample of an image from a bunch of rays, one pixel for each. (default:1024 pixels).\n
    Build 11D rays (`ro`=3D, `rd`=3D, `near`=1D, `far`=1D, `viewdirs`=3D), and batch them before giving to `predict_and_render_radiance()` for predictions retrieval

    :param int height: of image
    :param int width: of image
    :param float focal_length: of cameras
    :param torch.nn.Module model_coarse:
    :param None | torch.nn.Module model_fine:
    :param torch.Tensor ray_origins:
    :param torch.Tensor ray_directions:
    :param nerf.cfgnode.CfgNode options: dictionary with all config parameters
    :param str mode: "train" or "validation"
    :param None | function encode_position_fn: positional encoding of points positions
    :param None | function encode_direction_fn: positional encoding of rays view directions
    :param None | torch.Tensor penultimate_features_coarse: is used only by weak learners, they connect to the penultimate features of the last learner of the ensemble for training
    :param None | torch.Tensor penultimate_features_fine: is used only by weak learners, they connect to the penultimate features of the last learner of the ensemble for training
    :param None | list[torch.Tensor] ray_batches: pre-batched rays, recycled by weak learners from the ensemble in order to avoid redundant computations
    :param None | tuple[torch.Tensor, torch.Tensor] pts_and_zvals: sampled points and their intervals on rays for each model, recycled by weak learners because recomputation is affected by random noise (would sample different points)
    :return: rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, None | torch.Tensor, None | torch.Tensor, None | torch.Tensor]
    """

    if not ray_batches:
        ###############################################################################################################
        ### Prepare rays ##############################################################################################
        viewdirs = None
        if options.nerf.use_viewdirs:
            # Provide ray directions as input
            viewdirs = ray_directions
            viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
            viewdirs = viewdirs.view((-1, 3))
        # Cache shapes now, for later restoration.
        restore_shapes = [
            ray_directions.shape,
            ray_directions.shape[:-1],
            ray_directions.shape[:-1],
        ]
        if model_fine:
            restore_shapes += restore_shapes
        if options.dataset.no_ndc is False:
            ro, rd = ndc_rays(*hwf, 1.0, ray_origins, ray_directions)
            ro = ro.view((-1, 3))
            rd = rd.view((-1, 3))
        else:
            ro = ray_origins.view((-1, 3))
            rd = ray_directions.view((-1, 3))
        near = options.dataset.near * torch.ones_like(rd[..., :1])
        far = options.dataset.far * torch.ones_like(rd[..., :1])
        rays = torch.cat((ro, rd, near, far), dim=-1)
        if options.nerf.use_viewdirs:
            rays = torch.cat((rays, viewdirs), dim=-1)
        ###############################################################################################################

        ray_batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)  # INFO: this is the same function as in "run_network()", why is needed here?

    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            mode,
            encode_position_fn,
            encode_direction_fn,
            penultimate_features_coarse,
            penultimate_features_fine,
            pts_and_zvals
        )
        for batch in ray_batches
    ]

    final_preds = list(zip(*pred))
    radiance_field_penultimate_coarse, radiance_field_penultimate_fine = [f[0] for f in final_preds[-3:-1]]
    penultimate_features = radiance_field_penultimate_coarse, radiance_field_penultimate_fine
    pts_and_zvals = final_preds[-1][0]
    synthesized_images = final_preds[:-3]
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)  # Remove empty elements from the list
        for image in synthesized_images
    ]

    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images), penultimate_features, ray_batches, pts_and_zvals
