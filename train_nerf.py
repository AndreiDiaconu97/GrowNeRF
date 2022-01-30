import argparse
import glob
import itertools
import os
import time
from datetime import timedelta
from math import sqrt, ceil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from grownet.grownet import DynamicNet
from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf)


# TODO: improve config (epochs, stages, ...)
# TODO: implement ensemble checkpointing
# TODO: improve Wandb logging (n_params, ...)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    hwf, USE_CACHED_DATASET, data_dict = load_data(cfg)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Setup tensorboard logging. # OLD
    # logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    # os.makedirs(logdir, exist_ok=True)
    # writer = SummaryWriter(logdir)
    # # Write out config parameters.
    # with open(os.path.join(logdir, "config.yml"), "w") as f:
    #     f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    wandb_cfg = {
        "project": "GrowNeRF",
        "entity": "a-di",
        "mode": "online",  # ["online", "offline", "disabled"]
        "tags": ["grownet"],
        "group": None,  # "exp_1",
        "job_type": None,
        "id": None
    }
    wandb.init(**wandb_cfg, dir=cfg.experiment.logdir, config=cfg)
    wandb.config.USE_CACHED_DATASET = USE_CACHED_DATASET
    if not os.path.isdir(os.path.join(cfg.experiment.logdir, wandb.run.name)):
        os.mkdir(os.path.join(cfg.experiment.logdir, wandb.run.name))

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0
    # Load an existing checkpoint, if a path is specified.
    # if os.path.exists(configargs.load_checkpoint):
    #     checkpoint = torch.load(configargs.load_checkpoint)
    #     model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    #     if checkpoint["model_fine_state_dict"]:
    #         model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     start_iter = checkpoint["iter"]

    ###################################################################################################################
    ### ENSEMBLE TRAINING #############################################################################################
    net_ensemble_coarse = DynamicNet(c0=torch.Tensor([0.0, 0.0, 0.0, 0.0]).to(device), lr=cfg.experiment.boost_rate)
    net_ensemble_fine = DynamicNet(c0=torch.Tensor([0.0, 0.0, 0.0, 0.0]).to(device), lr=cfg.experiment.boost_rate)

    for stage in range(cfg.experiment.n_stages):
        weak_model_coarse, weak_model_fine = train_weak(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, start_iter, net_ensemble_coarse, net_ensemble_fine,
                                                        stage)
        net_ensemble_coarse.add(weak_model_coarse)
        net_ensemble_fine.add(weak_model_fine)

        if stage > 0:
            fully_corrective_steps(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, start_iter, net_ensemble_coarse, net_ensemble_fine, stage)

        validate(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, stage, net_ensemble_coarse, net_ensemble_fine)
    ###################################################################################################################
    print("Done!")


def train_weak(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, start_iter, net_ensemble_coarse, net_ensemble_fine, stage):
    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        skip_connect_every=cfg.models.coarse.skip_connect_every,
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        append_penultimate=stage
    )
    model_coarse.to(device)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_layers=cfg.models.fine.num_layers,
            hidden_size=cfg.models.fine.hidden_size,
            skip_connect_every=cfg.models.fine.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            append_penultimate=stage
        )
        model_fine.to(device)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    net_ensemble_coarse.to_train()  # return models to train mode after validation
    if net_ensemble_fine:
        net_ensemble_fine.to_train()

    # for i in trange(start_iter, cfg.experiment.weak_train_iters):
    pbar = tqdm(range(cfg.experiment.weak_train_iters), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{cfg.experiment.n_stages}: weak model training", unit=" epoch")
    for epoch in pbar:
        model_coarse.train()  # return models to train mode after validation
        if model_fine:
            model_coarse.train()

        hwf, ray_directions, ray_origins, target_ray_values = get_random_rays(cfg, hwf, USE_CACHED_DATASET, device, data_dict)
        grad_direction_coarse, grad_direction_fine = target_ray_values, target_ray_values
        penultimate_features_coarse, penultimate_features_fine, ray_batches, pts_and_zvals = None, None, None, None

        if stage > 0:
            out_ensemble, penultimate_features, ray_batches, pts_and_zvals = run_one_iter_of_nerf(
                hwf,
                net_ensemble_coarse,
                net_ensemble_fine,
                ray_origins,
                ray_directions,
                cfg,
                "train",
                encode_position_fn,
                encode_direction_fn
            )
            rgb_coarse_ensemble, _, _, rgb_fine_ensemble, _, _ = out_ensemble  # predictions, one pixel for each ray (default: 1024 pixels)
            penultimate_features_coarse, penultimate_features_fine = penultimate_features
            grad_direction_coarse = -(rgb_coarse_ensemble - target_ray_values)  # grad_direction = y / (1.0 + torch.exp(y * out))
            grad_direction_fine = -(rgb_fine_ensemble - target_ray_values)

        out, _, _, _ = run_one_iter_of_nerf(
            hwf,
            model_coarse,
            model_fine,
            ray_origins,
            ray_directions,
            cfg,
            "train",
            encode_position_fn,
            encode_direction_fn,
            penultimate_features_coarse,
            penultimate_features_fine,
            ray_batches,
            pts_and_zvals,
        )
        rgb_coarse, _, _, rgb_fine, _, _ = out  # predictions, one pixel for each ray (default: 1024 pixels)

        # LOSS + UPDATE #
        coarse_loss = F.mse_loss(
            net_ensemble_coarse.boost_rate * rgb_coarse[..., :3], grad_direction_coarse[..., :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = F.mse_loss(
                net_ensemble_fine.boost_rate * rgb_fine[..., :3], grad_direction_fine[..., :3]
            )

        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        lr_old = optimizer.param_groups[0]["lr"]
        weak_train_log(cfg, device, epoch + 1, stage, loss, coarse_loss, fine_loss, psnr, lr_old, pbar, rgb_coarse, rgb_fine, grad_direction_coarse, grad_direction_fine)

        # UPDATE LR #
        num_decay_steps = cfg.scheduler.lr_decay * 1000  # * 1000
        lr_new = cfg.optimizer.lr * (
                cfg.scheduler.lr_decay_factor ** (epoch / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        # Weak model Validation #
        # if i % cfg.experiment.validate_every == 0 or i == cfg.experiment.weak_train_iters - 1:
        #     loss, psnr = validate(cfg, H, W, focal, USE_CACHED_DATASET, device, encode_direction_fn, encode_position_fn, i, i_val, images, model_coarse, model_fine, poses, validation_paths, writer)

        # if i % cfg.experiment.save_every == 0 or i == cfg.experiment.weak_train_iters - 1:
        #     save_checkpoint(i, logdir, loss, model_coarse, model_fine, optimizer, psnr)

    return model_coarse, model_fine


def fully_corrective_steps(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, start_iter, net_ensemble_coarse, net_ensemble_fine, stage):
    # Initialize optimizer.
    trainable_parameters = list(net_ensemble_coarse.parameters())
    if net_ensemble_fine is not None:
        trainable_parameters += list(net_ensemble_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr_ensemble
    )

    net_ensemble_coarse.to_train()  # return models to train mode after validation
    if net_ensemble_fine:
        net_ensemble_fine.to_train()

    # print("CORRECTIVE STEPS...")
    pbar = tqdm(range(cfg.experiment.corrective_iters), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{cfg.experiment.n_stages}: fully corrective steps",
                unit=" epoch")
    for epoch in pbar:
        hwf, ray_directions, ray_origins, target_ray_values = get_random_rays(cfg, hwf, USE_CACHED_DATASET, device, data_dict)

        out_ensemble, _, _, _ = run_one_iter_of_nerf(
            hwf,
            net_ensemble_coarse.forward_grad,
            net_ensemble_fine.forward_grad,
            ray_origins,
            ray_directions,
            cfg,
            "train",
            encode_position_fn,
            encode_direction_fn,
            ray_batches=None,
            pts_and_zvals=None
        )
        rgb_coarse_ensemble, _, _, rgb_fine_ensemble, _, _ = out_ensemble  # predictions, one pixel for each ray (default: 1024 pixels)

        coarse_loss = F.mse_loss(
            rgb_coarse_ensemble[..., :3], target_ray_values[..., :3]
        )
        fine_loss = None
        if rgb_fine_ensemble is not None:
            fine_loss = F.mse_loss(
                rgb_fine_ensemble[..., :3], target_ray_values[..., :3]
            )

        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        psnr = mse2psnr(loss.item())
        lr_old = optimizer.param_groups[0]["lr"]
        corrective_step_log(cfg, coarse_loss, epoch, fine_loss, loss, lr_old, pbar, psnr, stage, net_ensemble_coarse, net_ensemble_fine)

        # UPDATE LR #
        num_decay_steps = cfg.scheduler.lr_decay * 1000  # * 1000
        lr_new = cfg.optimizer.lr * (
                cfg.scheduler.lr_decay_factor ** (epoch / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new


def get_random_rays(cfg, hwf, USE_CACHED_DATASET, device, data_dict):
    """
    Collect a bunch of random rays (default:1024) with ground truth from a single random image.\n
    Rays are defined by `ray_origins[]` and `ray_directions[]`.

    :param nerf.cfgnode.CfgNode cfg: dictionary with all config parameters
    :param list[int,int,float] hwf: list containing image height, width, and cameras focal length
    :param bool USE_CACHED_DATASET: cache folder must exist for it to have an effect
    :param str device: usually "cuda" or "cpu"
    :param data_dict: contains i_train, images, poses and train_paths
        - None | list[int] i_train: indices of data images used for training, ignored if using cached dataset
        - None | torch.Tensor images: loaded dataset [H,W,N,RGB], ignored if using cached dataset
        - None | torch.Tensor poses: camera parameters, one camera for each image. Ignored if using cached dataset
        - list[str] | None train_paths: one path for each training image, ignored if dataset is not cached
    :return: hwf, ray_directions, ray_origins, target_ray_values
    :rtype: tuple[ list[int,int,float], torch.Tensor, torch.Tensor, torch.Tensor]
    """

    if USE_CACHED_DATASET:
        datafile = np.random.choice(data_dict["train_paths"])
        cache_dict = torch.load(datafile)
        H = cache_dict["height"]
        W = cache_dict["width"]
        focal = cache_dict["focal_length"]
        if not hwf:
            wandb.config.HWF = [H, W, focal]
        hwf = H, W, focal
        ray_bundle = cache_dict["ray_bundle"].to(device)
        ray_origins, ray_directions = (
            ray_bundle[0].reshape((-1, 3)),
            ray_bundle[1].reshape((-1, 3)),
        )
        target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
        select_inds = np.random.choice(
            ray_origins.shape[0],
            size=(cfg.nerf.train.num_random_rays),
            replace=False,
        )
        ray_origins, ray_directions = (
            ray_origins[select_inds],
            ray_directions[select_inds],
        )
        target_ray_values = target_ray_values[select_inds].to(device)
        # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)
    else:
        H, W, focal = hwf
        img_idx = np.random.choice(data_dict["i_train"])
        img_target = data_dict["images"][img_idx].to(device)
        pose_target = data_dict["poses"][img_idx, :3, :4].to(device)
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
        coords = torch.stack(
            meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
            dim=-1,
        )
        coords = coords.reshape((-1, 2))
        select_inds = np.random.choice(
            coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
        )
        select_inds = coords[select_inds]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
        target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
        target_ray_values = target_s

    return hwf, ray_directions, ray_origins, target_ray_values


def weak_train_log(cfg, device, epoch, stage, loss, coarse_loss, fine_loss, psnr, lr, pbar, rgb_coarse, rgb_fine, target_ray_values_coarse, target_ray_values_fine):
    # if i % cfg.experiment.print_every == 0 or i == cfg.experiment.weak_train_iters - 1:
    #     tqdm.write(
    #         "[TRAIN] Iter: "
    #         + str(i)
    #         + " Loss: "
    #         + str(loss.item())
    #         + " PSNR: "
    #         + str(psnr)
    #     )

    if epoch % cfg.experiment.print_every == 0 or epoch == cfg.experiment.weak_train_iters - 1:
        if fine_loss is not None:
            wandb.log({
                "Weak/train/loss_fine": fine_loss.item(),
            }, commit=False)

            pbar.set_postfix({
                'lr': lr,
                'psnr': psnr,
                'loss': loss.item(),
                'loss_coarse': coarse_loss.item(),
                'loss_fine': fine_loss.item(),
                # 'n_params': _n_params,
            })
        else:
            pbar.set_postfix({
                'lr': lr,
                'psnr': psnr,
                'loss': loss.item(),
                'loss_coarse': coarse_loss.item(),
                # 'n_params': _n_params,
            })

        wandb.log({
            "epoch": epoch,
            # "stage": stage,
            "Weak/train/lr": lr,
            "Weak/train/loss": loss.item(),
            "Weak/train/loss_coarse": coarse_loss.item(),
            "Weak/train/psnr": psnr,
        }, commit=True)

    if epoch % 100 == 0:  # save grid of predictions
        tensors = [rgb_coarse, target_ray_values_coarse]
        if fine_loss is not None:
            tensors += [rgb_fine, target_ray_values_fine]

        imgs = []
        w = ceil(sqrt(cfg.nerf.train.num_random_rays))

        for t in tensors:
            trail = torch.zeros((w ** 2 - t.shape[0], 3)).to(device)
            t = torch.cat((t, trail))
            imgs.append(cv2.cvtColor(t[..., :3].view(w, w, 3).detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
        img_grid = get_img_grid(2, len(tensors) // 2, len(tensors), imgs, margin=1)

        cv2.imwrite(f'{cfg.experiment.logdir}/{wandb.run.name}/{epoch + (stage * cfg.experiment.weak_train_iters)}_weak.png', img_grid)


def get_img_grid(h, w, n, images, margin=1):  # from internet
    if len(images) != n:
        raise ValueError('Number of images ({}) does not match '
                         'matrix size {}x{}'.format(len(images), w, h))

    imgs = images

    if any(i.shape != imgs[0].shape for i in imgs[1:]):
        raise ValueError('Not all images have the same shape.')

    img_h, img_w, img_c = imgs[0].shape

    m_x = 0
    m_y = 0
    if margin is not None:
        m_x = int(margin)
        m_y = m_x

    imgmatrix = np.zeros((img_h * h + m_y * (h - 1),
                          img_w * w + m_x * (w - 1),
                          img_c),
                         np.uint8)

    imgmatrix.fill(255)

    imgmatrix = np.zeros((img_h * h + m_y * (h - 1),
                          img_w * w + m_x * (w - 1),
                          img_c),
                         np.uint8)

    imgmatrix.fill(255)

    positions = itertools.product(range(w), range(h))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y + img_h, x:x + img_w, :] = img

    return imgmatrix


def corrective_step_log(cfg, coarse_loss, epoch, fine_loss, loss, lr, pbar, psnr, stage, net_ensemble_coarse, net_ensemble_fine):
    if epoch % cfg.experiment.print_every == 0 or epoch == cfg.experiment.weak_train_iters - 1:
        if fine_loss is not None:
            wandb.log({
                "Ensemble/corrective/loss_fine": fine_loss.item(),
                "Ensemble/corrective/boost_rate_fine": net_ensemble_fine.boost_rate.item()
            }, commit=False)

            pbar.set_postfix({
                'lr': lr,
                'psnr': psnr,
                'loss': loss.item(),
                'loss_coarse': coarse_loss.item(),
                'loss_fine': fine_loss.item(),
                # 'n_params': _n_params,
            })
        else:
            pbar.set_postfix({
                'lr': lr,
                'psnr': psnr,
                'loss': loss.item(),
                'loss_coarse': coarse_loss.item(),
                # 'n_params': _n_params,
            })

        wandb.log({
            "epoch": epoch,
            "stage": stage,
            "Ensemble/corrective/lr": lr,
            "Ensemble/corrective/loss": loss.item(),
            "Ensemble/corrective/loss_coarse": coarse_loss.item(),
            "Ensemble/corrective/psnr": psnr,
            "Ensemble/corrective/boost_rate_coarse": net_ensemble_coarse.boost_rate.item()
        }, commit=True)


def save_checkpoint(i, logdir, loss, model_coarse, model_fine, optimizer, psnr):
    checkpoint_dict = {
        "iter": i,
        "model_coarse_state_dict": model_coarse.state_dict(),
        "model_fine_state_dict": None
        if not model_fine
        else model_fine.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "psnr": psnr,
    }
    torch.save(
        checkpoint_dict,
        os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
    )
    tqdm.write("================== Saved Checkpoint =================")


def validate(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, i, model_coarse, model_fine):
    # tqdm.write("[VAL] =======> Iter: " + str(i))
    # model_coarse.eval()
    # if model_fine:
    #     model_coarse.eval()
    model_coarse.to_eval()
    if model_fine:
        model_coarse.to_eval()

    start = time.time()
    with torch.no_grad():
        if USE_CACHED_DATASET:
            datafile = np.random.choice(data_dict["validation_paths"])
            cache_dict = torch.load(datafile)
            H = cache_dict["height"]
            W = cache_dict["width"]
            focal = cache_dict["focal_length"]
            hwf = H, W, focal
            ray_origins = cache_dict["ray_origins"].to(device)
            ray_directions = cache_dict["ray_directions"].to(device)
            target_ray_values = cache_dict["target"].to(device)
        else:
            img_idx = np.random.choice(data_dict["i_val"])
            img_target = data_dict["images"][img_idx].to(device)
            target_ray_values = img_target
            pose_target = data_dict["poses"][img_idx, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(*hwf, pose_target)

        rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
            hwf,
            model_coarse,
            model_fine,
            ray_origins,
            ray_directions,
            cfg,
            mode="validation",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
        )

        coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
        loss, fine_loss = None, None
        if rgb_fine is not None:
            fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
        loss = coarse_loss + fine_loss
        psnr = mse2psnr(loss.item())

        if fine_loss is not None:
            wandb.log({
                "Ensemble/validation/loss_fine": fine_loss,
            }, commit=False)

        wandb.log({
            "stage": i,
            "Ensemble/validation/loss": loss.item(),
            "Ensemble/validation/loss_coarse": coarse_loss.item(),
            "Ensemble/validation/psnr": psnr,
        }, commit=True)

        # SAVE validation prediction on file #
        cv2.imwrite(f'{cfg.experiment.logdir}/{wandb.run.name}/_ensVal_rgb_coarse{i}.png', cv2.cvtColor(rgb_coarse[..., :3].detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f'{cfg.experiment.logdir}/{wandb.run.name}/_ensVal_target_rgb{i}.png', cv2.cvtColor(target_ray_values[..., :3].detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
        if rgb_fine is not None:
            cv2.imwrite(f'{cfg.experiment.logdir}/{wandb.run.name}/_ensVal_rgb_fine{i}.png', cv2.cvtColor(rgb_fine[..., :3].detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f'{cfg.experiment.logdir}/{wandb.run.name}/_ensVal_grad{i}.png',
                    cv2.cvtColor((target_ray_values[..., :3].detach() - rgb_coarse[..., :3].detach()).cpu().numpy() * 255, cv2.COLOR_BGR2RGB))

        tqdm.write(
            f"Validation loss: {loss.item():.5f}"
            + f" Validation PSNR: {psnr:.5f}"
            + f" Time: {time.time() - start:.5f}"
        )


def load_data(cfg):
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
        print("Found cache.")
    else:
        # Load dataset
        print("No cache found or set, loading dataset...")
        images, poses, render_poses, hwf = None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            images, poses, render_poses, hwf, i_split = load_blender_data(
                cfg.dataset.basedir,
                half_res=cfg.dataset.half_res,
                testskip=cfg.dataset.testskip,
            )
            i_train, i_val, i_test = i_split  # select data indices for training, validation, and testing
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        elif cfg.dataset.type.lower() == "llff":
            images, poses, bds, render_poses, i_test = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
            i_val = i_test
            i_train = np.array(
                [
                    i
                    for i in np.arange(images.shape[0])
                    if (i not in i_test and i not in i_val)
                ]
            )
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            images = torch.from_numpy(images)
            poses = torch.from_numpy(poses)

    data_dict = {
        "i_train": i_train,
        "i_val": i_val,
        "images": images,
        "poses": poses,
        "train_paths": train_paths,
        "validation_paths": validation_paths
    }

    return hwf, USE_CACHED_DATASET, data_dict


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    start = time.time()
    main()
    print("seconds: ", time.time() - start)
