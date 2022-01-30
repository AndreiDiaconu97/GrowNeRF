import argparse
import os
import time
from datetime import timedelta
from math import sqrt, ceil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from tqdm import tqdm

from grownet.grownet import DynamicNet
from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse, mse2psnr, run_one_iter_of_nerf)
from train_nerf_utils import load_checkpoint, get_img_grid, save_checkpoint, load_data, get_random_rays, get_model_coarse, get_model_fine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(  # loading checkpoint ignores config argument
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from. Creates separate run",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint. (On wandb too)"
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

    ###################################################################################################################
    ### POSITIONAL ENCODINGS used in run_network() ####################################################################

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

    ###################################################################################################################
    ### INIT + LOAD CHECKPOINT ########################################################################################

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_stage, start_weak_epoch = 0, 0
    weak_model_coarse, weak_model_fine = None, None
    net_ensemble_coarse = DynamicNet(torch.Tensor([0.0, 0.0, 0.0, 0.0]).to(device), cfg.experiment.boost_rate, device)
    net_ensemble_fine = DynamicNet(torch.Tensor([0.0, 0.0, 0.0, 0.0]).to(device), cfg.experiment.boost_rate, device)

    run_id = None
    resume = False
    if os.path.exists(configargs.load_checkpoint):  # Load an existing checkpoint, if a path is specified
        cfg_loaded, run_id, start_stage, start_weak_epoch, weak_model_coarse, weak_model_fine = load_checkpoint(configargs, device, net_ensemble_coarse, net_ensemble_fine)
        cfg = cfg_loaded
        if configargs.resume:  # if resume is given, run is continued from selected checkpoint. A separate run is done otherwise
            resume = "must"
        else:
            run_id = None

    # Setup tensorboard logging. # OLD
    # logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    # os.makedirs(logdir, exist_ok=True)
    # writer = SummaryWriter(logdir)
    # # Write out config parameters.
    # with open(os.path.join(logdir, "config.yml"), "w") as f:
    #     f.write(cfg.dump())  # cfg, f, default_flow_style=False

    ###################################################################################################################
    ### LOGGING INIT ##################################################################################################

    wandb_cfg = {
        "project": "GrowNeRF",
        "entity": "a-di",
        "mode": "online",  # ["online", "offline", "disabled"]
        "tags": ["grownet"],
        "group": None,  # "exp_1",
        "job_type": None,
        "id": run_id,
        "resume": resume
    }
    wandb.init(**wandb_cfg, dir=cfg.experiment.logdir, config=cfg)
    wandb.config.USE_CACHED_DATASET = USE_CACHED_DATASET
    if not os.path.isdir(os.path.join(cfg.experiment.logdir, wandb.run.name)):
        os.mkdir(os.path.join(cfg.experiment.logdir, wandb.run.name))

    ###################################################################################################################
    ### ENSEMBLE TRAINING #############################################################################################

    for stage in range(start_stage, cfg.experiment.n_stages):
        if stage == start_stage:  # check checkpoint loaded only at first iteration
            weak_model_coarse, weak_model_fine = train_weak(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, start_weak_epoch, net_ensemble_coarse,
                                                            net_ensemble_fine, stage, weak_model_coarse, weak_model_fine)
        else:
            weak_model_coarse, weak_model_fine = train_weak(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, 0, net_ensemble_coarse,
                                                            net_ensemble_fine, stage)
        net_ensemble_coarse.add(weak_model_coarse)
        net_ensemble_fine.add(weak_model_fine)

        if stage > 0:
            fully_corrective_steps(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, net_ensemble_coarse, net_ensemble_fine, stage)

        validate(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, stage, net_ensemble_coarse, net_ensemble_fine)
    ###################################################################################################################
    print("- DONE -")


def train_weak(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn,
               start_epoch, net_ensemble_coarse, net_ensemble_fine, stage, weak_model_coarse=None, weak_model_fine=None):
    if weak_model_coarse:
        model_coarse = weak_model_coarse
        model_fine = weak_model_fine  # no problem if is None
    else:  # if no existent coarse, create both coarse and fine models
        model_coarse = get_model_coarse(cfg, stage)
        model_coarse.to(device)
        model_fine = get_model_fine(cfg, stage)
    if model_fine:
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
    pbar = tqdm(range(start_epoch, cfg.experiment.weak_train_iters), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{cfg.experiment.n_stages}: weak model training",
                unit=" epoch")
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

        # SAVE CHECKPOINT #
        if epoch % cfg.experiment.save_every == 0 or epoch == cfg.experiment.weak_train_iters - 1:
            save_checkpoint(cfg, stage, epoch, loss, model_coarse, model_fine, net_ensemble_coarse, net_ensemble_fine, optimizer, psnr)

    return model_coarse, model_fine


def fully_corrective_steps(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, net_ensemble_coarse, net_ensemble_fine, stage):
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


if __name__ == "__main__":
    start = time.time()
    main()
    print("seconds: ", time.time() - start)
