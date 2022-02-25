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
from train_nerf_utils import load_checkpoint, get_img_grid, load_config, save_checkpoint, load_data, get_random_rays, get_model_coarse, get_model_fine

def main():
    cfg, configargs = load_config()
    MAX_MINUTES = configargs.max_mins

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
    resumed_weak_model_coarse, resumed_weak_model_fine = None, None
    net_ensemble_coarse = DynamicNet(torch.Tensor([.0, .0, .0, .0]).to(device), cfg.experiment.boost_rate, device, cfg.experiment.learn_boost_rate, cfg.experiment.propagate_context)
    net_ensemble_fine = DynamicNet(torch.Tensor([.0, .0, .0, .0]).to(device), cfg.experiment.boost_rate, device, cfg.experiment.learn_boost_rate, cfg.experiment.propagate_context)

    run_id = None
    resume = False
    if os.path.exists(configargs.load_checkpoint):  # Load an existing checkpoint, if a path is specified
        cfg_loaded, run_id, start_stage, start_weak_epoch, resumed_weak_model_coarse, resumed_weak_model_fine = load_checkpoint(configargs, device, net_ensemble_coarse, net_ensemble_fine)
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
    if configargs.run_name:
        wandb_cfg["name"] = configargs.run_name
        
    wandb.init(**wandb_cfg, dir=cfg.experiment.logdir, config=cfg)
    wandb.config.USE_CACHED_DATASET = USE_CACHED_DATASET
    logdir = os.path.join(cfg.experiment.logdir, wandb.run.name)
    if not os.path.isdir(logdir):
        os.mkdir(os.path.join(cfg.experiment.logdir, wandb.run.name))
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False

    ###################################################################################################################
    ### ENSEMBLE TRAINING #############################################################################################

    for stage in range(start_stage, cfg.experiment.n_stages):
        if stage == start_stage:  # check checkpoint loaded only at first iteration
            weak_model_coarse, weak_model_fine = train_weak(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, start_weak_epoch, net_ensemble_coarse,
                                                            net_ensemble_fine, stage, resumed_weak_model_coarse, resumed_weak_model_fine)
        else:
            weak_model_coarse, weak_model_fine = train_weak(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, 0, net_ensemble_coarse,
                                                            net_ensemble_fine, stage)
        net_ensemble_coarse.add(weak_model_coarse)
        if weak_model_fine is not None:
            net_ensemble_fine.add(weak_model_fine)

        if stage > 0:
            fully_corrective_steps(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, net_ensemble_coarse, net_ensemble_fine, stage)

        validate(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, stage + 1, 0, net_ensemble_coarse, net_ensemble_fine, is_weak=False, datafile_idx=1)

        if MAX_MINUTES:
            if time.time() - start > MAX_MINUTES * 60:
                print("Time is up! Closing training...")
                break
    ###################################################################################################################
    print("- TRAIN DONE -")


def train_weak(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn,
               start_epoch, net_ensemble_coarse, net_ensemble_fine, stage, model_coarse=None, model_fine=None):
    if not model_coarse: # if no existent coarse, create both coarse and fine models
        model_coarse = get_model_coarse(cfg, stage)
        model_fine = get_model_fine(cfg, stage)
        model_coarse.to(device)
        if model_fine:
            model_fine.to(device)
    print(f"New learner. Width={int(cfg.models.fine.hidden_size * cfg.models.fine.hierarchical_factor**stage)}")

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    net_ensemble_coarse.train()  # return models to train mode after validation
    if net_ensemble_fine:
        net_ensemble_fine.train()

    # for i in trange(start_iter, cfg.experiment.weak_train_iters):
    pbar = tqdm(range(start_epoch, cfg.experiment.weak_train_iters), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{cfg.experiment.n_stages}: weak model training",
                unit=" epoch")
    for epoch in pbar:
        model_coarse.train()  # return models to train mode after validation
        if model_fine:
            model_coarse.train()

        # UPDATE LR #
        num_decay_steps = cfg.scheduler.lr_decay_weak # * 1000
        lr_new = cfg.optimizer.lr * (
                cfg.scheduler.lr_decay_factor_weak ** (epoch / num_decay_steps) if cfg.scheduler.lr_reset_weak else
                cfg.scheduler.lr_decay_factor_weak ** ((epoch + stage*cfg.experiment.weak_train_iters) / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        hwf, ray_directions, ray_origins, target_ray_values = get_random_rays(cfg, hwf, USE_CACHED_DATASET, device, data_dict)  # target_ray_values is RGBa if background is black, and RBG if it is white
        grad_direction_coarse, grad_direction_fine = target_ray_values[..., :3], target_ray_values[..., :3]
        penultimate_features_coarse, penultimate_features_fine, ray_batches, pts_and_zvals = None, None, None, None

        ### GETTING ENSEMBLE RESIDUAL #####################################################################################################
        disp_coarse_ensemble, disp_fine_ensemble = None, None
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
                encode_direction_fn,
                render_activation_fn=getattr(torch, cfg.experiment.render_activation_fn) if "render_activation_fn" in cfg.experiment else None
            )
            rgb_coarse_ensemble, disp_coarse_ensemble, _, rgb_fine_ensemble, disp_fine_ensemble, _ = out_ensemble  # predictions, one pixel for each ray (default: 1024 pixels)
            penultimate_features_coarse, penultimate_features_fine = penultimate_features
            if not cfg.experiment.propagate_context:
                penultimate_features_coarse, penultimate_features_fine = None, None

            grad_direction_coarse = -(rgb_coarse_ensemble[..., :3] - target_ray_values[..., :3])  # grad_direction = y / (1.0 + torch.exp(y * out))
            if rgb_fine_ensemble is not None:
                grad_direction_fine = -(rgb_fine_ensemble[..., :3] - target_ray_values[..., :3])
        ###################################################################################################################################
        
        ### FITTING RESIDUAL WITH WEAK LEARNER ############################################################################################
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
            render_activation_fn=getattr(torch, cfg.experiment.render_activation_fn) if "render_activation_fn" in cfg.experiment else None
        )
        rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _ = out  # predictions, one pixel for each ray (default: 1024 pixels)
        ###################################################################################################################################

        # LOSS + UPDATE ###################################################################################################################
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
        ###################################################################################################################################

        weak_train_log(cfg, device, epoch + 1, stage, loss, coarse_loss, fine_loss, psnr, lr_new, pbar, rgb_coarse, rgb_fine, grad_direction_coarse, grad_direction_fine, disp_coarse, disp_fine, disp_coarse_ensemble, disp_fine_ensemble)

        # Weak model Validation #
        if (epoch % cfg.experiment.validate_every == 0) or (epoch == cfg.experiment.weak_train_iters - 1):
            net_ensemble_coarse.add(model_coarse)
            net_ensemble_fine.add(model_fine)
            validate(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, stage, epoch, net_ensemble_coarse, net_ensemble_fine, is_weak=True, datafile_idx=1)
            net_ensemble_coarse.pop()
            net_ensemble_fine.pop()

        # SAVE CHECKPOINT #
        if epoch % cfg.experiment.save_every == 0 or epoch == cfg.experiment.weak_train_iters - 1:
            checkpoint_path = save_checkpoint(cfg, stage, epoch, loss, model_coarse, model_fine, net_ensemble_coarse, net_ensemble_fine, optimizer, psnr)
            if os.path.isfile(checkpoint_path):
                wandb.log({"checkpoint(KB)": os.path.getsize(checkpoint_path) / 1000})

        if MAX_MINUTES:
            if time.time() - start > MAX_MINUTES * 60:
                print("Time is up! Closing weak training...")
                break

    return model_coarse, model_fine


def fully_corrective_steps(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, net_ensemble_coarse, net_ensemble_fine, stage):
    # Initialize optimizer.
    trainable_parameters = list(net_ensemble_coarse.parameters())
    if net_ensemble_fine is not None:
        trainable_parameters += list(net_ensemble_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr_ensemble
    )

    net_ensemble_coarse.train()  # return models to train mode after validation
    if net_ensemble_fine:
        net_ensemble_fine.train()

    # print("CORRECTIVE STEPS...")
    pbar = tqdm(range(cfg.experiment.corrective_iters), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{cfg.experiment.n_stages}: fully corrective steps",
                unit=" epoch")
    for epoch in pbar:

        # UPDATE LR #
        tot_iters_per_weak = cfg.experiment.corrective_iters + cfg.experiment.weak_train_iters
        num_decay_steps = cfg.scheduler.lr_decay_corrective
        lr_new = cfg.optimizer.lr_ensemble * (
                cfg.scheduler.lr_decay_factor_corrective ** (epoch / num_decay_steps) if cfg.scheduler.lr_reset_corrective else
                # cfg.scheduler.lr_decay_factor_corrective ** ((
                #         epoch + stage * tot_iters_per_weak * ((epoch+1)/cfg.scheduler.lr_decay_corrective_peaked)
                #     ) / num_decay_steps
                # )

                cfg.scheduler.lr_decay_corrective_peaked*cfg.scheduler.lr_decay_factor_corrective ** ((
                        (epoch + (stage * tot_iters_per_weak)*cfg.scheduler.lr_decay_corrective_peaked)
                    ) / num_decay_steps
                )
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

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
            pts_and_zvals=None,
            render_activation_fn=getattr(torch, cfg.experiment.render_activation_fn) if "render_activation_fn" in cfg.experiment else None
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
        corrective_step_log(cfg, coarse_loss, epoch, fine_loss, loss, lr_new, pbar, psnr, stage, net_ensemble_coarse, net_ensemble_fine)

        # Ensemble model Validation #
        if epoch % (cfg.experiment.validate_every + 1) == cfg.experiment.validate_every:
            validate(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, stage, epoch, net_ensemble_coarse, net_ensemble_fine, is_weak=False, datafile_idx=1)

        if MAX_MINUTES:
            if time.time() - start > MAX_MINUTES * 60:
                print("Time is up! Closing fully corrective phase...")
                break


def weak_train_log(cfg, device, epoch, stage, loss, coarse_loss, fine_loss, psnr, lr, pbar, rgb_coarse, rgb_fine, target_ray_values_coarse, target_ray_values_fine, disp_coarse, disp_fine, disp_coarse_ensemble, disp_fine_ensemble):
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
                'loss_C': coarse_loss.item(),
                'loss_F': fine_loss.item(),
                # 'n_params': _n_params,
            })
        else:
            pbar.set_postfix({
                'lr': lr,
                'psnr': psnr,
                'loss': loss.item(),
                'loss_C.': coarse_loss.item(),
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

    if epoch % 500 == 0:  # save grid of predictions
        tensors = [rgb_coarse, target_ray_values_coarse]
        if fine_loss is not None:
            tensors += [rgb_fine, target_ray_values_fine]
        tensors += [disp_coarse]
        if disp_coarse_ensemble is not None:
            tensors += [disp_coarse_ensemble]
        if fine_loss is not None:
            tensors += [disp_fine]
        if disp_fine_ensemble is not None:
                tensors += [disp_fine_ensemble]
        imgs = []
        w = ceil(sqrt(cfg.nerf.train.num_random_rays))

        for t in tensors:
            channels = 3 if len(t.shape) > 1 else 1
            trail = torch.zeros((w ** 2 - t.shape[0], channels)).to(device)
            if channels == 1:
                t = t.unsqueeze(dim=-1)
            t = torch.cat((t[..., : None if channels == 1 else channels], trail))
            imgs.append(cv2.cvtColor(t.view(w, w, channels).detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
        img_grid = get_img_grid(2, len(tensors) // 2, len(tensors), imgs, margin=1)

        save_path = os.path.join(cfg.experiment.logdir, wandb.run.name, "rays")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(f'{save_path}/{epoch + (stage * cfg.experiment.weak_train_iters)}_weak.png', img_grid)


def corrective_step_log(cfg, coarse_loss, epoch, fine_loss, loss, lr, pbar, psnr, stage, net_ensemble_coarse, net_ensemble_fine):
    if epoch % cfg.experiment.print_every == 0 or epoch == cfg.experiment.weak_train_iters - 1:
        if fine_loss is not None:
            wandb.log({
                "Ensemble/corrective/loss_fine": fine_loss.item(),
                "Ensemble/corrective/boost_rate_fine": net_ensemble_fine.boost_rate.item(),
                "Ensemble_fine_n_params": sum(p.numel() for p in net_ensemble_fine.parameters())
            }, commit=False)

            pbar.set_postfix({
                'lr': lr,
                'psnr': psnr,
                'loss': loss.item(),
                'loss_C': coarse_loss.item(),
                'loss_F': fine_loss.item(),
                'boost_C': net_ensemble_coarse.boost_rate.item(),
                'boost_F': net_ensemble_fine.boost_rate.item(),
                # 'n_params': _n_params,
            })
        else:
            pbar.set_postfix({
                'lr': lr,
                'psnr': psnr,
                'loss': loss.item(),
                'loss_C': coarse_loss.item(),
                'boost_C': net_ensemble_coarse.boost_rate.item(),
                # 'n_params': _n_params,
            })

        wandb.log({
            "epoch": epoch,
            "stage": stage,
            "Ensemble/corrective/lr": lr,
            "Ensemble/corrective/loss": loss.item(),
            "Ensemble/corrective/loss_coarse": coarse_loss.item(),
            "Ensemble/corrective/psnr": psnr,
            "Ensemble/corrective/boost_rate_coarse": net_ensemble_coarse.boost_rate.item(),
            "Ensemble_coarse_n_params": sum(p.numel() for p in net_ensemble_coarse.parameters())
        }, commit=True)


def validate(cfg, hwf, USE_CACHED_DATASET, device, data_dict, encode_direction_fn, encode_position_fn, stage, epoch, model_coarse, model_fine, is_weak, penultimate_features_coarse=None, penultimate_features_fine=None, datafile_idx=None):
    # tqdm.write("[VAL] =======> Iter: " + str(i))
    model_coarse.eval()
    if model_fine:
        model_coarse.eval()

    start = time.time()
    with torch.no_grad():
        if USE_CACHED_DATASET:
            if datafile_idx: # and cfg.experiment.n_stages!=None # I want to see the same image when validating during weak training
                datafile = data_dict["validation_paths"][datafile_idx]
            else:
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
            if datafile_idx: # I want to see the same image when validating during weak training
                img_idx = datafile_idx
            else:
                img_idx = np.random.choice(data_dict["i_val"])
            img_target = data_dict["images"][img_idx].to(device)
            target_ray_values = img_target
            pose_target = data_dict["poses"][img_idx, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(*hwf, pose_target)

        out, _, _, _ = run_one_iter_of_nerf(
            hwf,
            model_coarse,
            model_fine,
            ray_origins,
            ray_directions,
            cfg,
            "validation",
            encode_position_fn,
            encode_direction_fn,
            penultimate_features_coarse,
            penultimate_features_fine,
            render_activation_fn=getattr(torch, cfg.experiment.render_activation_fn) if "render_activation_fn" in cfg.experiment else None
        )
        rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine = out

        coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
        loss, fine_loss = None, None
        if rgb_fine is not None:
            fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
            loss = coarse_loss + fine_loss
        else:
            loss = coarse_loss
        psnr = mse2psnr(loss.item()) # TODO: check this

        if (not is_weak) or cfg.experiment.n_stages<=1: # only validation during weak learning uses datafile_idx, I don't want to log that, too noisy (apart from single learner)
            if fine_loss is not None:
                wandb.log({
                    "Ensemble/validation/loss_fine": fine_loss,
                }, commit=False)

            wandb.log({
                "stage": stage,
                "epoch": epoch,
                "Ensemble/validation/loss": loss.item(),
                "Ensemble/validation/loss_coarse": coarse_loss.item(),
                "Ensemble/validation/psnr": psnr,
            }, commit=True)

        ### SAVE VALIDATION IMAGES ########################################################################################################
        mode = "ensVal"
        if is_weak:
            mode = "weakVal"

        save_path = os.path.join(cfg.experiment.logdir, wandb.run.name)
        if mode == "weakVal": # only validation during weak learning uses datafile_idx, I want to save that into a separate folder
            
            if not os.path.exists(os.path.join(save_path, "rgb_coarse")):
                os.makedirs(os.path.join(save_path, "rgb_coarse"))
            if not os.path.exists(os.path.join(save_path, "disp_coarse")):
                os.makedirs(os.path.join(save_path, "disp_coarse"))
            cv2.imwrite(f'{save_path}/rgb_coarse/s{stage}_e{epoch}_{mode}_rgb_coarse.png', cv2.cvtColor(rgb_coarse[..., :3].detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
            cv2.imwrite(f'{save_path}/disp_coarse/s{stage}_e{epoch}_{mode}_disp_coarse.png', disp_coarse.detach().cpu().numpy() * 255)

            if rgb_fine is not None:
                if not os.path.exists(os.path.join(save_path, "rgb_fine")):
                    os.makedirs(os.path.join(save_path, "rgb_fine"))
                if not os.path.exists(os.path.join(save_path, "disp_fine")):
                    os.makedirs(os.path.join(save_path, "disp_fine"))
                cv2.imwrite(f'{save_path}/rgb_fine/s{stage}_e{epoch}_{mode}_rgb_fine.png', cv2.cvtColor(rgb_fine[..., :3].detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
                cv2.imwrite(f'{save_path}/disp_fine/s{stage}_e{epoch}_{mode}_disp_fine.png', disp_fine.detach().cpu().numpy() * 255)

            save_path = os.path.join(save_path, f"val_stage{stage}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        cv2.imwrite(f'{save_path}/s{stage}_e{epoch}_{mode}_rgb_coarse.png', cv2.cvtColor(rgb_coarse[..., :3].detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f'{save_path}/s{stage}_e{epoch}_{mode}_disp_coarse.png', disp_coarse.detach().cpu().numpy() * 255)
        # cv2.imwrite(f'{save_path}/s{stage}_e{epoch}_{mode}_acc_coarse.png', acc_coarse.detach().cpu().numpy() * 255)
        if rgb_fine is not None:
            cv2.imwrite(f'{save_path}/s{stage}_e{epoch}_{mode}_rgb_fine.png', cv2.cvtColor(rgb_fine[..., :3].detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
            cv2.imwrite(f'{save_path}/s{stage}_e{epoch}_{mode}_disp_fine.png', disp_fine.detach().cpu().numpy() * 255)
            # cv2.imwrite(f'{save_path}/s{stage}_e{epoch}_{mode}_acc_fine.png', acc_fine.detach().cpu().numpy() * 255)
            cv2.imwrite(f'{save_path}/s{stage}_e{epoch}_{mode}_grad.png', cv2.cvtColor((target_ray_values[..., :3].detach() - rgb_fine[..., :3].detach()).cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite(f'{save_path}/s{stage}_e{epoch}_{mode}_grad.png', cv2.cvtColor((target_ray_values[..., :3].detach() - rgb_coarse[..., :3].detach()).cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
        
        cv2.imwrite(f'{save_path}/s{stage}_e{epoch}_{mode}_target_rgb.png', cv2.cvtColor(target_ray_values[..., :3].detach().cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
        ###################################################################################################################################
        
        tqdm.write(
            f"Validation loss: {loss.item():.5f}"
            + f" Validation PSNR: {psnr:.5f}"
            + f" Time: {time.time() - start:.5f}"
        )


if __name__ == "__main__":
    MAX_MINUTES=None
    start = time.time()
    main()
    print("seconds: ", time.time() - start)
