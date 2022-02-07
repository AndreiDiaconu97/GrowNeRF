import glob
import itertools
import os
from tabnanny import check

import numpy as np
import torch
import torchvision
import wandb
from tqdm import tqdm

from nerf import models, load_blender_data, load_llff_data, get_ray_bundle, meshgrid_xy


def save_checkpoint(cfg, stage, epoch, loss, model_coarse, model_fine, ensemble_coarse, ensemble_fine, optimizer, psnr):
    checkpoint_dict = {
        "cfg": cfg,
        "run_id": wandb.run.id,
        "ensemble_stage": stage,
        "weak_epoch": epoch,
        "model_coarse_state_dict": model_coarse.state_dict(),
        "model_fine_state_dict": None if not model_fine else model_fine.state_dict(),
        "ensemble_coarse_state_dict": ensemble_coarse.state_dict(),
        "ensemble_fine_state_dict": None if not ensemble_fine else ensemble_fine.state_dict(),
        "weak_optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "psnr": psnr,
    }

    checkpoint_path = os.path.join(cfg.experiment.logdir, wandb.run.name, "checkpoint_stage" + str(stage).zfill(2) + "_epoch" + str(epoch).zfill(5) + ".ckpt")
    torch.save(
        checkpoint_dict,
        checkpoint_path
    )
    tqdm.write("================== Saved Checkpoint =================")

    return checkpoint_path


def load_checkpoint(configargs, device, net_ensemble_coarse, net_ensemble_fine):
    checkpoint = torch.load(configargs.load_checkpoint)
    cfg = checkpoint["cfg"]
    start_stage = checkpoint["ensemble_stage"]
    start_weak_epoch = checkpoint["weak_epoch"]
    run_id = checkpoint["run_id"]

    # Load weak models
    weak_model_coarse, weak_model_fine = get_model_coarse(cfg, start_stage), get_model_fine(cfg, start_stage)
    weak_model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    weak_model_coarse.to(device)
    if checkpoint["model_fine_state_dict"]:
        weak_model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        weak_model_fine.to(device)

    # Initialize weak optimizer
    trainable_parameters = list(weak_model_coarse.parameters())
    if weak_model_fine is not None:
        trainable_parameters += list(weak_model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )
    optimizer.load_state_dict(checkpoint["weak_optimizer_state_dict"])

    # Load ensembles
    net_ensemble_coarse.load_state_dict(checkpoint["ensemble_coarse_state_dict"], cfg, get_model_coarse)
    if checkpoint["ensemble_fine_state_dict"]:
        net_ensemble_fine.load_state_dict(checkpoint["ensemble_fine_state_dict"], cfg, get_model_fine)
    print(f'Loaded checkpoint: logdir={cfg.experiment.logdir} run_id={run_id} stage={start_stage}, weak_epoch={start_weak_epoch} loss={checkpoint["loss"]}, psnr={checkpoint["psnr"]}')
    return cfg, run_id, start_stage, start_weak_epoch, weak_model_coarse, weak_model_fine


def get_model_coarse(cfg, stage):
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
    return model_coarse


def get_model_fine(cfg, stage):
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
    return model_fine


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
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


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
            size=cfg.nerf.train.num_random_rays,
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
            coords.shape[0], size=cfg.nerf.train.num_random_rays, replace=False
        )
        select_inds = coords[select_inds]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
        target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
        target_ray_values = target_s

    return hwf, ray_directions, ray_origins, target_ray_values
