import glob
import itertools
import os
from tabnanny import check

import numpy as np
import torch
import torchvision
import wandb
from tqdm import tqdm
import yaml
import argparse


from nerf import models, load_blender_data, load_llff_data, get_ray_bundle, meshgrid_xy, CfgNode

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, 
                        help='config file path')
    # parser.add_argument("--expname", type=str, 
    #                     help='experiment name', required=False)
    # parser.add_argument("--basedir", type=str, default='cache/nerf_synthetic/lego', 
    #                     help='where to store ckpts and logs')
    # parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
    #                     help='input data directory')

    # training options
    parser.add_argument("--depth", type=int, 
                        help='layers in network')
    parser.add_argument("--width", type=int, 
                        help='channels per layer')
    parser.add_argument("--num_random_rays", type=int, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lr", type=float, 
                        help='learning rate')
    parser.add_argument("--lr_ensemble", type=float, 
                        help='learning rate of ensemble fully corrective steps')
    parser.add_argument("--lr_decay_weak", type=int, 
                        help='exponential learning rate decay for weak lr')
    parser.add_argument("--lr_decay_corrective", type=int, 
                        help='exponential learning rate decay for corrective lr')
    parser.add_argument("--lr_decay_factor_weak", type=float, 
                        help='strength of the decay for weak lr')
    parser.add_argument("--lr_decay_factor_corrective", type=float, 
                        help='strength of the decay for corrective lr')
    parser.add_argument("--chunksize", type=int, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    # parser.add_argument("--netchunk", type=int, default=1024*64, 
    #                     help='number of pts sent through network in parallel, decrease if running out of memory')
    # parser.add_argument("--no_batching", action='store_true', 
    #                     help='only take random rays from 1 image at a time')
    # parser.add_argument("--no_reload", action='store_true', 
    #                     help='do not reload weights from saved ckpt')
    # parser.add_argument("--ft_path", type=str, default=None, 
    #                     help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--samples_coarse", type=int, 
                        help='number of coarse samples per ray')
    parser.add_argument("--samples_fine", type=int,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=str_to_bool,
                        help='set to False for no jitter, True for jitter')
    parser.add_argument("--use_viewdirs", type=str_to_bool, 
                        help='use full 5D input instead of 3D')
    # parser.add_argument("--i_embed", type=int, 
    #                     help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--num_encoding_xyz", type=int, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--num_encoding_dir", type=int, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # parser.add_argument("--render_only", action='store_true', 
    #                     help='do not optimize, reload weights and render out render_poses path')
    # parser.add_argument("--render_test", action='store_true', 
    #                     help='render the test set instead of render_poses path')
    # parser.add_argument("--render_factor", type=int, 
    #                     help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    # parser.add_argument("--precrop_iters", type=int, default=0,
    #                     help='number of steps to train on central crops')
    # parser.add_argument("--precrop_frac", type=float,
    #                     default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    # parser.add_argument("--shape", type=str, default='greek', 
    #                     help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", type=str_to_bool, 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", type=str_to_bool, 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    # parser.add_argument("--factor", type=int, default=8, 
    #                     help='downsample factor for LLFF images')
    # parser.add_argument("--no_ndc", action='store_true', 
    #                     help='do not use normalized device coordinates (set for non-forward facing scenes)')
    # parser.add_argument("--lindisp", action='store_true', 
    #                     help='sampling linearly in disparity rather than depth')
    # parser.add_argument("--spherify", action='store_true', 
    #                     help='set for spherical 360 scenes')
    # parser.add_argument("--llffhold", type=int, default=8, 
    #                     help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    # parser.add_argument("--i_print",   type=int, default=100, 
    #                     help='frequency of console printout and metric loggin')
    # parser.add_argument("--i_img",     type=int, default=500, 
    #                     help='frequency of tensorboard image logging')
    # parser.add_argument("--i_weights", type=int, default=10000, 
    #                     help='frequency of weight ckpt saving')
    # parser.add_argument("--i_testset", type=int, default=50000, 
    #                     help='frequency of testset saving')
    # parser.add_argument("--i_video",   type=int, default=50000, 
    #                     help='frequency of render_poses video saving')

    ### Mine ###

    parser.add_argument("--max_mins", type=int, 
                        help='Maximum duration of the experiment, in minutes')
    parser.add_argument("--load-checkpoint", type=str, default="",
                        help="Path to load saved checkpoint from. Creates separate run",)
    parser.add_argument("--resume", type=str_to_bool,
                        help="Resume from last checkpoint. (On wandb too)")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name of the run (for wandb), leave empty for random name.")

    parser.add_argument("--weak_iters", type=int, 
                        help='number of iterations for the training of a weak learner')
    parser.add_argument("--corrective_iters", type=int, 
                        help='number of corrective steps of the ensemble')
    parser.add_argument("--n_stages", type=int, 
                        help='final number of weak learners')
    parser.add_argument("--boost_rate", type=float, 
                        help='starting boosting rate of grownet')
    parser.add_argument("--learn_boost_rate", type=str_to_bool,
                        help="Update boost rate with training. Default=False")
    parser.add_argument("--propagate_context", type=str_to_bool,
                        help="Propagation of the penultimate layer to the input of the next weak learner. Default=True")
    parser.add_argument("--render_activation_fn", type=str, 
                        help='torch activation function to use (sigmoid, tanh, ...)')

    parser.add_argument("--lr_reset_weak", type=str_to_bool,
                        help="Wether to restart the scheduler for each new weak learner")
    parser.add_argument("--lr_reset_corrective", type=str_to_bool,
                        help="Wether to restart the scheduler for each corrective phase")
    parser.add_argument("--lr_decay_corrective_peaked", type=float, 
                        help='multiplier used to create peaks in the scheduler (works only if lr_reset_corrective: False)')
    parser.add_argument("--no_fine", type=str_to_bool,
                        help="Wether to use the fine model")
    parser.add_argument("--hierarchical_factor", type=float, 
                        help='weak hidden size = hidden_size*factor**stage')

    args = parser.parse_args()

    # Read config file.
    cfg = None
    with open(args.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Update cfg with user arguments
    if args.chunksize is not None:
        cfg.nerf.train.chunksize= args.chunksize
        cfg.nerf.validation.chunksize= args.chunksize
    if args.dataset_type is not None:
        cfg.dataset.type= args.dataset_type
    if args.depth is not None:
        cfg.models.coarse.num_layers = args.depth
        cfg.models.fine.num_layers = args.depth
    if args.half_res is not None:
        cfg.dataset.half_res = args.half_res
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    if args.lr_decay_weak is not None:
        cfg.scheduler.lr_decay_weak = args.lr_decay_weak
    if args.lr_decay_corrective is not None:
        cfg.scheduler.lr_decay_corrective = args.lr_decay_corrective
    if args.lr_decay_factor_weak is not None:
        cfg.scheduler.lr_decay_factor_weak = args.lr_decay_factor_weak
    if args.lr_decay_factor_corrective is not None:
        cfg.scheduler.lr_decay_factor_corrective = args.lr_decay_factor_corrective
    if args.lr_ensemble is not None:
        cfg.optimizer.lr_ensemble = args.lr_ensemble
    if args.num_encoding_dir is not None:
        cfg.models.coarse.num_encoding_fn_dir = args.num_encoding_dir
        cfg.models.fine.num_encoding_fn_dir = args.num_encoding_dir
    if args.num_encoding_xyz is not None:
        cfg.models.coarse.num_encoding_fn_xyz = args.num_encoding_xyz
        cfg.models.fine.num_encoding_fn_xyz = args.num_encoding_xyz
    if args.num_random_rays is not None:
        cfg.nerf.train.num_random_rays = args.num_random_rays
    if args.perturb is not None:
        cfg.nerf.train.perturb = args.perturb
    if args.raw_noise_std is not None:
        cfg.nerf.train.radiance_field_noise_std = args.raw_noise_std
    if args.samples_coarse is not None:
        cfg.nerf.train.num_coarse = args.samples_coarse
    if args.samples_fine is not None:
        cfg.nerf.train.num_fine = args.samples_fine
    if args.testskip is not None:
        cfg.dataset.testskip = args.testskip
    if args.use_viewdirs is not None:
        cfg.models.coarse.use_viewdirs = args.use_viewdirs
        cfg.models.fine.use_viewdirs = args.use_viewdirs
        cfg.nerf.use_viewdirs = args.use_viewdirs
    if args.white_bkgd is not None:
        cfg.nerf.train.white_background = args.white_bkgd
        cfg.nerf.validation.white_background = args.white_bkgd
    if args.width is not None:
        cfg.models.coarse.hidden_size = args.width
        cfg.models.fine.hidden_size = args.width
    if args.weak_iters is not None:
        cfg.experiment.weak_train_iters = args.weak_iters
    if args.corrective_iters is not None:
        cfg.experiment.corrective_iters = args.corrective_iters
    if args.n_stages is not None:
        cfg.experiment.n_stages = args.n_stages
    if args.boost_rate is not None:
        cfg.experiment.boost_rate = args.boost_rate
    if args.learn_boost_rate is not None:
        cfg.experiment.learn_boost_rate = args.learn_boost_rate
    if args.render_activation_fn is not None:
        cfg.experiment.render_activation_fn = args.render_activation_fn
    if args.lr_decay_corrective_peaked is not None:
        cfg.scheduler.lr_decay_corrective_peaked = args.lr_decay_corrective_peaked
    if args.no_fine is not None:
        cfg.models.no_fine = args.no_fine  
        cfg.nerf.train.num_fine = 0
        cfg.nerf.validation.num_fine = 0
    if args.lr_reset_weak is not None:
        cfg.scheduler.lr_reset_weak = args.lr_reset_weak
    if args.lr_reset_corrective is not None:
        cfg.scheduler.lr_reset_corrective = args.lr_reset_corrective
    if args.hierarchical_factor is not None:
        cfg.models.coarse.hierarchical_factor = args.hierarchical_factor
        cfg.models.fine.hierarchical_factor = args.hierarchical_factor
    if args.propagate_context is not None:
        cfg.experiment.propagate_context = args.propagate_context
    

    return cfg, args


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
    weak_model_coarse = get_model_coarse(cfg, start_stage)
    weak_model_fine = get_model_fine(cfg, start_stage)
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
    if hasattr(cfg.experiment, "propagate_context") and not cfg.experiment.propagate_context:
        stage = 0
    if not hasattr(cfg.models.coarse, "hierarchical_factor"):
        cfg.models.coarse.hierarchical_factor = 1.0

    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=int(cfg.models.coarse.hidden_size * cfg.models.coarse.hierarchical_factor**stage),
        skip_connect_every=cfg.models.coarse.skip_connect_every,
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        append_penultimate=stage,
        prev_penultimate_size=int(cfg.models.coarse.hidden_size * cfg.models.coarse.hierarchical_factor**(stage-1))
    )
    return model_coarse


def get_model_fine(cfg, stage):
    if hasattr(cfg.experiment, "propagate_context") and not cfg.experiment.propagate_context:
        stage = 0
    if not hasattr(cfg.models.fine, "hierarchical_factor"):
        cfg.models.fine.hierarchical_factor = 1.0

    model_fine = None
    # if hasattr(cfg.models, "fine"):
    if (not hasattr(cfg.models, "no_fine")) or (not cfg.models.no_fine):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_layers=cfg.models.fine.num_layers,
            hidden_size=int(cfg.models.fine.hidden_size * cfg.models.fine.hierarchical_factor**stage),
            skip_connect_every=cfg.models.fine.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            append_penultimate=stage,
            prev_penultimate_size=int(cfg.models.fine.hidden_size * cfg.models.fine.hierarchical_factor**(stage-1))
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
        - list[str] | None train_paths: one path for each training image, ignored if dataset is NOT cached
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
