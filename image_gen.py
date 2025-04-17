import os, re, json, ast, numpy, torch
from datetime import timedelta
from PIL import Image
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import UNet2DModel, DDIMPipeline, DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMScheduler, DPMSolverMultistepScheduler,  EulerDiscreteScheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from mask_only_dataset import MaskOnlyDataset

check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# ---------- config ----------
output_dir           = "ddpm-ema"
resume_ckpt          = "latest"          # or provide specific path / None
resolution           = 256
eval_bs              = 1
device               = "cuda" if torch.cuda.is_available() else "cpu"
mask_root            = "/content/train_local/cholec_train_test/train/groundtruth_no_white_color"
maps_out             = "/content/synthesised_maps"
imgs_out             = "/content/synthesised_images"
# ----------------------------

os.makedirs(maps_out, exist_ok=True)
os.makedirs(imgs_out,  exist_ok=True)

# ---------- class colors ----------
with open('/content/cholecSegClasses.json', 'r') as f:
    data = json.load(f)
gray = {i['name']: round(0.299*r + 0.587*g + 0.114*b)
        for i in data['classes'] for r, g, b in [ast.literal_eval(i['color'])]}
rev_sorted = [k for k, _ in sorted(gray.items(), key=lambda kv: kv[1])]

def mask_id_word(mask: torch.Tensor) -> str:
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    name = [(rev_sorted[i].replace(' ', '_') + '_')
            if torch.any(mask[0, i] > 0) else '' for i in range(len(rev_sorted))]
    return ''.join(name)[:-1]

mapping = {128:3,161:1,226:8,201:5,172:4,77:12,76:7,44:11,221:9,156:2,189:10,126:6,127:0}
rgb_map = {0:[127]*3,1:[210,140,140],2:[255,114,114],3:[231,70,156],4:[186,183,75],5:[170,255,0],
           6:[255,85,0],7:[255,0,0],8:[255,255,0],9:[169,255,184],10:[255,160,165],
           11:[0,50,128],12:[111,74,0]}
def mask2rgb(arr: numpy.ndarray) -> numpy.ndarray:
    rgb = numpy.zeros((*arr.shape, 3), dtype=numpy.uint8)
    for k, v in rgb_map.items():
        rgb[arr == k] = v
    return rgb

# ---------- accelerator ----------
project_cfg = ProjectConfiguration(project_dir=output_dir,
                                   logging_dir=os.path.join(output_dir, "logs"))
acc = Accelerator(mixed_precision="fp16",
                  kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))],
                  project_config=project_cfg)
logger.info(acc.state, main_process_only=False)

# ---------- model ----------
unet = UNet2DModel(image_size=resolution, in_channels=3, out_channels=3,
                   model_channels=256, num_res_blocks=2,
                   attention_resolutions=(8,16,32),
                   channel_mult=(1,1,2,2,4,4), num_classes=13)
ema  = EMAModel(unet.parameters(), decay=0.9999, use_ema_warmup=True,
                inv_gamma=1., power=0.75, model_cls=UNet2DModel,
                model_config=unet.config)

# scheduler=DDPMScheduler(
#     num_train_timesteps=1000,
#     beta_schedule='scaled_linear',  #'scaled_linear' or 'squaredcos_cap_v2'
#     prediction_type='epsilon',
#     rescale_betas_zero_snr=True,  
#     variance_type='fixed_small',  #'fixed_large' or 'fixed_large_log'
#     clip_sample=True,
#     clip_sample_range=1.0,
#     timestep_spacing='leading', #'trailing'
# )

scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=1e-4,
                                        beta_end=2e-2, prediction_type="epsilon",
                                        algorithm_type="dpmsolver++",
                                        rescale_betas_zero_snr=True)

# scheduler = DDIMScheduler(
#     num_train_timesteps=1000,
#     beta_schedule='linear',  #'scaled_linear' or 'squaredcos_cap_v2'
#     prediction_type='epsilon',  
#     rescale_betas_zero_snr=True,    #'fixed_large' or 'fixed_large_log'
#     clip_sample=True,
#     clip_sample_range=1.0,
#     # timestep_spacing='leading'  #'trailing'
# )




unet.to(device)

# ---------- load diffusers checkpoint ----------
def load_diffusers_ckpt(ckpt_dir: str):
    unet_sd = UNet2DModel.from_pretrained(ckpt_dir, subfolder="unet").state_dict()
    unet.load_state_dict(unet_sd)
    ema_dir = os.path.join(ckpt_dir, "unet_ema")
    if os.path.isdir(ema_dir):
        ema_sd = EMAModel.from_pretrained(ema_dir, UNet2DModel).state_dict()
        ema.load_state_dict(ema_sd)
    logger.info(f"Loaded diffusers checkpoint from {ckpt_dir}")

ckpt_to_load = None
if resume_ckpt:
    if resume_ckpt != "latest":
        ckpt_to_load = resume_ckpt
    else:
        cands = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if cands:
            cands.sort(key=lambda n: int(re.search(r"checkpoint-(\d+)", n).group(1)) if '-' in n else 0)
            ckpt_to_load = os.path.join(output_dir, cands[-1])
if ckpt_to_load and os.path.isdir(ckpt_to_load):
    load_diffusers_ckpt(ckpt_to_load)
print(f"Loaded checkpoint from {ckpt_to_load}")
ema.copy_to(unet.parameters())
unet.eval()
print("Model loaded")


# pipeline = DDPMPipeline(
#     unet=unet,
#     scheduler=scheduler,
# )
pipe = DDIMPipeline(unet=unet, scheduler=scheduler).to(device)
dataset = MaskOnlyDataset(root_dir_masks=mask_root, image_size=256, crop=None, onehot=True)

# rare case indices
rare_idx = [i for i in range(len(dataset))
            if any(t in mask_id_word(dataset[i]['mask']) for t in ("Liver_Ligament","Connective","L-hook"))]

def cyclic_batches(idx_list, bs):
    i, n = 0, len(idx_list)
    while True:
        yield [idx_list[(i+j)%n] for j in range(bs)]
        i = (i+bs)%n

batch_gen = cyclic_batches(rare_idx or list(range(len(dataset))), eval_bs)
unique_vals = torch.tensor(sorted(mapping), dtype=torch.long, device=device)

step = 0
with torch.inference_mode():
    while True:
        ids = next(batch_gen)
        masks = [dataset[i]['mask'] for i in ids]
        masks_t = torch.stack(masks).to(device)
        preds = pipe(batch_size=eval_bs, num_inference_steps=50,
                     output_type="numpy", mask=masks_t.float()).images
        class_idx = torch.argmax(masks_t, 1)
        orig = unique_vals[class_idx]
        for o, n in mapping.items():
            orig = torch.where(orig==o, torch.tensor(n, device=device), orig)
        rgb_masks = [mask2rgb(m.cpu().numpy()) for m in orig]
        names = [mask_id_word(m) for m in masks]
        for j, (rgb, img, nm) in enumerate(zip(rgb_masks, preds, names)):
            Image.fromarray(rgb).save(os.path.join(maps_out,  f"{step}_{j}_{nm}_mask.png"))
            Image.fromarray((numpy.nan_to_num(img)*255).round().astype("uint8")
                            ).save(os.path.join(imgs_out, f"{step}_{j}_{nm}_image.png"))
        step += 1
