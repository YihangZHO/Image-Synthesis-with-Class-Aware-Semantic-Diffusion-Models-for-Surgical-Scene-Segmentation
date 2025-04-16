import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
import diffusers
from diffusers import (
    DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMPipeline,
    DPMSolverMultistepScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version, is_accelerate_version, is_tensorboard_available,
    is_wandb_available
)
from diffusers.utils.import_utils import is_xformers_available
from image_dataset import SegmentationDataset
import random
from PIL import Image
import numpy
import copy

check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def tensor_to_pil(tensor):
    tensor = (tensor + 1) / 2 * 255
    tensor = tensor.numpy().astype(numpy.uint8)
    if tensor.shape[0] == 1:
        tensor = numpy.repeat(tensor, 3, axis=0)
    tensor = numpy.transpose(tensor, (1, 2, 0))
    return Image.fromarray(tensor)

def make_grid(images, rows, cols):
    if isinstance(images, numpy.ndarray):
        w, h = Image.fromarray(images[0]).size
    else:
        w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        if isinstance(images, numpy.ndarray):
            image = Image.fromarray(image)
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def load_image(image_path):
    return Image.open(image_path)

def main():
    lr_scheduler = 'cosine'

    adam_beta1 = 0.95
    adam_beta2 = 0.999
    adam_weight_decay = 1e-6
    adam_epsilon = 1e-08

    ema_inv_gamma = 1.0
    ema_power = 3 / 4
    ema_max_decay = 0.9999

    push_to_hub = False
    hub_token = None
    hub_model_id = None
    logger_type = 'wandb'

    logging_dir = 'logs'
    local_rank = -1

    prediction_type = 'epsilon'
    ddpm_num_steps = 1000
    ddpm_num_inference_steps = 1000
    ddpm_beta_schedule = 'linear'

    enable_xformers_memory_efficient_attention = False

    output_dir = "ddpm-ema"
    gradient_accumulation_steps = 1
    lr_warmup_steps = 500

    checkpointing_steps = 500
    checkpoints_total_limit = None
    resume_from_checkpoint = "latest"
    # resume_from_checkpoint = None

    eval_batch_size = 2
    dataloader_num_workers = 0

    save_images_epochs = 1
    save_model_epochs = 1

    resolution = 256
    train_batch_size = 10
    num_epochs_first_stage = 30
    num_epochs = 500
    use_ema = True
    mixed_precision = "fp16"
    learning_rate = 2e-5

    num_classes = 13
    if_shape = True
    two_stage_training = True # this will set if_perceptual to True after training certain steps
    if_perceptual = False

    logging_dir = os.path.join(output_dir, logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=logging_dir
    )

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if logger_type == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Install tensorboard to use it for logging during training.")
    elif logger_type == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb to use it for logging during training.")
        import wandb

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                model = models.pop()
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        if push_to_hub:
            repo_id = create_repo(
                repo_id=hub_model_id or Path(output_dir).name,
                exist_ok=True,
                token=hub_token
            ).repo_id

        model = UNet2DModel(
            image_size=resolution,
            in_channels=3,
            out_channels=3,
            model_channels=256,
            num_res_blocks=2,
            attention_resolutions=(8, 16, 32),
            channel_mult=(1, 1, 2, 2, 4, 4),
            dropout=0,
            num_classes=num_classes,
            use_checkpoint=True,
            use_fp16=False,
            num_heads=1,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False
        )

    if use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=ema_inv_gamma,
            power=ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        mixed_precision = accelerator.mixed_precision

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 may cause issues on some GPUs. Consider upgrading to >= 0.0.17."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available. Install it properly.")

    # Initialize the scheduler

    # noise_scheduler = DDPMScheduler(
    #     num_train_timesteps=ddpm_num_steps,
    #     beta_schedule=ddpm_beta_schedule,
    #     prediction_type=prediction_type,
    # )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='scaled_linear',
        prediction_type='epsilon',
        rescale_betas_zero_snr=True,
        variance_type='fixed_small',
        clip_sample=True,
        clip_sample_range=1.0,
        timestep_spacing='leading',
    )

    noise_scheduler_inference = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear',
        algorithm_type='sde-dpmsolver++',
        solver_order=2,
        prediction_type='epsilon',
        rescale_betas_zero_snr=True,
        use_karras_sigmas=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    root_dir_images_train = '/content/train_local/cholec_train_test/train/images'
    root_dir_masks_train = '/content/train_local/cholec_train_test/train/groundtruth_no_white_color'
    dataset = SegmentationDataset(
        root_dir_images=root_dir_images_train,
        root_dir_masks=root_dir_masks_train,
        image_size=resolution,
        horiztonal_flip=True,
        vertical_flip=True,
        rotate=True,
        onehot=True
    )

    logger.info(f"Dataset size: {len(dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, shuffle=True, num_workers=dataloader_num_workers
    )

    root_dir_images_val = '/content/train_local/cholec_train_test/test/images'
    root_dir_masks_val = '/content/train_local/cholec_train_test/test/groundtruth_no_white_color'
    val_dataset = SegmentationDataset(
        root_dir_images=root_dir_images_val,
        root_dir_masks=root_dir_masks_val,
        image_size=resolution,
        horiztonal_flip=False,
        vertical_flip=False,
        rotate=False,
        onehot=True
    )

    random_indices = random.sample(range(len(val_dataset)), eval_batch_size)
    print(random_indices)
    fixed_val_batch_image = torch.stack([val_dataset[i]['image'] for i in random_indices], dim=0).squeeze()
    fixed_val_batch_mask = torch.stack([val_dataset[i]['mask'] for i in random_indices], dim=0).squeeze()

    val_image_names = [name for name in os.listdir(root_dir_images_val) if os.path.isfile(os.path.join(root_dir_images_val, name))]
    val_image_names.sort()
    val_masks = []

    for idx in random_indices:
        mask_name = os.path.join(root_dir_masks_val, os.path.splitext(val_image_names[idx])[0] + '_mapped_mask.png')
        mask_image = load_image(mask_name)
        val_masks.append(mask_image)

    mask_grid = make_grid(val_masks, rows=1, cols=2)
    pil_images = [tensor_to_pil(img) for img in fixed_val_batch_image]
    img_grid = make_grid(pil_images, rows=1, cols=2)

    validation_original_dir = "/content/validation_original"
    if not os.path.exists(validation_original_dir):
        os.makedirs(validation_original_dir)
    img_grid.save(os.path.join(validation_original_dir, "image_grid.png"))
    mask_grid.save(os.path.join(validation_original_dir, "mask_grid.png"))

    # initialize the scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # accelerator 
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if use_ema:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

    perceptual_loss_started = False
    if (two_stage_training and first_epoch > num_epochs_first_stage) or if_perceptual:
        print("Loading perceptual model...")
        perceptual_model = copy.deepcopy(model)
        perceptual_model.use_checkpoint = False
        for param in perceptual_model.parameters():
            param.requires_grad = False
        perceptual_model.eval()
        perceptual_model.to('cpu')
        perceptual_loss_started = True

    def compute_self_perceptual_loss(perceptual_model, noise_scheduler, noisy_images, timesteps, clean_images, clean_masks, model_output):
        batch_size = noisy_images.shape[0]
        sp_loss = torch.tensor(0.0, device=noisy_images.device, requires_grad=True)
        perceptual_model.to(noisy_images.device)

        for i in range(batch_size):
            v_pred = model_output[i].unsqueeze(0)
            timestep = timesteps[i].item()

            pred_original_sample = noise_scheduler.step(
                model_output=v_pred,
                timestep=timestep,
                sample=noisy_images[i].unsqueeze(0)
            ).pred_original_sample

            t_prime = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,)).long().to(noisy_images.device)
            noisy_images_prime = noise_scheduler.add_noise(clean_images[i].unsqueeze(0), v_pred, t_prime)
            noisy_images_prime_pred = noise_scheduler.add_noise(pred_original_sample, v_pred, t_prime)

            with torch.no_grad():
                feature_real = perceptual_model(
                    noisy_images_prime,
                    t_prime,
                    clean_masks[i].unsqueeze(0),
                    return_feature='midblock'
                )
                feature_pred = perceptual_model(
                    noisy_images_prime_pred,
                    t_prime,
                    clean_masks[i].unsqueeze(0),
                    return_feature='midblock'
                )

            sp_loss = sp_loss + F.mse_loss(feature_pred, feature_real)

        perceptual_model.to('cpu')
        sp_loss /= batch_size
        return sp_loss

    # Train!
    for epoch in range(first_epoch, num_epochs):
        if not perceptual_loss_started and two_stage_training and epoch > num_epochs_first_stage:
            print("Loading perceptual model...")
            perceptual_model = copy.deepcopy(model)
            perceptual_model.use_checkpoint = False
            for param in perceptual_model.parameters():
                param.requires_grad = False
            perceptual_model.eval()
            perceptual_model.to('cpu')
            perceptual_loss_started = True
        
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch['image'].to(weight_dtype)
            clean_masks = batch['mask'].to(weight_dtype).squeeze()

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            bsz = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                model_output = model(noisy_images, timesteps, clean_masks).sample

                if prediction_type == "epsilon":
                    if if_shape:
                        loss = F.mse_loss(model_output.float(), noise.float(), reduction='none')
                        pixels_per_class_per_batch = clean_masks.sum(dim=[2, 3])
                        total_pixels_per_batch = resolution * resolution
                        class_ratios_per_batch = pixels_per_class_per_batch / total_pixels_per_batch
                        weights_per_batch = torch.where(
                            pixels_per_class_per_batch > 0, 
                            1.0 / class_ratios_per_batch, 
                            torch.tensor(0.0)
                        )

                        weights_per_batch[:, 0] *= 1.5
                        weights_per_batch[:, 1] *= 1.5
                        weights_per_batch[:, -1] *= 1.5

                        weights_per_batch_normalized = weights_per_batch / weights_per_batch.sum(dim=1, keepdim=True)
                        expanded_weights = weights_per_batch_normalized.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, resolution, resolution)
                        pixel_weights = (clean_masks * expanded_weights).sum(1, keepdim=True).to(weight_dtype)

                        loss = loss * pixel_weights
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(model_output.float(), noise.float())

                    if if_perceptual:
                        with accelerator.autocast():
                            sp_loss = compute_self_perceptual_loss(
                                perceptual_model, noise_scheduler,
                                noisy_images, timesteps, clean_images, clean_masks, model_output
                            )
                        loss = loss + sp_loss

                elif prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        if checkpoints_total_limit is not None:
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }

            if use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if epoch in [0, 1, 2, 3] or epoch % save_images_epochs == 0 or epoch == num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                    
                # pipeline = DDPMPipeline(
                #    unet=unet,
                #    scheduler=noise_scheduler,
                # )
                
                # generator = torch.Generator(device=pipeline.device)
                # # run pipeline in inference (sample random noise and denoise)
                # images = pipeline(
                #    generator=generator,
                #    batch_size=eval_batch_size,
                #    num_inference_steps=ddpm_num_inference_steps,
                #    output_type="numpy",
                #    mask = fixed_val_batch_mask.float().to(clean_images.device)
                # ).images

                # # denormalize the images and save to tensorboard
                # images_processed = (images * 255).round().astype("uint8")

                # # Make a grid out of the images
                # image_grid = make_grid(images_processed, rows=1, cols=2)

                # # Save the images
                # test_dir = os.path.join("ddpm_spade", "samples")
                # os.makedirs(test_dir, exist_ok=True)
                # image_grid.save(f"{test_dir}/{epoch:04d}.png")

                pipeline_DDIM = DDIMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler_inference,
                )
                generator = torch.Generator(device=pipeline_DDIM.device)

                images = pipeline_DDIM(
                    generator=generator,
                    batch_size=eval_batch_size,
                    num_inference_steps=50,
                    output_type="numpy",
                    mask=fixed_val_batch_mask.float().to(clean_images.device)
                ).images

                images = numpy.nan_to_num(images)
                images_processed = (images * 255).round().astype("uint8")
                image_grid = make_grid(images_processed, rows=1, cols=2)

                test_dir = os.path.join("ddpm_spade", "samples")
                os.makedirs(test_dir, exist_ok=True)
                image_grid.save(f"{test_dir}/{epoch:04d}_DPM.png")

                if use_ema:
                    ema_model.restore(unet.parameters())

                if logger_type == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")

                    print('\nBefore adding images to TensorBoard\n')
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                    print('\nAfter adding images to TensorBoard\n')

                elif logger_type == "wandb":
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            if epoch % save_model_epochs == 0 or epoch == num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )
                pipeline.save_pretrained(output_dir)

                if use_ema:
                    ema_model.restore(unet.parameters())

                if push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
