from collections import deque
import torch
import torch.nn as nn
from torchvision import transforms
from functools import partial
from src.data.normalizer import StateActionNormalizer, StateActionSkillNormalizer
from src.models.vision import ResnetEncoder, get_encoder
from src.models.unet import ConditionalUnet1D
from src.common.pytorch_util import replace_submodules
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torchvision.transforms.functional as F
from src.behavior.tasks import OneLeg

from ipdb import set_trace as bp
import numpy as np
from src.models.module_attr_mixin import ModuleAttrMixin
from typing import Union
from src.common.pytorch_util import dict_apply


class DoubleImageActor(torch.nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: StateActionNormalizer,
        config,
    ) -> None:
        super().__init__()
        self.action_dim = config.action_dim
        self.pred_horizon = config.pred_horizon
        self.action_horizon = config.action_horizon
        self.obs_horizon = config.obs_horizon
        self.inference_steps = config.inference_steps
        self.observation_type = config.observation_type
        self.robot_state_dim = config.robot_state_dim
        self.down_dims = config.down_dims
        self.device = device

        self.train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule=config.beta_schedule,
            # clip output to [-1,1] to improve stability
            clip_sample=config.clip_sample,
            # our network predicts noise (instead of denoised action)
            prediction_type=config.prediction_type,
        )

        self.inference_noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
        )

        # Convert the stats to tensors on the device
        self.normalizer = normalizer.to(device)

        self.encoder1 = get_encoder(encoder_name, freeze=freeze_encoder, device=device)
        self.encoder2 = (
            get_encoder(encoder_name, freeze=freeze_encoder, device=device) if not freeze_encoder else self.encoder1
        )

        self.encoding_dim = self.encoder1.encoding_dim + self.encoder2.encoding_dim
        self.obs_dim = self._obs_dim()

        self.model = self._create_unet().to(device)

        self.print_model_params()

    def _create_unet(self):
        return ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * self.obs_horizon,
            down_dims=self.down_dims,
        )

    def _obs_dim(self):
        return self.robot_state_dim + self.encoding_dim

    def print_model_params(self: torch.nn.Module):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:.2e}")

        for name, submodule in self.named_children():
            params = sum(p.numel() for p in submodule.parameters())
            print(f"{name}: {params:.2e} parameters")

    def _normalized_obs(self, obs: deque):
        # Convert robot_state from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        robot_state = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        # Get the batch size as the number of environments
        B = nrobot_state.shape[0]

        # Get size of the image
        img_size = obs[0]["color_image1"].shape[-3:]

        # Images come in as obs_horizon x (n_envs, 224, 224, 3) concatenate to (n_envs * obs_horizon, 224, 224, 3)
        img1 = torch.cat([o["color_image1"].unsqueeze(1) for o in obs], dim=1).reshape(B * self.obs_horizon, *img_size)
        img2 = torch.cat([o["color_image2"].unsqueeze(1) for o in obs], dim=1).reshape(B * self.obs_horizon, *img_size)

        # Encode the images and reshape back to (B, obs_horizon, -1)
        features1 = self.encoder1(img1).reshape(B, self.obs_horizon, -1)
        features2 = self.encoder2(img2).reshape(B, self.obs_horizon, -1)

        if "feature1" in self.normalizer.stats:
            features1 = self.normalizer(features1, "feature1", forward=True)
            features2 = self.normalizer(features2, "feature2", forward=True)

        # Concatenate the features
        nobs = torch.cat([nrobot_state, features1, features2], dim=-1)

        return nobs

    def _action(self, obs_cond: torch.Tensor) -> torch.Tensor:
        """
        Function to perform the mechanic of the reverse diffusion process to produce a normalized action

        :param obs_cond: (B, obs_horizon * obs_dim)
        """
        # Get the batch size as the number of environments
        B = obs_cond.shape[0]

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=self.device,
        )
        naction = noisy_action

        # init scheduler
        self.inference_noise_scheduler.set_timesteps(self.inference_steps)

        for k in self.inference_noise_scheduler.timesteps:
            # predict noise
            # Print dtypes of all tensors to the model
            noise_pred = self.model(sample=naction, timestep=k, global_cond=obs_cond)

            # inverse diffusion step (remove noise)
            naction = self.inference_noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        return naction

    # === Inference ===
    @torch.no_grad()
    def action(self, obs: deque):
        nobs = self._normalized_obs(obs)
        obs_cond = nobs.flatten(start_dim=1)
        naction = self._action(obs_cond)

        # unnormalize action
        # (B, pred_horizon, action_dim)
        action_pred = self.normalizer(naction, "action", forward=False)

        return action_pred

    # === Training ===
    def compute_loss(self, batch):
        obs_cond = self._concatenate_obs(batch)
        B = obs_cond.shape[0]

        # observation as FiLM conditioning
        # (B, obs_horizon * obs_dim)
        obs_cond = obs_cond.flatten(start_dim=1)

        # Action already normalized in the dataset
        # naction = normalize_data(batch["action"], stats=self.stats["action"])
        naction = batch["action"]
        # sample noise to add to actions
        noise, timesteps, noisy_action = self._noisy_action(B, naction)

        # forward pass
        noise_pred = self.model(noisy_action, timesteps, global_cond=obs_cond.float())
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss

    def _concatenate_obs(self, batch):
        nrobot_state = batch["robot_state"]
        B = nrobot_state.shape[0]

        if self.observation_type == "image":
            # State already normalized in the dataset
            # Convert images from obs_horizon x (n_envs, 224, 224, 3) -> (n_envs, obs_horizon, 224, 224, 3)
            # so that it's compatible with the encoder
            image1 = batch["color_image1"].reshape(B * self.obs_horizon, 224, 224, 3)
            image2 = batch["color_image2"].reshape(B * self.obs_horizon, 224, 224, 3)

            # Encode images and reshape back to (B, obs_horizon, -1)
            image1 = self.encoder1(image1).reshape(B, self.obs_horizon, -1)
            image2 = self.encoder2(image2).reshape(B, self.obs_horizon, -1)

            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, image1, image2], dim=-1)

        elif self.observation_type == "feature":
            # All observations already normalized in the dataset
            feature1 = batch["feature1"]
            feature2 = batch["feature2"]
            nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)
        return nobs

    def _noisy_action(self, naction):
        noise = torch.randn(naction.shape, device=self.device)
        B = naction.shape[0]

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action = self.train_noise_scheduler.add_noise(naction, noise, timesteps)
        return noise, timesteps, noisy_action


class SkillImageActor(DoubleImageActor):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: StateActionSkillNormalizer,
        config,
    ) -> None:
        # TODO: This needs to be solved in a better way
        self.task = OneLeg()
        self.skill_emedding_dim = config.skill_embedding_dim

        super().__init__(device, encoder_name, freeze_encoder, normalizer, config)
        self.skill_embedding = nn.Embedding(self.task.n_skills, self.skill_emedding_dim).to(device)

    def _concatenate_obs(self, batch):
        nobs = super()._concatenate_obs(batch)

        # Get the skill index and embed it
        skill_embedding = self.skill_embedding(batch["skill_idx"])

        # Concatenate the skill embedding to the observation
        nobs = torch.cat([nobs, skill_embedding], dim=-1)

        return nobs

    def _obs_dim(self):
        # Add the skill embedding to the original observation dimension
        return super()._obs_dim() + self.skill_emedding_dim

    def _normalized_obs(self, obs: deque):
        nobs = super()._normalized_obs(obs)

        # Concatenate the skill embedding and make into right shpe
        # o["skill_idx"] is (n_envs, 1) -> (n_envs, obs_horizon)
        skill_idx = torch.cat([o["skill_idx"] for o in obs], dim=1)

        # Get the skill index and embed it, (n_envs, obs_horizon, skill_embedding_dim)
        skill_embedding = self.skill_embedding(skill_idx)

        # Concatenate the skill embedding to the observation
        nobs = torch.cat([nobs, skill_embedding], dim=-1)

        return nobs

    # === Inference ===
    @torch.no_grad()
    def action(self, obs: deque):
        nobs = self._normalized_obs(obs)
        obs_cond = nobs.flatten(start_dim=1)
        naction = self._action(obs_cond)
        # Unnormalize action and the last "done" action
        # The done will be a value between 0 and 1 that we interpret as the probability of the skill being done
        # (B, pred_horizon, action_dim)
        naction = self.normalizer(naction, "action", forward=False)

        # # Pass the "done" action through the sigmoid function
        # # We interpret it as the probability of the skill being done
        # naction[:, :, -1] = torch.sigmoid(naction[:, :, -1])

        return naction

    # === Training ===
    def compute_loss(self, batch):
        # This thing contains the robot_state, image features, and skill embedding
        nobs = self._concatenate_obs(batch)

        # observation as FiLM conditioning
        # (B, obs_horizon * obs_dim)
        obs_cond = nobs.flatten(start_dim=1)

        # Action already normalized in the dataset
        # naction = normalize_data(batch["action"], stats=self.stats["action"])
        naction = batch["action"]
        # sample noise to add to actions
        noise, timesteps, noisy_action = self._noisy_action(naction)

        # forward pass
        noise_pred = self.model(noisy_action, timesteps, global_cond=obs_cond.float())

        # In the loss, we should account for the fact that the last action is a "done" action
        # As a first attempt, just optimize as the other actions as a value between -1 and 1
        # and interpret positive values as "done"
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss
