"""Define data collection class that rollout the environment, get action from the interface (e.g., teleoperation, automatic scripts), and save data."""
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import Union, List, Dict

import gym
from tqdm import tqdm, trange
from ipdb import set_trace as bp

from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.data.collect_enum import CollectEnum
from furniture_bench.sim_config import sim_config
from src.data_processing.utils import resize, resize_crop
from furniture_bench.envs.initialization_mode import Randomness
from src.visualization.render_mp4 import pickle_data, unpickle_data


import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import scipy.spatial.transform as st

import argparse
import random
import torch

from furniture_bench.device import make_device
from furniture_bench.config import config

from src.common.files import trajectory_save_dir


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


class DataCollectorAugmentor:
    """Demonstration collection class.
    `pkl` files have resized images while `mp4` / `png` files save raw camera inputs.
    """

    def __init__(
        self,
        is_sim: bool,
        data_path: str,
        device_interface: DeviceInterface,
        furniture: str,
        headless: bool,
        draw_marker: bool,
        manual_label: bool,
        scripted: bool,
        randomness: Randomness.LOW,
        compute_device_id: int,
        graphics_device_id: int,
        pkl_only: bool = False,
        save_failure: bool = False,
        num_demos: int = 100,
        resize_sim_img: bool = True,
        verbose: bool = True,
        show_pbar: bool = False,
        ctrl_mode: str = "osc",
        ee_laser: bool = True,
        right_multiply_rot: bool = True,
        compress_pickles: bool = False,
        resume_trajectory_paths: Union[List[str], None] = None,
    ):
        """
        Args:
            is_sim (bool): Whether to use simulator or real world environment.
            data_path (str): Path to save data.
            device_interface (DeviceInterface): Keyboard and/or Oculus interface.
            furniture (str): Name of the furniture.
            headless (bool): Whether to use headless mode.
            draw_marker (bool): Whether to draw AprilTag marker.
            manual_label (bool): Whether to manually label the reward.
            scripted (bool): Whether to use scripted function for getting action.
            randomness (str): Initialization randomness level.
            gpu_id (int): GPU ID.
            pkl_only (bool): Whether to save only `pkl` files (i.e., exclude *.mp4 and *.png).
            save_failure (bool): Whether to save failure trajectories.
            num_demos (int): The maximum number of demonstrations to collect in this run. Internal loop will be terminated when this number is reached.
            ctrl_mode (str): 'osc' (joint torque, with operation space control) or 'diffik' (joint impedance, with differential inverse kinematics control)
            ee_laser (bool): If True, show a line coming from the end-effector in the viewer
            right_multiply_rot (bool): If True, convert rotation actions (delta rot) assuming they're applied as RIGHT multiplys (local rotations)
        """
        if is_sim:
            self.env = gym.make(
                "FurnitureSimFull-v0",
                furniture=furniture,
                max_env_steps=sim_config["scripted_timeout"][furniture]
                if scripted
                else 3000,
                headless=headless,
                num_envs=1,  # Only support 1 for now.
                manual_done=False if scripted else True,
                resize_img=resize_sim_img,
                np_step_out=False,  # Always output Tensor in this setting. Will change to numpy in this code.
                channel_first=False,
                randomness=randomness,
                compute_device_id=compute_device_id,
                graphics_device_id=graphics_device_id,
                ctrl_mode=ctrl_mode,
                ee_laser=ee_laser,
            )
        else:
            if randomness == "med":
                randomness = Randomness.MEDIUM_COLLECT
            elif randomness == "high":
                randomness = Randomness.HIGH_COLLECT

            self.env = gym.make(
                "FurnitureBench-v0",
                furniture=furniture,
                resize_img=False,
                manual_done=True,
                with_display=not headless,
                draw_marker=draw_marker,
                randomness=randomness,
            )

        self.data_path = Path(data_path)
        self.device_interface = device_interface
        self.headless = headless
        self.manual_label = manual_label
        self.furniture = furniture
        self.num_demos = num_demos
        self.scripted = scripted

        self.traj_counter = 0
        self.num_success = 0
        self.num_fail = 0

        self.pkl_only = pkl_only
        self.save_failure = save_failure
        self.resize_sim_img = resize_sim_img
        self.compress_pickles = compress_pickles
        self.resume_trajectory_paths = resume_trajectory_paths

        self.iter_idx = 0

        self.verbose = verbose
        self.pbar = None if not show_pbar else tqdm(total=self.num_demos)

        # Parameters for controlling the time it takes for the robot to settle at the start of a trajectory
        self.start_delay = 0  # seconds
        self.robot_settled = False
        self.starttime = datetime.now()

        # our flags
        self.right_multiply_rot = right_multiply_rot

        self._reset_collector_buffer()

    def _squeeze_and_numpy(
        self, d: Dict[str, Union[torch.Tensor, np.ndarray, float, int, None]]
    ):
        """
        Recursively squeeze and convert tensors to numpy arrays
        Convert scalars to floats
        Leave NoneTypes alone
        """
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = self._squeeze_and_numpy(v)

            elif v is None:
                continue

            elif isinstance(v, (torch.Tensor, np.ndarray)):
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                d[k] = v.squeeze()

            elif k == "rewards":
                d[k] = float(v)
            elif k == "skills":
                d[k] = int(v)
            else:
                raise ValueError(f"Unsupported type: {type(v)}")

        return d

    def collect(self):
        self.verbose_print("[data collection] Start collecting the data!")

        from collections import namedtuple

        args = namedtuple(
            "Args",
            [
                "frequency",
                "command_latency",
                "deadzone",
                "max_pos_speed",
                "max_rot_speed",
            ],
        )

        args.frequency = 10
        args.command_latency = 0.01
        args.deadzone = 0.05
        if self.env.ctrl_mode == "diffik":
            args.max_pos_speed = 0.3
            args.max_rot_speed = 0.7
        else:
            args.max_pos_speed = 0.8
            # args.max_rot_speed = 2.5
            args.max_rot_speed = 4.0

        frequency = args.frequency
        dt = 1 / frequency
        command_latency = args.command_latency

        obs = self.reset()
        done = False

        t_start = time.monotonic()

        while self.num_success < self.num_demos:

            # calculate timing
            t_cycle_end = t_start + (self.iter_idx + 1) * dt
            t_sample = t_cycle_end - command_latency
            # t_command_target = t_cycle_end + dt
            precise_wait(t_sample)

            if not len(self.reverse_actions):
                collect_enum = CollectEnum.FAIL
            else:
                # get next command from the buffer
                rev_action = self.reverse_actions.pop()
                rev_ee_pose = self.reverse_ee_poses.pop()
                action = self.create_delta_action(rev_ee_pose, [rev_action[-1]])
                action_taken = True
                collect_enum = CollectEnum.DONE_FALSE

            skill_complete = int(collect_enum == CollectEnum.SKILL)
            if skill_complete == 1:
                self.skill_set.append(skill_complete)

            if collect_enum == CollectEnum.TERMINATE:
                self.verbose_print("Terminate the program.")
                break

            # An episode is done.
            if done or collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                self.store_transition(next_obs)

                if (
                    done and not self.env.furnitures[0].all_assembled()
                ) or collect_enum is CollectEnum.FAIL:
                    collect_enum = CollectEnum.FAIL
                    if self.save_failure:
                        self.verbose_print("Saving failure trajectory.")
                        obs = self.save_and_reset(collect_enum, {})
                    else:
                        self.verbose_print(
                            "Failed to assemble the furniture, reset without saving."
                        )
                        obs = self.reset()
                    self.num_fail += 1
                else:
                    if done:
                        collect_enum = CollectEnum.SUCCESS

                    obs = self.save_and_reset(collect_enum, {})
                    self.num_success += 1
                    self.update_pbar()

                self.traj_counter += 1
                self.verbose_print(
                    f"Success: {self.num_success}, Fail: {self.num_fail}"
                )

                done = False

                continue

            # Execute action.
            next_obs, rew, done, info = self.env.step(action)

            if rew == 1:
                self.last_reward_idx = len(self.transitions)

            # Error handling.
            if not info["obs_success"]:
                self.verbose_print(
                    "Getting observation failed, save trajectory."
                )
                # Pop the last reward and action so that obs has length plus 1 then those of actions and rewards.
                self.transitions["rewards"] = None
                self.transitions["actions"] = None
                self.transitions["skills"] = None

                obs = self.save_and_reset(CollectEnum.FAIL, info)
                continue

            # Logging a step.
            if action_taken:
                # Store a transition.
                if info["action_success"]:
                    self.store_transition(obs, action, rew, skill_complete)

                    # Intrinsic rotation
                    translation, quat_xyzw = self.env.get_ee_pose()
                    translation, quat_xyzw = (
                        translation.cpu().numpy().squeeze(),
                        quat_xyzw.cpu().numpy().squeeze(),
                    )

            obs = next_obs

            # target_pose = new_target_pose
            translation, quat_xyzw = self.env.get_ee_pose()
            translation, quat_xyzw = (
                translation.cpu().numpy().squeeze(),
                quat_xyzw.cpu().numpy().squeeze(),
            )

            # SM wait
            precise_wait(t_cycle_end)
            self.iter_idx += 1

            if (not self.robot_settled) and (
                (datetime.now() - self.starttime).seconds > self.start_delay
                ):
                    self.robot_settled = True
                    print("Robot settled")

            self.verbose_print(
                f"Collected {self.traj_counter} / {self.num_demos} successful trajectories!"
            )

    def set_target_pose(self):
        translation, quat_xyzw = self.env.get_ee_pose()
        translation, quat_xyzw = (
            translation.cpu().numpy().squeeze(),
            quat_xyzw.cpu().numpy().squeeze(),
        )
        gripper_width = self.env.gripper_width()
        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        target_pose_rv = np.array([*translation, *rotvec])
        gripper_open = gripper_width >= 0.06
        grasp_flag = torch.from_numpy(np.array([-1 if gripper_open else 1])).to(
            self.env.device
        )

        return target_pose_rv, gripper_width, gripper_open, grasp_flag

    def undo_actions(self):
        self.verbose_print("Undo the last 10 actions.")

        # Remove the last 10 transitions from the buffer but keep at least one
        self.transitions = self.transitions[:1] + self.transitions[1:-10]

        # Set the environment to the state before the last 10 actions.
        self.env.reset_env_to(env_idx=0, state=self.transitions[-1]["observations"])
        self.env.refresh()

    def store_transition(
        self, obs, action=None, rew=None, skill_complete=None, setup_phase=False
    ):
        """Store the observation, action, and reward."""
        if (not setup_phase) and (not self.robot_settled):
            # Don't store anything until the robot has settled
            # Without this, we get ~8 useless actions at the start of every trajectory
            return

        # We want to resize the images while tensors for maximum compatibility with the rest of the code
        n_ob = {}
        n_ob["color_image1"] = resize(obs["color_image1"])
        n_ob["color_image2"] = resize_crop(obs["color_image2"])
        n_ob["robot_state"] = obs["robot_state"]
        n_ob["parts_poses"] = obs["parts_poses"]

        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.squeeze().cpu().numpy()
            elif isinstance(action, np.ndarray):
                action = action.squeeze()
            else:
                raise ValueError(f"Unsupported action type: {type(action)}")

        if rew is not None:
            if isinstance(rew, torch.Tensor):
                rew = rew.item()
            elif isinstance(rew, np.ndarray):
                rew = rew.item()
            elif isinstance(rew, float):
                rew = rew
            elif isinstance(rew, int):
                rew = float(rew)

        transition = {
            "observations": n_ob,
            "actions": action,
            "rewards": rew,
            "skills": skill_complete,
        }

        # Treat the whole transition as a dictionary, and squeeze all the tensors and make scalars into floats
        transition = self._squeeze_and_numpy(transition)
        self.transitions.append(transition)

        # We'll update the steps counter whenever we store an observation
        if not setup_phase:
            print(
                f"{[self.step_counter]} assembled: {self.env.furniture.assembled_set} "
                f"num assembled: {len(self.env.furniture.assembled_set)} "
                f"Skill: {len(self.skill_set)}."
            )

    @property
    def step_counter(self):
        return len(self.transitions)

    def save_and_reset(self, collect_enum: CollectEnum, info):
        """Saves the collected data and reset the environment."""
        self.save(collect_enum, info)
        self.verbose_print(f"Saved {self.traj_counter} trajectories in this run.")
        return self.reset()

    def reset(self):
        obs = self.env.reset()

        print("State from reset:")
        for k, v in obs.items():
            print(k, type(v))

        self._reset_collector_buffer()

        for _ in range(50):
            self.env.step(torch.randn((8,)).float().to(self.env.device))

        if self.resume_trajectory_paths:
            obs = self.load_state()

        self.verbose_print("Start collecting the data!")

        self.starttime = datetime.now()
        self.robot_settled = False
        return obs

    def _reset_collector_buffer(self):
        # Now, observations, actions, rewards, and skall_complete flags are stored as transition "tuples"
        self.transitions = []

        self.last_reward_idx = -1
        self.skill_set = []

        self.reverse_actions = []
        self.reverse_ee_poses = []

    def load_state(self, offset_steps: int=20, aug_transition_idx: int=3, n_back_steps: int=10):
        """
        Load the state of the environment from a one_leg trajectory
        from the currently first pickle in the resume_trajectory_paths list
        """

        # Get the state dict at the end of a one_leg trajectory
        # trajectory_path = self.resume_trajectory_paths.pop(0)
        trajectory_path = random.sample(self.resume_trajectory_paths, 1)[0]

        print("Loading state from:")
        print(trajectory_path)

        state = unpickle_data(trajectory_path)

        print("State from pickle:")
        for k, v in state.items():
            print(k, type(v))

        skill_transition_indices = []
        original_episode_actions = []    
        original_episode_ee_poses = []    
        original_episode_horizon = len(state["observations"]) - 1

        def ee_pose_from_robot_state(robot_state_dict):
            ee_pos, ee_quat = robot_state_dict["ee_pos"], robot_state_dict["ee_quat"]
            ee_pose = np.concatenate([ee_pos, ee_quat])
            return ee_pose

        # Add all the data so far in the trajectory to the collect buffer, but stop when we reach the transition to hit
        for i in trange(original_episode_horizon, desc="Hydrating state"):
            obs={
                "color_image1": np.array(state["observations"][i]["color_image1"]),
                "color_image2": np.array(state["observations"][i]["color_image2"]),
                "robot_state": state["observations"][i]["robot_state"],
                "parts_poses": np.array(state["observations"][i]["parts_poses"]),
            }
            action=state["actions"][i] if i < len(state["actions"]) else None
            rew=state["rewards"][i] if i < len(state["rewards"]) else None
            skill_complete=state["skills"][i] if i < len(state["skills"]) else None
            setup_phase=True

            original_episode_actions.append(action)
            original_episode_ee_poses.append(ee_pose_from_robot_state(state["observations"][i]["robot_state"]))

            if skill_complete:
                skill_transition_indices.append(i)
                print(f'Step: {i}, Skill complete: {skill_complete}, Skills: {state["skills"][i]}, Rewards: {state["rewards"][i]}')
        
        # log the actions in reverse, starting at the end and going to our reset state
        aug_episode_start = skill_transition_indices[aug_transition_idx] - offset_steps
        for i in range(original_episode_horizon - 1, aug_episode_start, -1):
            self.reverse_actions.append(original_episode_actions[i])
            self.reverse_ee_poses.append(original_episode_ee_poses[i])
        
        # create an additional set of reverse actions by sampling an ee pose and interpolating
        start_robot_state = state["observations"][aug_episode_start]["robot_state"]
        start_ee_pos, start_ee_quat = start_robot_state["ee_pos"], start_robot_state["ee_quat"]

        # random delta position and euler angle
        xmax, ymax = 0.01, 0.01
        zmin, zmax = 0.095, 0.12
        rmax, pmax, yawmax = 15, 15, 15
        # rmax, pmax, ymax = 0.1, 0.1, 0.1
        dx, dy, dz = np.random.uniform(-xmax, xmax), np.random.uniform(-ymax, ymax), np.random.uniform(zmin, zmax)
        dr, dp, dyaw = np.random.uniform(-np.deg2rad(rmax), np.deg2rad(rmax)), np.random.uniform(-np.deg2rad(pmax), np.deg2rad(pmax)), np.random.uniform(-np.deg2rad(yawmax), np.deg2rad(yawmax))

        # get absolute "goal" pose
        goal_ee_pos = start_ee_pos + np.array([dx, dy, dz])

        start_ee_mat = st.Rotation.from_quat(start_ee_quat).as_matrix()
        delta_ee_mat = st.Rotation.from_euler('xyz', [dr, dp, dyaw]).as_matrix()
        goal_ee_quat = st.Rotation.from_matrix(start_ee_mat @ delta_ee_mat).as_quat()

        # interpolate and record delta actions
        interp_ee_pos = np.linspace(start_ee_pos, goal_ee_pos, n_back_steps)
        slerp = st.Slerp([0, 1], st.Rotation.from_quat([start_ee_quat, goal_ee_quat]))
        interp_ee_rot = slerp(np.linspace(0, 1, n_back_steps))

        self.env.reset_env_to(env_idx=0, state=state["observations"][aug_episode_start])
        self.env.refresh()

        print(f'Start position: {start_ee_pos}, goal position: {goal_ee_pos}')
        print(f'Start rvec: {st.Rotation.from_quat(start_ee_quat).as_rotvec()}, goal rvec: {st.Rotation.from_quat(goal_ee_quat).as_rotvec()}')

        print(f'Executing "reverse" actions... ')
        for i in range(n_back_steps - 1):
            
            # first, translate along our path
            action_pos = interp_ee_pos[i+1] - interp_ee_pos[i]

            # make quat action to keep same orientation
            action_quat = self.make_quat_action_stay(stay_quat=start_ee_quat)

            action = np.concatenate([action_pos, action_quat, [original_episode_actions[aug_episode_start][-1]]])

            action_t = torch.from_numpy(action).float().to(self.env.device)
            next_obs, _, _, _ = self.env.step(action_t)

            rev_action = np.zeros_like(action)
            rev_action[:3] = -1.0 * action_pos
            rev_action[3:-1] = st.Rotation.from_quat(action_quat).inv().as_quat()
            rev_action[-1] = action[-1]
            self.reverse_actions.append(rev_action)
            self.reverse_ee_poses.append(np.concatenate(self.get_ee_pose_np()))
        
        stay_pos = self.get_ee_pose_np()[0]

        for i in range(n_back_steps - 1):
            
            # make pos action to keep the same position
            action_pos = self.make_pos_action_stay(stay_pos=stay_pos)

            # now, follow rotation path
            action_quat = (interp_ee_rot[i].inv() * interp_ee_rot[i+1]).as_quat()

            action = np.concatenate([action_pos, action_quat, [original_episode_actions[aug_episode_start][-1]]])

            action_t = torch.from_numpy(action).float().to(self.env.device)
            next_obs, _, _, _ = self.env.step(action_t)

            rev_action = np.zeros_like(action)
            rev_action[:3] = -1.0 * action_pos
            rev_action[3:-1] = st.Rotation.from_quat(action_quat).inv().as_quat()
            rev_action[-1] = action[-1]
            self.reverse_actions.append(rev_action)
            self.reverse_ee_poses.append(np.concatenate(self.get_ee_pose_np()))

        return next_obs

    def get_ee_pose_np(self):
        translation, quat_xyzw = self.env.get_ee_pose()
        translation, quat_xyzw = (
            translation.cpu().numpy().squeeze(),
            quat_xyzw.cpu().numpy().squeeze(),
        )

        return translation, quat_xyzw

    def make_pos_action_stay(self, stay_pos):
        current_pos =self.get_ee_pose_np()[0]
        action_pos = stay_pos - current_pos
        return action_pos

    def make_quat_action_stay(self, stay_quat):
        current_quat = self.get_ee_pose_np()[1]
        stay_rot, current_rot = st.Rotation.from_quat(stay_quat), st.Rotation.from_quat(current_quat)
        action_quat = (current_rot.inv() * stay_rot).as_quat()
        return action_quat
    
    def create_delta_action(self, next_ee_pose, grip_action):
        current_ee_pos, current_ee_quat = self.get_ee_pose_np()
        next_ee_pos, next_ee_quat = next_ee_pose[:3], next_ee_pose[3:]

        action_pos = next_ee_pos - current_ee_pos
        action_quat = (st.Rotation.from_quat(current_ee_quat).inv() * st.Rotation.from_quat(next_ee_quat)).as_quat()

        action = np.concatenate([action_pos, action_quat, grip_action])

        return action

    def save(self, collect_enum: CollectEnum, info):
        print(f"Length of trajectory: {len(self.transitions)}")

        # Save transitions with resized images.
        data = {}
        data["observations"] = [t["observations"] for t in self.transitions]
        data["actions"] = [t["actions"] for t in self.transitions][:-1]
        data["rewards"] = [t["rewards"] for t in self.transitions][:-1]
        data["skills"] = [t["skills"] for t in self.transitions][:-1]
        data["success"] = True if collect_enum == CollectEnum.SUCCESS else False
        data["furniture"] = self.furniture

        if "error" in info:
            data["error_description"] = info["error"].value
            data["error"] = True
        else:
            data["error"] = False
            data["error_description"] = ""

        # Save data.
        demo_path = self.data_path / ("success" if data["success"] else "failure")
        demo_path.mkdir(parents=True, exist_ok=True)

        path = demo_path / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"

        if self.compress_pickles:
            # Add the suffix .gz if we are compressing the pickle files
            path = path.with_suffix(".pkl.xz")

        pickle_data(data, path)

        print(f"Data saved at {path}")

    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def update_pbar(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def __del__(self):
        del self.env

        if self.device_interface is not None:
            self.device_interface.close()


def main():
    parser = argparse.ArgumentParser(description="Collect IL data")
    # parser.add_argument(
    #     "--out-data-path", help="Path to directory to save the data", required=True
    # )
    parser.add_argument(
        "--furniture",
        help="Name of the furniture",
        choices=list(config["furniture"].keys()),
        required=True,
    )
    parser.add_argument(
        "--is-sim",
        action="store_true",
        help="Use simulator, else use real world environment.",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use scripted function for getting action.",
    )
    parser.add_argument(
        "--pkl-only",
        action="store_true",
        help="Only save the pickle file, not .mp4 and .pngs",
    )
    parser.add_argument(
        "--save-failure",
        action="store_true",
        help="Save failure trajectories.",
    )
    parser.add_argument(
        "--headless", help="With front camera view", action="store_true"
    )
    parser.add_argument(
        "--draw-marker", action="store_true", help="Draw AprilTag marker"
    )
    parser.add_argument(
        "--manual-label",
        action="store_true",
        help="Manually label the reward",
    )
    parser.add_argument("--randomness", default="low", choices=["low", "med", "high"])
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--num-demos", default=100, type=int)

    parser.add_argument(
        "--ctrl-mode",
        type=str,
        help="Type of low level controller to use.",
        choices=["osc", "diffik"],
        default="osc",
    )

    parser.add_argument(
        "--no-ee-laser",
        action="store_false",
        help="If set, will not show the laser coming from the end effector",
        dest="ee_laser",
    )

    args = parser.parse_args()

    keyboard_device_interface = make_device("keyboard")

    data_path = trajectory_save_dir(
        environment="sim" if args.is_sim else "real",
        task=args.furniture,
        demo_source="teleop",
        randomness=args.randomness,
    )

    from pathlib import Path
    # pickle_paths = list(
    #     Path(
    #         "/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/sim/one_leg/scripted"
    #     ).rglob("**/success/*.pkl*")
    # )

    pickle_paths = list(
        Path(
            "/home/anthony/repos/research/furniture-diffusion/furniture-data/raw/sim/one_leg/scripted"
        ).rglob("**/success/*.pkl*")
    )

    random.shuffle(pickle_paths)

    pickle_paths = pickle_paths[:10]

    print("loaded num trajectories", len(pickle_paths))

    data_collector = DataCollectorAugmentor(
        is_sim=args.is_sim,
        data_path=data_path,
        device_interface=keyboard_device_interface,
        furniture=args.furniture,
        headless=args.headless,
        draw_marker=args.draw_marker,
        manual_label=args.manual_label,
        resize_sim_img=False,
        scripted=args.scripted,
        randomness=args.randomness,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        pkl_only=args.pkl_only,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        ctrl_mode=args.ctrl_mode,
        ee_laser=args.ee_laser,
        compress_pickles=False,
        resume_trajectory_paths=pickle_paths,
    )
    data_collector.collect()


if __name__ == "__main__":
    main()