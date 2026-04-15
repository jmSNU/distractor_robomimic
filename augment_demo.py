import h5py
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from copy import deepcopy
from tqdm import tqdm


def perturb_camera(env, pos_noise=0.01, angle_noise=1.0):
    sim = env.env.sim  # robosuite env

    for cam_name in sim.model.camera_names:
        cam_id = sim.model.camera_name2id(cam_name)

        # position perturb
        sim.model.cam_pos[cam_id] += np.random.randn(3) * pos_noise

        # orientation perturb (axis-angle small rotation)
        axis = np.random.randn(3)
        axis /= (np.linalg.norm(axis) + 1e-8)
        angle = np.random.randn() * angle_noise * np.pi / 180.0

        # convert to rotation matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        sim.model.cam_mat0[cam_id] = (R @ sim.model.cam_mat0[cam_id].reshape(3, 3)).reshape(-1)


def perturb_lighting(env, light_noise=0.1):
    sim = env.env.sim

    for i in range(sim.model.light_pos.shape[0]):
        sim.model.light_pos[i] += np.random.randn(3) * light_noise
        sim.model.light_diffuse[i] = np.clip(
            sim.model.light_diffuse[i] + np.random.randn(3) * 0.05, 0, 1
        )

def infer_obs_specs(f):
    demo = list(f["data"].keys())[0]
    print(f["data"][demo].keys())
    obs_keys = list(f[f"data/{demo}/obs"].keys())

    rgb_keys = []
    depth_keys = []
    low_dim_keys = []

    for k in obs_keys:
        if k.endswith("_image"):
            rgb_keys.append(k)
        elif k.endswith("_depth"):
            depth_keys.append(k)
        else:
            low_dim_keys.append(k)

    obs_specs = {
        "obs": {
            "low_dim": low_dim_keys,
            "rgb": rgb_keys,
        }
    }

    if len(depth_keys) > 0:
        obs_specs["obs"]["depth"] = depth_keys

    return obs_specs

def to_hwc(img):
    img = np.array(img)

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))

    # grayscale
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    return img

def replay_and_augment(input_hdf5, output_hdf5):
    f = h5py.File(input_hdf5, "r")

    # env metadata
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=input_hdf5)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,   # 추가
    )
    obs_specs = infer_obs_specs(f)
    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs=obs_specs
    )

    out_f = h5py.File(output_hdf5, "w")
    data_grp = out_f.create_group("data")

    demo_keys = list(f["data"].keys())

    for ep_i, ep in enumerate(tqdm(demo_keys, "Augmenting Demos")):
        ep_grp = f["data"][ep]

        states = ep_grp["states"][()]
        actions = ep_grp["actions"][()]

        obs_list = []
        next_obs_list = []
        act_list = []
        rew_list = []
        done_list = []

        obs = env.reset()
        env.env.sim.set_state_from_flattened(states[0])
        env.env.sim.forward()

        for t in range(len(actions)):
            perturb_camera(env)
            perturb_lighting(env)

            next_obs, reward, done, _ = env.step(actions[t])

            obs_list.append(deepcopy(obs))
            next_obs_list.append(deepcopy(next_obs))
            act_list.append(actions[t])
            rew_list.append(reward)
            done_list.append(done)

            obs = next_obs

        ep_out = data_grp.create_group(f"demo_{ep_i}")

        # flatten obs dict (robomimic format)
        for k in obs_list[0]:
            data = np.stack([o[k] for o in obs_list], axis=0)

            if "image" in k or "rgb" in k:
                data = np.stack([to_hwc(x) for x in data], axis=0)

            ep_out.create_dataset(f"obs/{k}", data=data)

        for k in next_obs_list[0]:
            data = np.stack([o[k] for o in next_obs_list], axis=0)

            if "image" in k or "rgb" in k:
                data = np.stack([to_hwc(x) for x in data], axis=0)

            ep_out.create_dataset(f"next_obs/{k}", data=data)

        ep_out.create_dataset("actions", data=np.array(act_list))
        ep_out.create_dataset("rewards", data=np.array(rew_list))
        ep_out.create_dataset("dones", data=np.array(done_list))

    f.close()
    out_f.close()


if __name__ == "__main__":
    replay_and_augment(
        input_hdf5="data/lift_ph.hdf5",
        output_hdf5="data/lift_aug_ph.hdf5",
    )