# Tensorboard view
# ~/.local/share/ov/pkg/isaac_sim-2021.2.1/python.sh ~/.local/share/ov/pkg/isaac_sim-2021.2.1/tensorboard --logdir ./

# Real time nvidia-smi
# watch -n0.1 nvidia-smi

# Train Commands
# cd ~/.local/share/ov/pkg/isaac_sim-2021.2.1/DRL_Isaac_lib/
# ~/.local/share/ov/pkg/isaac_sim-2021.2.1/python.sh train_d.py

from tabnanny import verbose
from env import Isaac_envs
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
from torch import nn
import gym

log_dir = "./mlp_policy"
# set headles to false to visualize training
my_env = Isaac_envs(headless=False, max_episode_length=3000)

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors_policy = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            # if key == "image_depth":
            #     extractors_policy[key] = nn.Sequential(nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            #                                            nn.ReLU(),
            #                                            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            #                                            nn.ReLU(),
            #                                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            #                                            nn.ReLU(),
            #                                            nn.Flatten(),
            #                                            nn.Linear(9216, 256),
            #                                            nn.ReLU(),
            #                                            nn.Linear(256, 64),
            #                                            nn.ReLU(),
            #                                            nn.Linear(64, 4),)
            #     total_concat_size += 4

            # if key == "h_raleted":
            #     extractors_policy[key] = nn.Sequential(nn.Flatten(),
            #                                            nn.Linear(10, 256),
            #                                            nn.ReLU(),
            #                                            nn.Linear(256, 256),
            #                                            nn.ReLU(),
            #                                            nn.Linear(256, 4),
            #                                            nn.ReLU(),)
            #     total_concat_size += 4

            if key == "IR_raleted":
                extractors_policy[key] = nn.Sequential(nn.Flatten(),)
                                                    #    nn.Linear(24, 256),
                                                    #    nn.ReLU(),
                                                    #    nn.Linear(256, 128),
                                                    #    nn.ReLU(),
                                                    #    nn.Linear(128, 64),
                                                    #    nn.ReLU(),
                                                    #    nn.Linear(64, 32),
                                                    #    nn.ReLU(),
                                                    #    nn.Linear(32, 4),
                                                    #    nn.ReLU(),)
                total_concat_size += 12
            
            if key == "pos_raleted":
                extractors_policy[key] = nn.Sequential(nn.Flatten(),)
                #                                        nn.Linear(2, 256),
                #                                        nn.ReLU(),
                #                                        nn.Linear(256, 64),
                #                                        nn.ReLU(),
                #                                        nn.Linear(64, 1),
                #                                        nn.ReLU(),)
                total_concat_size += 2

            if key == "vel_raleted":
                extractors_policy[key] = nn.Sequential(nn.Flatten(),)
                                                    #    nn.Linear(2, 256),
                                                    #    nn.ReLU(),
                                                    #    nn.Linear(256, 64),
                                                    #    nn.ReLU(),
                                                    #    nn.Linear(64, 1),
                                                    #    nn.ReLU(),)
                total_concat_size += 2

            # if key == "mapita":
            #     extractors_policy[key] = nn.Sequential(nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=0),
            #                                            nn.ReLU(),
            #                                            nn.Conv2d(8, 1, kernel_size=2, stride=1, padding=0),
            #                                            nn.ReLU(),
            #                                            nn.Flatten(),
            #                                            nn.Linear(9, 64),
            #                                            nn.ReLU(),
            #                                            nn.Linear(64, 16),
            #                                            nn.ReLU(),
            #                                            nn.Linear(16, 4),)
            #     total_concat_size += 4

        self.extractors_policy = nn.ModuleDict(extractors_policy)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list_policy = []

        # self.extractors contain nn.Modules that do all the processing.
        # policy
        for key, extractor in self.extractors_policy.items():
            encoded_tensor_list_policy.append(extractor(observations[key]))
        
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        self.policy_net = th.cat(encoded_tensor_list_policy, dim=1)
        
        return self.policy_net

policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor, 
                     activation_fn=th.nn.ReLU, 
                     net_arch=[512, 512, 512],
                     )


policy = "MultiInputPolicy"
total_timesteps = 3000000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="jetbot_policy_checkpoint")


# model = PPO(
#     policy,
#     my_env,
#     policy_kwargs=policy_kwargs,
#     verbose=1,
#     # n_steps=10000,
#     # batch_size=1000,
#     # learning_rate=0.00015,
#     # gamma=0.9995,
#     device="cuda",
#     # ent_coef=0,
#     # vf_coef=0.5,
#     # max_grad_norm=10,
#     # clip_range=1,
#     tensorboard_log=log_dir,
# )

model = DQN(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    buffer_size=800000,
    learning_starts=1000, 
    learning_rate=0.00015,
    exploration_fraction=0.35,
    device="cuda",
    tensorboard_log=log_dir,
)


model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(log_dir + "/jetbot_policy")

my_env.close()
