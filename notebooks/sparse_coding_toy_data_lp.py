#!/usr/bin/env python

# In[ ]:


# import os
# from collections.abc import Generator
# from dataclasses import dataclass, field
# from typing import Any, Optional, Union

# import numpy as np
# import numpy.typing as npt
# import torch
# import torch.multiprocessing as mp
# import torch.nn as nn
# import torch.optim as optim
# from torchtyping import TensorType
# from tqdm import tqdm


# @dataclass
# class ToyArgs:
#     device: Union[str, Any] = "cuda:0" if torch.cuda.is_available() else "cpu"
#     tied_ae: bool = False
#     seed: int = 3
#     learned_dict_ratio: float = 1.0
#     output_folder: str = "outputs"
#     # dtype: torch.dtype = torch.float32
#     activation_dim: int = 32
#     feature_prob_decay: float = 0.99
#     feature_num_nonzero: int = 8
#     correlated_components: bool = False
#     n_ground_truth_components: int = 32
#     batch_size: int = 8_192
#     lr: float = 4e-3
#     epochs: int = 50_000  # 50_000
#     n_components_dictionary: int = 64
#     n_components_dictionary_trans: int = 64
#     l1_alpha: float = 5e-3
#     lp_alphas: list[float] = field(
#         default_factory=lambda: [
#             1e-6,
#             3e-6,
#             1e-5,
#             3e-5,
#             1e-4,
#             3e-4,
#             1e-3,
#             3e-3,
#             1e-2,
#             3e-2,
#             1e-1,
#             3e-1,
#             1,
#             3,
#         ]
#     )
#     p: float = 0.5  # the p in the L_p norm used to induce sparsity
#     ps: list[float] = field(
#         default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]
#     )


# @dataclass
# class RandomDatasetGenerator(Generator):
#     activation_dim: int
#     n_ground_truth_components: int
#     batch_size: int
#     feature_num_nonzero: int
#     feature_prob_decay: float
#     correlated: bool
#     device: Union[torch.device, str]

#     feats: Optional[TensorType["n_ground_truth_components", "activation_dim"]] = None
#     generated_so_far: int = 0

#     frac_nonzero: float = field(init=False)
#     decay: TensorType["n_ground_truth_components"] = field(init=False)
#     corr_matrix: Optional[
#         TensorType["n_ground_truth_components", "n_ground_truth_components"]
#     ] = field(init=False)
#     component_probs: Optional[TensorType["n_ground_truth_components"]] = field(init=False)

#     def __post_init__(self):
#         self.frac_nonzero = self.feature_num_nonzero / self.n_ground_truth_components

#         # Define the probabilities of each component being included in the data
#         self.decay = torch.tensor(
#             [self.feature_prob_decay**i for i in range(self.n_ground_truth_components)]
#         ).to(self.device)  # FIXME: 1 / i

#         self.component_probs = self.decay * self.frac_nonzero  # Only if non-correlated
#         if self.feats is None:
#             self.feats = generate_rand_feats(
#                 self.activation_dim,
#                 self.n_ground_truth_components,
#                 device=self.device,
#             )

#     def send(self, ignored_arg: Any) -> TensorType["dataset_size", "activation_dim"]:
#         torch.manual_seed(self.generated_so_far)  # Set a deterministic seed for reproducibility
#         self.generated_so_far += 1

#         # Assuming generate_rand_dataset is your data generation function
#         _, ground_truth, data = generate_rand_dataset(
#             self.n_ground_truth_components,
#             self.batch_size,
#             self.component_probs,
#             self.feats,
#             self.device,
#         )
#         return ground_truth, data

#     def throw(self, type: Any = None, value: Any = None, traceback: Any = None) -> None:
#         raise StopIteration


# def generate_rand_dataset(
#     n_ground_truth_components: int,  #
#     dataset_size: int,
#     feature_probs: TensorType["n_ground_truth_components"],
#     feats: TensorType["n_ground_truth_components", "activation_dim"],
#     device: Union[torch.device, str],
# ) -> tuple[
#     TensorType["n_ground_truth_components", "activation_dim"],
#     TensorType["dataset_size", "n_ground_truth_components"],
#     TensorType["dataset_size", "activation_dim"],
# ]:
#     # generate random feature strengths
#     feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
#     # only some features are activated, chosen at random
#     dataset_thresh = torch.rand(dataset_size, n_ground_truth_components, device=device)
#     data_zero = torch.zeros_like(dataset_thresh, device=device)

#     dataset_codes = torch.where(
#         dataset_thresh <= feature_probs,
#         feature_strengths,
#         data_zero,
#     )  # dim: dataset_size x n_ground_truth_components

#     dataset = dataset_codes @ feats

#     return feats, dataset_codes, dataset


# def generate_rand_feats(
#     feat_dim: int,
#     num_feats: int,
#     device: Union[torch.device, str],
# ) -> TensorType["n_ground_truth_components", "activation_dim"]:
#     data_path = os.path.join(os.getcwd(), "data")
#     data_filename = os.path.join(data_path, f"feats_{feat_dim}_{num_feats}.npy")

#     feats = np.random.multivariate_normal(np.zeros(feat_dim), np.eye(feat_dim), size=num_feats)
#     feats = feats.T / np.linalg.norm(feats, axis=1)
#     feats = feats.T

#     feats_tensor = torch.from_numpy(feats).to(device).float()
#     return feats_tensor


# # AutoEncoder Definition
# class AutoEncoder(nn.Module):
#     def __init__(self, activation_size, n_dict_components):
#         super(AutoEncoder, self).__init__()

#         self.encoder = nn.Sequential(nn.Linear(activation_size, n_dict_components), nn.ReLU())
#         self.decoder = nn.Linear(n_dict_components, activation_size, bias=False)

#         # Initialize the decoder weights orthogonally
#         nn.init.orthogonal_(self.decoder.weight)

#     def forward(self, x):
#         c = self.encoder(x)

#         # Apply unit norm constraint to the decoder weights
#         self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

#         x_hat = self.decoder(c)
#         return x_hat, c

#     def get_dictionary(self):
#         self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)
#         return self.decoder.weight

#     @property
#     def device(self):
#         return next(self.parameters()).device


# def cosine_sim(
#     vecs1: Union[torch.Tensor, torch.nn.parameter.Parameter, npt.NDArray],
#     vecs2: Union[torch.Tensor, torch.nn.parameter.Parameter, npt.NDArray],
# ) -> np.ndarray:
#     vecs = [vecs1, vecs2]
#     for i in range(len(vecs)):
#         if not isinstance(vecs[i], np.ndarray):
#             vecs[i] = vecs[i].detach().cpu().numpy()  # type: ignore
#     vecs1, vecs2 = vecs
#     normalize = lambda v: (v.T / np.linalg.norm(v, axis=1)).T
#     vecs1_norm = normalize(vecs1)
#     vecs2_norm = normalize(vecs2)

#     return vecs1_norm @ vecs2_norm.T


# def mean_max_cosine_similarity(ground_truth_features, learned_dictionary, debug=False):
#     # Calculate cosine similarity between all pairs of ground truth and learned features
#     cos_sim = cosine_sim(ground_truth_features, learned_dictionary)
#     # Find the maximum cosine similarity for each ground truth feature, then average
#     mmcs = cos_sim.max(axis=1).mean()
#     return mmcs


# def calculate_mmcs(auto_encoder, ground_truth_features):
#     learned_dictionary = auto_encoder.decoder.weight.data.t()
#     with torch.no_grad():
#         mmcs = mean_max_cosine_similarity(
#             ground_truth_features.to(auto_encoder.device), learned_dictionary
#         )
#     return mmcs


# def get_alive_neurons(auto_encoder, data_generator, n_batches=10):
#     """
#     :param result_dict: dictionary containing the results of a single run
#     :return: number of dead neurons

#     Estimates the number of dead neurons in the network by running a few batches of data through the network and
#     calculating the mean activation of each neuron. If the mean activation is 0 for a neuron, it is considered dead.
#     """
#     outputs = []
#     for i in range(n_batches):
#         ground_truth, batch = next(data_generator)
#         x_hat, c = auto_encoder(
#             batch
#         )  # x_hat: (batch_size, activation_dim), c: (batch_size, n_dict_components)
#         outputs.append(c)
#     outputs = torch.cat(outputs)  # (n_batches * batch_size, n_dict_components)
#     mean_activations = outputs.mean(
#         dim=0
#     )  # (n_dict_components), c is after the ReLU, no need to take abs
#     alive_neurons = mean_activations > 0
#     return alive_neurons


# def train_model(arg):
#     (
#         worker_id,
#         epochs,
#         p_id,
#         p,
#         lp_id,
#         lp_alpha,
#         ground_truth_features,
#         cfg_dict,
#         init_seed,
#         device,
#         run_num,
#     ) = arg
#     torch.cuda.set_device(device)  # Set the device for the process

#     data_generator = RandomDatasetGenerator(
#         activation_dim=cfg_dict["activation_dim"],
#         n_ground_truth_components=cfg_dict["n_ground_truth_components"],
#         batch_size=cfg_dict["batch_size"],
#         feature_num_nonzero=cfg_dict["feature_num_nonzero"],
#         feature_prob_decay=cfg_dict["feature_prob_decay"],
#         correlated=cfg_dict["correlated_components"],
#         device=device,
#         feats=(ground_truth_features),
#         generated_so_far=init_seed,
#     )

#     auto_encoder = AutoEncoder(cfg_dict["activation_dim"], cfg_dict["n_components_dictionary"]).to(
#         device
#     )

#     optimizer = optim.Adam(auto_encoder.parameters(), lr=cfg_dict["lr"])

#     for _ in range(epochs):
#         ground_truth, batch = next(data_generator)

#         optimizer.zero_grad()

#         # Forward pass
#         x_hat, c = auto_encoder(batch)

#         # Compute the reconstruction loss and L1 regularization
#         l_reconstruction = torch.nn.MSELoss()(batch, x_hat)
#         lp_norm = torch.norm(c, p, dim=1).mean() / c.size(1)
#         l_lp = lp_alpha * lp_norm

#         # Print the losses, mmcs, and current epoch
#         # mmcs = calculate_mmcs(auto_encoder, ground_truth_features)
#         # sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
#         # num_dead_features = (c == 0).float().mean(dim=0).sum().cpu().item()

#         # log_prefix = f"{lp_alpha} L{p}"
#         # wandb_log = {
#         #     f"{log_prefix} MMCS": mmcs,
#         #     f"{log_prefix} Sparsity": sparsity,
#         #     f"{log_prefix} Dead Features": num_dead_features,
#         #     f"{log_prefix} Reconstruction Loss": l_reconstruction.detach().item(),
#         #     f"{log_prefix} Lp Loss": l_lp.detach().item(),
#         #     f"{log_prefix} Lp Norm": lp_norm.detach().item(),
#         #     f"{log_prefix} Tokens": epoch * cfg.batch_size,
#         # }

#         # Compute the total loss
#         loss = l_reconstruction + l_lp

#         # Backward pass
#         loss.backward()
#         optimizer.step()

#     # Save model
#     save_name = f"sae_l{p}_{lp_alpha}"
#     torch.save(auto_encoder, f"/root/sparsify/trained_models/toy_saes{run_num}/{save_name}.pt")


# def main():
#     cfg = ToyArgs()
#     cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     run_num = cfg.seed

#     torch.manual_seed(cfg.seed)
#     np.random.seed(cfg.seed)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     ground_truth_features = generate_rand_feats(
#         cfg.activation_dim,
#         cfg.n_ground_truth_components,
#         device=device,
#     )

#     mp.set_start_method("spawn")

#     if not os.path.exists(f"/root/sparsify/trained_models/toy_saes{run_num}"):
#         os.makedirs(f"/root/sparsify/trained_models/toy_saes{run_num}")

#     torch.save(
#         ground_truth_features,
#         f"/root/sparsify/trained_models/toy_saes{run_num}/ground_truth_features.pt",
#     )
#     args_list = []
#     for p_id, p in enumerate(cfg.ps):
#         for lp_id, lp_alpha in enumerate(cfg.lp_alphas):
#             # Pass necessary arguments to the worker function
#             args = (
#                 p_id * len(cfg.lp_alphas) + lp_id,
#                 cfg.epochs,
#                 p_id,
#                 p,
#                 lp_id,
#                 lp_alpha,
#                 ground_truth_features,
#                 # output_queue,
#                 cfg.__dict__,
#                 run_num,
#                 device,
#                 run_num,
#             )
#             args_list.append(args)

#     with mp.Pool(
#         processes=5
#     ) as pool:  # Adjust the number of processes based on your system's capabilities
#         max_ = len(args_list)
#         with tqdm(total=max_) as pbar:
#             for ret in pool.imap_unordered(train_model, args_list):
#                 pbar.update()


# if __name__ == "__main__":
#     main()


# In[ ]:


import os
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torchtyping import TensorType
from tqdm import tqdm


@dataclass
class ToyArgs:
    device: Union[str, Any] = "cuda:0" if torch.cuda.is_available() else "cpu"
    tied_ae: bool = False
    seed: int = 3
    learned_dict_ratio: float = 1.0
    output_folder: str = "outputs"
    # dtype: torch.dtype = torch.float32
    activation_dim: int = 32
    feature_prob_decay: float = 0.99
    feature_num_nonzero: int = 8
    correlated_components: bool = False
    n_ground_truth_components: int = 32
    batch_size: int = 8_192
    lr: float = 4e-3
    epochs: int = 1_000  # 50_000
    n_components_dictionary: int = 64
    n_components_dictionary_trans: int = 64
    l1_alpha: float = 5e-3
    lp_alphas: list[float] = field(
        default_factory=lambda: [
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            1e-1,
            3e-1,
            1,
            3,
        ]
    )
    p: float = 0.5  # the p in the L_p norm used to induce sparsity
    ps: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]
    )


@dataclass
class RandomDatasetGenerator(Generator):
    activation_dim: int
    n_ground_truth_components: int
    batch_size: int
    feature_num_nonzero: int
    feature_prob_decay: float
    correlated: bool
    device: Union[torch.device, str]

    feats: Optional[TensorType["n_ground_truth_components", "activation_dim"]] = None
    generated_so_far: int = 0

    frac_nonzero: float = field(init=False)
    decay: TensorType["n_ground_truth_components"] = field(init=False)
    corr_matrix: Optional[
        TensorType["n_ground_truth_components", "n_ground_truth_components"]
    ] = field(init=False)
    component_probs: Optional[TensorType["n_ground_truth_components"]] = field(init=False)

    def __post_init__(self):
        self.frac_nonzero = self.feature_num_nonzero / self.n_ground_truth_components

        # Define the probabilities of each component being included in the data
        self.decay = torch.tensor(
            [self.feature_prob_decay**i for i in range(self.n_ground_truth_components)]
        ).to(self.device)  # FIXME: 1 / i

        self.component_probs = self.decay * self.frac_nonzero  # Only if non-correlated
        if self.feats is None:
            self.feats = generate_rand_feats(
                self.activation_dim,
                self.n_ground_truth_components,
                device=self.device,
            )

    def send(self, ignored_arg: Any) -> TensorType["dataset_size", "activation_dim"]:
        torch.manual_seed(self.generated_so_far)  # Set a deterministic seed for reproducibility
        self.generated_so_far += 1

        # Assuming generate_rand_dataset is your data generation function
        _, ground_truth, data = generate_rand_dataset(
            self.n_ground_truth_components,
            self.batch_size,
            self.component_probs,
            self.feats,
            self.device,
        )
        return ground_truth, data

    def throw(self, type: Any = None, value: Any = None, traceback: Any = None) -> None:
        raise StopIteration


def generate_rand_dataset(
    n_ground_truth_components: int,  #
    dataset_size: int,
    feature_probs: TensorType["n_ground_truth_components"],
    feats: TensorType["n_ground_truth_components", "activation_dim"],
    device: Union[torch.device, str],
) -> tuple[
    TensorType["n_ground_truth_components", "activation_dim"],
    TensorType["dataset_size", "n_ground_truth_components"],
    TensorType["dataset_size", "activation_dim"],
]:
    # generate random feature strengths
    feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
    # only some features are activated, chosen at random
    dataset_thresh = torch.rand(dataset_size, n_ground_truth_components, device=device)
    data_zero = torch.zeros_like(dataset_thresh, device=device)

    dataset_codes = torch.where(
        dataset_thresh <= feature_probs,
        feature_strengths,
        data_zero,
    )  # dim: dataset_size x n_ground_truth_components

    dataset = dataset_codes @ feats

    return feats, dataset_codes, dataset


def generate_rand_feats(
    feat_dim: int,
    num_feats: int,
    device: Union[torch.device, str],
) -> TensorType["n_ground_truth_components", "activation_dim"]:
    data_path = os.path.join(os.getcwd(), "data")
    data_filename = os.path.join(data_path, f"feats_{feat_dim}_{num_feats}.npy")

    feats = np.random.multivariate_normal(np.zeros(feat_dim), np.eye(feat_dim), size=num_feats)
    feats = feats.T / np.linalg.norm(feats, axis=1)
    feats = feats.T

    feats_tensor = torch.from_numpy(feats).to(device).float()
    return feats_tensor


# AutoEncoder Definition
class AutoEncoder(nn.Module):
    def __init__(self, activation_size, n_dict_components):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(activation_size, n_dict_components), nn.ReLU())
        self.decoder = nn.Linear(n_dict_components, activation_size, bias=False)

        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x):
        c = self.encoder(x)

        # Apply unit norm constraint to the decoder weights
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

        x_hat = self.decoder(c)
        return x_hat, c

    def get_dictionary(self):
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)
        return self.decoder.weight

    @property
    def device(self):
        return next(self.parameters()).device


def cosine_sim(
    vecs1: Union[torch.Tensor, torch.nn.parameter.Parameter, npt.NDArray],
    vecs2: Union[torch.Tensor, torch.nn.parameter.Parameter, npt.NDArray],
) -> np.ndarray:
    vecs = [vecs1, vecs2]
    for i in range(len(vecs)):
        if not isinstance(vecs[i], np.ndarray):
            vecs[i] = vecs[i].detach().cpu().numpy()  # type: ignore
    vecs1, vecs2 = vecs
    normalize = lambda v: (v.T / np.linalg.norm(v, axis=1)).T
    vecs1_norm = normalize(vecs1)
    vecs2_norm = normalize(vecs2)

    return vecs1_norm @ vecs2_norm.T


def mean_max_cosine_similarity(ground_truth_features, learned_dictionary, debug=False):
    # Calculate cosine similarity between all pairs of ground truth and learned features
    cos_sim = cosine_sim(ground_truth_features, learned_dictionary)
    # Find the maximum cosine similarity for each ground truth feature, then average
    mmcs = cos_sim.max(axis=1).mean()
    return mmcs


def calculate_mmcs(auto_encoder, ground_truth_features):
    learned_dictionary = auto_encoder.decoder.weight.data.t()
    with torch.no_grad():
        mmcs = mean_max_cosine_similarity(
            ground_truth_features.to(auto_encoder.device), learned_dictionary
        )
    return mmcs


def get_alive_neurons(auto_encoder, data_generator, n_batches=10):
    """
    :param result_dict: dictionary containing the results of a single run
    :return: number of dead neurons

    Estimates the number of dead neurons in the network by running a few batches of data through the network and
    calculating the mean activation of each neuron. If the mean activation is 0 for a neuron, it is considered dead.
    """
    outputs = []
    for i in range(n_batches):
        ground_truth, batch = next(data_generator)
        x_hat, c = auto_encoder(
            batch
        )  # x_hat: (batch_size, activation_dim), c: (batch_size, n_dict_components)
        outputs.append(c)
    outputs = torch.cat(outputs)  # (n_batches * batch_size, n_dict_components)
    mean_activations = outputs.mean(
        dim=0
    )  # (n_dict_components), c is after the ReLU, no need to take abs
    alive_neurons = mean_activations > 0
    return alive_neurons


def train_model(arg):
    (
        worker_id,
        epochs,
        p_id,
        p,
        lp_id,
        lp_alpha,
        ground_truth_features,
        cfg_dict,
        init_seed,
        device,
        run_num,
    ) = arg
    torch.cuda.set_device(device)  # Set the device for the process

    data_generator = RandomDatasetGenerator(
        activation_dim=cfg_dict["activation_dim"],
        n_ground_truth_components=cfg_dict["n_ground_truth_components"],
        batch_size=cfg_dict["batch_size"],
        feature_num_nonzero=cfg_dict["feature_num_nonzero"],
        feature_prob_decay=cfg_dict["feature_prob_decay"],
        correlated=cfg_dict["correlated_components"],
        device=device,
        feats=(ground_truth_features),
        generated_so_far=init_seed,
    )

    auto_encoder = AutoEncoder(cfg_dict["activation_dim"], cfg_dict["n_components_dictionary"]).to(
        device
    )

    optimizer = optim.Adam(auto_encoder.parameters(), lr=cfg_dict["lr"])

    logs = []

    for ep in range(epochs):
        ground_truth, batch = next(data_generator)

        optimizer.zero_grad()

        # Forward pass
        x_hat, c = auto_encoder(batch)

        # Compute the reconstruction loss and L1 regularization
        l_reconstruction = torch.nn.MSELoss()(batch, x_hat)
        lp_norm = torch.norm(c, p, dim=1).mean() / c.size(1)
        l_lp = lp_alpha * lp_norm

        # Print the losses, mmcs, and current epoch
        mmcs = float(calculate_mmcs(auto_encoder, ground_truth_features).cpu().item * ())
        print("mmcs: ", type(float(mmcs)))
        sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
        num_dead_features = (c == 0).float().mean(dim=0).sum().cpu().item()

        log_prefix = f"{lp_alpha} L{p}"
        wandb_log = {
            # f"{log_prefix} MMCS": mmcs,
            f"{log_prefix} Sparsity": sparsity,
            f"{log_prefix} Dead Features": num_dead_features,
            f"{log_prefix} Reconstruction Loss": l_reconstruction.detach().item(),
            f"{log_prefix} Lp Loss": l_lp.detach().item(),
            f"{log_prefix} Lp Norm": lp_norm.detach().item(),
            f"{log_prefix} Tokens": ep * cfg_dict["batch_size"],
        }
        logs.append(wandb_log)

        # Compute the total loss
        loss = l_reconstruction + l_lp

        # Backward pass
        loss.backward()
        optimizer.step()

    # Save model
    save_name = f"sae_l{p}_{lp_alpha}"
    torch.save(auto_encoder, f"/root/sparsify/trained_models/toy_saes{run_num}/{save_name}.pt")
    return logs


def main():
    cfg = ToyArgs()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_num = cfg.seed

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ground_truth_features = generate_rand_feats(
        cfg.activation_dim,
        cfg.n_ground_truth_components,
        device=device,
    )

    mp.set_start_method("spawn")

    if not os.path.exists(f"/root/sparsify/trained_models/toy_saes{run_num}"):
        os.makedirs(f"/root/sparsify/trained_models/toy_saes{run_num}")

    torch.save(
        ground_truth_features,
        f"/root/sparsify/trained_models/toy_saes{run_num}/ground_truth_features.pt",
    )
    args_list = []
    for p_id, p in enumerate(cfg.ps):
        for lp_id, lp_alpha in enumerate(cfg.lp_alphas):
            # Pass necessary arguments to the worker function
            args = (
                p_id * len(cfg.lp_alphas) + lp_id,
                cfg.epochs,
                p_id,
                p,
                lp_id,
                lp_alpha,
                ground_truth_features,
                # output_queue,
                cfg.__dict__,
                run_num,
                device,
                run_num,
            )
            args_list.append(args)

    combined_logs = None
    with mp.Pool(
        processes=2
    ) as pool:  # Adjust the number of processes based on your system's capabilities
        max_ = len(args_list)
        with tqdm(total=max_) as pbar:
            for ret_logs in pool.imap_unordered(train_model, args_list):
                pbar.update()
                # if combined_logs is None:
                #     combined_logs = ret_logs
                # else:
                #     for log_step in range(len(combined_logs)):
                #         combined_logs[log_step] = combined_logs[log_step].update(ret_logs[log_step])

    # secrets = json.load(open("secrets.json"))
    # wandb.login(key=secrets["wandb_key"])
    # start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # wandb_run_name = f"Toylp_{start_time[4:]}_{cfg.batch_size}_{cfg.epochs}"  # trim year
    # print(f"wandb_run_name: {wandb_run_name}")
    # wandb.init(project="sparse coding", name=wandb_run_name)

    # for log_step in range(len(combined_logs)):
    #     wandb.log(combined_logs[log_step])


if __name__ == "__main__":
    main()
