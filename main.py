from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from noise_build import dataset_split


# =========================================================
# Args / Config
# =========================================================

@dataclass(frozen=True)
class DatasetStats:
    num_classes: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


def get_dataset_stats(dataset: str) -> DatasetStats:
    if dataset == "cifar10":
        return DatasetStats(10, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    return DatasetStats(100, (0.507, 0.487, 0.441), (0.267, 0.256, 0.276))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMA Co-Teaching + FkL selection (cleaned)")

    p.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    p.add_argument("--root_dir", type=str, default=None,
                   help="If not set: ./cifar-10 or ./cifar-100")
    p.add_argument("--epoch", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--r", type=float, default=0.4, help="noise ratio in [0,1]")
    p.add_argument("--argsseed", type=int, default=1)
    p.add_argument("--noise_mode", type=str, default="cifarn")
    p.add_argument("--model_type", type=str, default="resnet101")
    p.add_argument("--warm_up_epochs", type=int, default=6)

    p.add_argument("--remove_rate_1", type=float, default=0.0)
    p.add_argument("--remove_rate_2", type=float, default=0.0)
    p.add_argument("--remove_rate_3", type=float, default=0.0)
    p.add_argument("--remove_rate_4", type=float, default=0.0)

    p.add_argument("--i_rate_1", type=int, default=0)
    p.add_argument("--i_rate_2", type=int, default=0)
    p.add_argument("--i_rate_3", type=int, default=0)
    p.add_argument("--i_rate_4", type=int, default=0)

    # legacy name kept, but now treated as COUNT
    p.add_argument("--newremove_rate", type=int, required=True,
                   help="how many highest-loss samples to consider (round 0)")
    p.add_argument("--early_cutting_rate", type=float, required=True,
                   help="subset size = int(len(FkL)/early_cutting_rate), must be > 0")
    p.add_argument("--global_file_name", type=str, default="global")
    p.add_argument("--num_workers", type=int, default=16)

    args = p.parse_args()

    if args.root_dir is None:
        args.root_dir = "./cifar-10" if args.dataset == "cifar10" else "./cifar-100"

    if not (0.0 <= args.r <= 1.0):
        raise ValueError("--r must be in [0,1]")
    if args.early_cutting_rate <= 0:
        raise ValueError("--early_cutting_rate must be > 0")
    if args.newremove_rate < 0:
        raise ValueError("--newremove_rate must be >= 0")

    return args


# =========================================================
# Reproducibility
# =========================================================

def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================
# EMA (fixed)
# =========================================================

class ModelEMA:
    def __init__(self, model: nn.Module, ema_model: nn.Module, alpha: float = 0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha

        self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def step(self) -> None:
        a = self.alpha
        # params
        for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
            if not ema_p.dtype.is_floating_point:
                ema_p.copy_(p)
            else:
                ema_p.mul_(a).add_(p, alpha=1.0 - a)
        # buffers (BN running stats, etc.)
        for b, ema_b in zip(self.model.buffers(), self.ema_model.buffers()):
            ema_b.copy_(b)


# =========================================================
# Data
# =========================================================

@dataclass
class DataCache:
    train_data: np.ndarray
    train_clean_labels: np.ndarray
    train_noisy_labels: np.ndarray
    test_data: np.ndarray
    test_labels: np.ndarray
    val_data: np.ndarray
    val_noisy_labels: np.ndarray


class IndexedNumpyDataset(Dataset):
    """Return (img_tensor, label, base_idx). base_idx indexes into the underlying arrays."""
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose,
        indices: Optional[Sequence[int]] = None,
    ):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.indices = np.arange(len(data)) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        base_idx = int(self.indices[i])
        img = Image.fromarray(self.data[base_idx])
        img = self.transform(img)
        label = int(self.labels[base_idx])
        return img, label, base_idx


class CifarNoisyDataModule:
    def __init__(self, dataset: str, root_dir: str, noise_ratio: float, noise_mode: str,
                 seed: int, stats: DatasetStats):
        self.dataset = dataset
        self.root_dir = root_dir
        self.noise_ratio = noise_ratio
        self.noise_mode = noise_mode
        self.seed = seed
        self.stats = stats

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(stats.mean, stats.std),
        ])
        self.transform_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats.mean, stats.std),
        ])

        self.cache = self._load_and_prepare()

    @staticmethod
    def _unpickle(path: str):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f, encoding="bytes")

    def _load_and_prepare(self) -> DataCache:
        print("============ Initialize data")

        if self.dataset == "cifar10":
            xs = []
            ys: List[int] = []
            for n in range(1, 6):
                d = self._unpickle(f"{self.root_dir}/data_batch_{n}")
                xs.append(d[b"data"])
                ys += d[b"labels"]
            train_data = np.concatenate(xs)
            test_dic = self._unpickle(f"{self.root_dir}/test_batch")
            test_data = test_dic[b"data"]
            test_labels = test_dic[b"labels"]
        else:
            train_dic = self._unpickle(f"{self.root_dir}/train")
            train_data = train_dic[b"data"]
            ys = train_dic[b"fine_labels"]
            test_dic = self._unpickle(f"{self.root_dir}/test")
            test_data = test_dic[b"data"]
            test_labels = test_dic[b"fine_labels"]

        train_data = train_data.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
        test_data = test_data.reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))

        noisy = dataset_split(
            train_images=train_data,
            train_labels=ys,
            noise_rate=self.noise_ratio,
            noise_type=self.noise_mode,
            random_seed=self.seed,
            num_classes=self.stats.num_classes,
        )
        noisy = np.asarray(noisy)
        ys = np.asarray(ys)

        print("============ Actual clean samples number: ", int(np.sum(noisy == ys)))

        rng = np.random.RandomState(self.seed)
        num_samples = int(noisy.shape[0])
        train_idx = rng.choice(num_samples, int(num_samples * 0.9), replace=False)
        all_idx = np.arange(num_samples)
        val_idx = np.delete(all_idx, train_idx)

        train_set = train_data[train_idx]
        val_set = train_data[val_idx]
        train_noisy = noisy[train_idx]
        val_noisy = noisy[val_idx]
        train_clean = ys[train_idx]

        return DataCache(
            train_data=train_set,
            train_clean_labels=train_clean,
            train_noisy_labels=train_noisy,
            test_data=test_data,
            test_labels=np.asarray(test_labels),
            val_data=val_set,
            val_noisy_labels=val_noisy,
        )

    def make_loader(
        self,
        split: str,
        indices: Optional[Sequence[int]],
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        drop_last: bool,
        use_train_transform: bool,
    ) -> DataLoader:
        if split == "train":
            data, labels = self.cache.train_data, self.cache.train_noisy_labels
        elif split == "val":
            data, labels = self.cache.val_data, self.cache.val_noisy_labels
        elif split == "test":
            data, labels = self.cache.test_data, self.cache.test_labels
        else:
            raise ValueError(split)

        tfm = self.transform_train if use_train_transform else self.transform_eval
        ds = IndexedNumpyDataset(data, labels, tfm, indices=indices)

        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )


# =========================================================
# Models
# =========================================================

def build_model(model_type: str, num_classes: int, device: torch.device, ema: bool = False) -> nn.Module:
    if model_type == "resnet18":
        from resnetnew import ResNet18
        m = ResNet18(num_classes)
    elif model_type == "resnet34":
        from resnetnew import ResNet34
        m = ResNet34(num_classes)
    elif model_type == "resnet50":
        from resnetnew import ResNet50
        m = ResNet50(num_classes)
    elif model_type == "resnet101":
        from resnetnew import ResNet101
        m = ResNet101(num_classes)
    else:
        raise ValueError(model_type)

    m = m.to(device)
    if ema:
        for p in m.parameters():
            p.requires_grad_(False)
    return m


def ensemble_logits(models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module], x: torch.Tensor) -> torch.Tensor:
    net1, net2, net1e, net2e = models
    return (net1(x) + net2(x) + net1e(x) + net2e(x)) / 4.0


# =========================================================
# Train / Eval
# =========================================================

def train_one_epoch_mutual_kd(
    net1: nn.Module,
    net2: nn.Module,
    net1_ema: nn.Module,
    net2_ema: nn.Module,
    ema1: ModelEMA,
    ema2: ModelEMA,
    opt1: optim.Optimizer,
    opt2: optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> None:
    net1.train()
    net2.train()

    for x, y, _ in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            t1 = net1(x)
            t2 = net2(x)
            t1e = net1_ema(x)
            t2e = net2_ema(x)

            soft1 = F.softmax(t1 / temperature, dim=1)
            soft2 = F.softmax(t2 / temperature, dim=1)
            soft1e = F.softmax(t1e / temperature, dim=1)
            soft2e = F.softmax(t2e / temperature, dim=1)

            soft1_comb = (soft1 + soft1e) / 2.0
            soft2_comb = (soft2 + soft2e) / 2.0

        # net1
        z1 = net1(x)
        hard1 = F.cross_entropy(z1, y)
        soft1_loss = F.kl_div(F.log_softmax(z1 / temperature, dim=1), soft2_comb,
                              reduction="batchmean") * (temperature ** 2)
        loss1 = hard1 + alpha * soft1_loss

        opt1.zero_grad(set_to_none=True)
        loss1.backward()
        opt1.step()
        ema1.step()

        # net2
        z2 = net2(x)
        hard2 = F.cross_entropy(z2, y)
        soft2_loss = F.kl_div(F.log_softmax(z2 / temperature, dim=1), soft1_comb,
                              reduction="batchmean") * (temperature ** 2)
        loss2 = hard2 + alpha * soft2_loss

        opt2.zero_grad(set_to_none=True)
        loss2.backward()
        opt2.step()
        ema2.step()


@torch.inference_mode()
def eval_accuracy(models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module],
                  loader: DataLoader,
                  device: torch.device) -> float:
    for m in models:
        m.eval()

    correct, total = 0, 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = ensemble_logits(models, x).argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / total


EpochRecord = List[Tuple[int, int, bool]]


@torch.inference_mode()
def eval_and_collect(models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module],
                     loader: DataLoader,
                     device: torch.device) -> Tuple[float, EpochRecord]:
    for m in models:
        m.eval()

    correct, total = 0, 0
    record: EpochRecord = []
    for x, y, base_idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = ensemble_logits(models, x)
        pred = logits.argmax(dim=1)
        corr = (pred == y)

        total += y.size(0)
        correct += corr.sum().item()

        pred_np = pred.cpu().numpy().astype(np.int64)
        idx_np = base_idx.cpu().numpy().astype(np.int64)
        corr_np = corr.cpu().numpy()

        for p, i, c in zip(pred_np, idx_np, corr_np):
            record.append((int(p), int(i), bool(c)))

    return 100.0 * correct / total, record


# =========================================================
# FkL selection
# =========================================================

def count_mislabeled(indices: Sequence[int], noisy: np.ndarray, clean: np.ndarray) -> int:
    if not indices:
        return 0
    idx = np.asarray(indices, dtype=np.int64)
    return int(np.sum(noisy[idx] != clean[idx]))


def compute_fkl(
    epoch_data: Dict[Tuple[int, int], EpochRecord],
    round_num: int,
    current_epoch: int,
    warm_up_epochs: int,
    num_rounds: int,
    i1: int, i2: int, i3: int, i4: int,
) -> Tuple[int, List[int]]:
    # final round: no selection
    if round_num == num_rounds - 1:
        return 0, []

    rn = round_num + 1
    b1 = i1
    b2 = i1 + i2
    b3 = i1 + i2 + i3
    b4 = i1 + i2 + i3 + i4

    if rn <= b1:
        phase = 1
    elif rn <= b2:
        phase = 2
    elif rn <= b3:
        phase = 3
    else:
        phase = 4  # rn <= b4 expected

    # infer dataset size from first usable epoch
    n = None
    for pe in range(current_epoch):
        if pe > warm_up_epochs:
            n = len(epoch_data[(round_num, pe)])
            break
    if n is None:
        return 0, []

    counter = [0] * n
    prev1 = [-1] * n
    prev2 = [-1] * n
    prev3 = [-1] * n

    selected: List[int] = []
    for pe in range(current_epoch):
        if pe <= warm_up_epochs:
            continue
        reader = epoch_data[(round_num, pe)]
        if len(reader) != n:
            raise RuntimeError("epoch record length changed within a round")

        if phase == 1:
            for t, (pred, base_idx, ok) in enumerate(reader):
                if ok:
                    counter[t] += 1
                    if counter[t] == 1:
                        selected.append(base_idx)

        elif phase == 2:
            for t, (pred, base_idx, ok) in enumerate(reader):
                if ok and pred == prev1[t]:
                    counter[t] += 1
                    if counter[t] == 1:
                        selected.append(base_idx)
                prev1[t] = pred

        elif phase == 3:
            for t, (pred, base_idx, ok) in enumerate(reader):
                if ok and pred == prev1[t] and pred == prev2[t]:
                    counter[t] += 1
                    if counter[t] == 1:
                        selected.append(base_idx)
                prev2[t] = prev1[t]
                prev1[t] = pred

        else:  # phase 4
            for t, (pred, base_idx, ok) in enumerate(reader):
                if ok and pred == prev1[t] and pred == prev2[t] and pred == prev3[t]:
                    counter[t] += 1
                    if counter[t] == 1:
                        selected.append(base_idx)
                prev3[t] = prev2[t]
                prev2[t] = prev1[t]
                prev1[t] = pred

    return len(selected), selected


def pick_remove_rate_by_phase(rn: int,
                              i1: int, i2: int, i3: int,
                              r1: float, r2: float, r3: float, r4: float) -> float:
    """Map remove_rate_1..4 to phase1..4."""
    if rn <= i1:
        return r1
    if rn <= i1 + i2:
        return r2
    if rn <= i1 + i2 + i3:
        return r3
    return r4


# =========================================================
# Round-0 refinement
# =========================================================

@torch.inference_mode()
def loss_ranking(models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module],
                 loader: DataLoader,
                 device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    all_idx: List[int] = []
    all_loss: List[float] = []

    for x, y, base_idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = ensemble_logits(models, x)
        loss = F.cross_entropy(logits, y, reduction="none")  # FIX: logits, not log(prob)

        all_idx.extend(base_idx.numpy().astype(np.int64).tolist())
        all_loss.extend(loss.cpu().numpy().astype(np.float32).tolist())

    return np.asarray(all_idx, dtype=np.int64), np.asarray(all_loss, dtype=np.float32)


def refine_delete_by_conf_and_grad(
    models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    loader: DataLoader,
    device: torch.device,
    top_conf_ratio: float = 0.2,
    low_grad_ratio: float = 0.2,
) -> List[int]:
    """
    Minimal refinement that actually affects outputs:
    refined = (top confidence) ∩ (low grad-norm)
    """
    conf: Dict[int, float] = {}
    gradn: Dict[int, float] = {}

    for x, y, base_idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        base_idx = base_idx.to(device, non_blocking=True)

        x.requires_grad_(True)

        logits = ensemble_logits(models, x)
        probs = F.softmax(logits, dim=1)
        conf_b = probs.max(dim=1).values

        losses = F.cross_entropy(logits, y, reduction="none")
        g = torch.autograd.grad(losses.sum(), x)[0]
        gnorm = torch.norm(g.flatten(1), dim=1)

        for i in range(x.size(0)):
            idx = int(base_idx[i].item())
            conf[idx] = float(conf_b[i].item())
            gradn[idx] = float(gnorm[i].item())

        x.requires_grad_(False)

    if not conf:
        return []

    items_conf = sorted(conf.items(), key=lambda kv: kv[1], reverse=True)
    k_conf = max(1, int(len(items_conf) * top_conf_ratio))
    top_conf = set(i for i, _ in items_conf[:k_conf])

    items_grad = sorted(gradn.items(), key=lambda kv: kv[1], reverse=True)
    k_low = max(1, int(len(items_grad) * low_grad_ratio))
    low_grad = set(i for i, _ in items_grad[-k_low:])

    return list(top_conf.intersection(low_grad))


def refine_round0(
    best_ckpt_path: str,
    models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    dm: CifarNoisyDataModule,
    fkl_indices: List[int],
    early_cutting_rate: float,
    num_remove: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[List[int], float]:
    net1, net2, net1e, net2e = models
    ckpt = torch.load(best_ckpt_path, map_location=device)
    net1.load_state_dict(ckpt["net1"])
    net2.load_state_dict(ckpt["net2"])
    net1e.load_state_dict(ckpt["net1_ema"])
    net2e.load_state_dict(ckpt["net2_ema"])
    print(f"Loaded best ckpt: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%")

    # early cut
    subset_len = int(len(fkl_indices) / early_cutting_rate)
    subset_len = max(1, min(subset_len, len(fkl_indices)))
    subset = fkl_indices[:subset_len]

    subset_loader = dm.make_loader(
        split="train",
        indices=subset,
        batch_size=256,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        use_train_transform=False,
    )
    idx_arr, loss_arr = loss_ranking(models, subset_loader, device)

    order = np.argsort(loss_arr)  # low -> high
    sorted_idx = idx_arr[order]

    delete_count = min(num_remove, int(sorted_idx.shape[0]))
    if delete_count <= 0:
        return fkl_indices, 0.0

    delete_idx = sorted_idx[-delete_count:]  # highest-loss
    noisy = dm.cache.train_noisy_labels[delete_idx]
    clean = dm.cache.train_clean_labels[delete_idx]
    delete_mislabeled_rate = float(np.mean(noisy != clean))
    print(f"Delete candidates: {delete_count}, mislabeled rate: {delete_mislabeled_rate:.2%}")

    delete_loader = dm.make_loader(
        split="train",
        indices=delete_idx.tolist(),
        batch_size=64,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        use_train_transform=False,
    )
    refined_delete = refine_delete_by_conf_and_grad(models, delete_loader, device,
                                                    top_conf_ratio=0.2, low_grad_ratio=0.2)
    refined_set = set(refined_delete)
    new_global = [i for i in fkl_indices if i not in refined_set]
    print(f"Refined delete removed: {len(refined_delete)} samples from FkL set.")
    return new_global, delete_mislabeled_rate


# =========================================================
# Main
# =========================================================

def main() -> None:
    args = parse_args()
    stats = get_dataset_stats(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.argsseed)

    print(f"Dataset={args.dataset}, root_dir={args.root_dir}, device={device}")

    dm = CifarNoisyDataModule(
        dataset=args.dataset,
        root_dir=args.root_dir,
        noise_ratio=args.r,
        noise_mode=args.noise_mode,
        seed=args.argsseed,
        stats=stats,
    )

    # Base loaders
    train0 = dm.make_loader("train", None, args.batch_size, args.num_workers,
                            shuffle=True, drop_last=True, use_train_transform=True)
    etrain0 = dm.make_loader("train", None, 1000, args.num_workers,
                             shuffle=False, drop_last=False,
                             use_train_transform=True)  # keep original behavior
    test_loader = dm.make_loader("test", None, 1000, args.num_workers,
                                 shuffle=False, drop_last=False, use_train_transform=False)
    val_loader = dm.make_loader("val", None, 1000, args.num_workers,
                                shuffle=False, drop_last=False, use_train_transform=False)

    num_rounds = args.i_rate_1 + args.i_rate_2 + args.i_rate_3 + args.i_rate_4 + 1

    run_name = (
        f"Results_ema_coteaching_ce_{args.dataset}_"
        f"rm1{args.remove_rate_1}_rm2{args.remove_rate_2}_rm3{args.remove_rate_3}_rm4{args.remove_rate_4}_"
        f"{args.noise_mode}{args.r}_lr{args.lr}_bs{args.batch_size}_"
        f"ECR{args.early_cutting_rate}_removeN{args.newremove_rate}"
    )
    print("Run name:", run_name)

    # per-epoch log
    with open(run_name + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "epoch", "etrain_acc", "test_acc", "val_acc",
                    "fkl_count", "mislabeled", "clean"])

    epoch_data: Dict[Tuple[int, int], EpochRecord] = {}
    global_indices: List[int] = []
    best_test_acc = 0.0
    delete_mislabeled_rate: Optional[float] = None

    # for global summary compatibility
    first_iter_last_epoch_mislabeled: Optional[int] = None
    first_iter_last_epoch_clean: Optional[int] = None

    print("===================Main===================")
    print("Start Training!")

    for round_num in range(num_rounds):
        print(f"\n========== Round {round_num}/{num_rounds-1} ==========")

        # models
        net1 = build_model(args.model_type, stats.num_classes, device)
        net2 = build_model(args.model_type, stats.num_classes, device)
        net1e = build_model(args.model_type, stats.num_classes, device, ema=True)
        net2e = build_model(args.model_type, stats.num_classes, device, ema=True)
        models = (net1, net2, net1e, net2e)

        ema1 = ModelEMA(net1, net1e, alpha=0.999)
        ema2 = ModelEMA(net2, net2e, alpha=0.999)

        opt1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        opt2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.epoch, eta_min=1e-5)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.epoch, eta_min=1e-5)

        # loaders for this round
        if round_num == 0:
            train_loader = train0
            etrain_loader = etrain0
            train_size = len(dm.cache.train_data)
        else:
            if not global_indices:
                raise RuntimeError("global_indices is empty. Round 0 did not finish properly.")
            train_loader = dm.make_loader("train", global_indices, args.batch_size, args.num_workers,
                                          shuffle=True, drop_last=True, use_train_transform=True)
            etrain_loader = dm.make_loader("train", global_indices, 1000, args.num_workers,
                                           shuffle=False, drop_last=False, use_train_transform=True)
            train_size = len(global_indices)

        best_val_acc = -1.0
        best_ckpt_path = f"{run_name}_round{round_num}_best_model.pth"
        round_finished = False

        for epoch in range(args.epoch):
            print(f"\nEpoch {epoch + 1}/{args.epoch} (lr={opt1.param_groups[0]['lr']:.6f})")

            train_one_epoch_mutual_kd(net1, net2, net1e, net2e, ema1, ema2,
                                      opt1, opt2, train_loader, device,
                                      temperature=2.0, alpha=0.5)
            sch1.step()
            sch2.step()

            etrain_acc, rec = eval_and_collect(models, etrain_loader, device)
            epoch_data[(round_num, epoch)] = rec

            test_acc = eval_accuracy(models, test_loader, device)
            val_acc = eval_accuracy(models, val_loader, device)
            best_test_acc = max(best_test_acc, test_acc)

            print(f"etrain={etrain_acc:.3f}%  test={test_acc:.3f}%  val={val_acc:.3f}%")

            # checkpoint best val
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "net1": net1.state_dict(),
                    "net2": net2.state_dict(),
                    "net1_ema": net1e.state_dict(),
                    "net2_ema": net2e.state_dict(),
                }, best_ckpt_path)

            # FkL
            fkl_count, fkl_idx = compute_fkl(
                epoch_data, round_num, epoch, args.warm_up_epochs, num_rounds,
                args.i_rate_1, args.i_rate_2, args.i_rate_3, args.i_rate_4
            )

            rn = round_num + 1
            remove_rate = pick_remove_rate_by_phase(
                rn,
                args.i_rate_1, args.i_rate_2, args.i_rate_3,
                args.remove_rate_1, args.remove_rate_2, args.remove_rate_3, args.remove_rate_4,
            )
            threshold = int(train_size * remove_rate)

            mislabeled = count_mislabeled(fkl_idx, dm.cache.train_noisy_labels, dm.cache.train_clean_labels)
            clean = fkl_count - mislabeled

            with open(run_name + ".csv", "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([round_num, epoch, f"{etrain_acc:.2f}", f"{test_acc:.2f}", f"{val_acc:.2f}",
                            fkl_count, mislabeled, clean])

            if round_num == 0:
                first_iter_last_epoch_mislabeled = mislabeled
                first_iter_last_epoch_clean = clean

            print(f"FkL={fkl_count}  threshold={threshold}  mislabeled={mislabeled}  clean={clean}")

            # proceed only if we have something to train on
            if fkl_count > 0 and fkl_count >= threshold:
                if round_num == 0:
                    global_indices, delete_mislabeled_rate = refine_round0(
                        best_ckpt_path, models, dm, fkl_idx,
                        early_cutting_rate=args.early_cutting_rate,
                        num_remove=args.newremove_rate,
                        num_workers=args.num_workers,
                        device=device,
                    )
                else:
                    global_indices = fkl_idx

                print(f"Round {round_num} finished at epoch {epoch}. Selected={len(global_indices)}")
                round_finished = True
                break

        if not round_finished and round_num < num_rounds - 1:
            # In original script this situation is undefined and often leads to empty subset later.
            print(f"WARNING: Round {round_num} never reached threshold within {args.epoch} epochs. "
                  f"Stopping further rounds to avoid empty subset.")
            break

    # Final subset noise rate
    if global_indices:
        idx = np.asarray(global_indices, dtype=np.int64)
        final_noise_rate = float(np.mean(dm.cache.train_noisy_labels[idx] != dm.cache.train_clean_labels[idx]))
    else:
        final_noise_rate = 0.0

    # global summary row
    global_csv = args.global_file_name + ".csv"
    need_header = (not os.path.exists(global_csv)) or (os.path.getsize(global_csv) == 0)
    with open(global_csv, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow([
                "newremove_N",
                "early_cutting_rate",
                "remove_rate_1",
                "remove_rate_2",
                "remove_rate_3",
                "remove_rate_4",
                "best_test_acc",
                "round0_last_mislabeled",
                "round0_last_clean",
                "final_subset_noise_rate",
                "delete_candidates_mislabeled_rate",
                "final_subset_size",
            ])
        w.writerow([
            args.newremove_rate,
            args.early_cutting_rate,
            args.remove_rate_1,
            args.remove_rate_2,
            args.remove_rate_3,
            args.remove_rate_4,
            best_test_acc,
            first_iter_last_epoch_mislabeled,
            first_iter_last_epoch_clean,
            final_noise_rate,
            delete_mislabeled_rate,
            len(global_indices),
        ])

    print("\n=================== Done ===================")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Final subset size: {len(global_indices)}")
    print(f"Final subset noise rate: {final_noise_rate:.2%}")
    if delete_mislabeled_rate is not None:
        print(f"Delete candidates mislabeled rate (round 0): {delete_mislabeled_rate:.2%}")


if __name__ == "__main__":
    main()