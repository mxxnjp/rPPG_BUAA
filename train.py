import argparse
import csv
import fcntl
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from datasets.buaa_dataset_fast import BUAARppgDataset


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Loss
# ============================================================

def neg_pearson_loss_per_sample(pred, target, eps=1e-6):
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)

    numerator = (pred_c * target_c).sum(dim=1)

    denominator = torch.sqrt(
        (pred_c ** 2).sum(dim=1)
        *
        (target_c ** 2).sum(dim=1)
        +
        eps
    )

    corr = numerator / (denominator + eps)

    corr = torch.clamp(
        corr,
        min=-1.0,
        max=1.0
    )

    return 1.0 - corr


# ============================================================
# Subject split
# ============================================================

def read_subjects(path):
    with open(path, "r") as f:
        return {
            line.strip().zfill(2)
            for line in f
            if line.strip()
        }


def select_windows(ds, subjects, max_lux=None):
    selected = []
    lux_count = {}

    for i, win in enumerate(ds.windows):
        row = ds.rows[win[0]]

        subject = str(row["subject"]).zfill(2)
        lux = float(row["lux"])

        if subject not in subjects:
            continue

        if max_lux is not None and lux > max_lux:
            continue

        selected.append(i)

        key = round(lux, 1)
        lux_count[key] = lux_count.get(key, 0) + 1

    return selected, lux_count


# ============================================================
# Temporal Shift Module
# ============================================================

class TSM(nn.Module):
    def __init__(self, n_segment, fold_div=3):
        super().__init__()

        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.shape

        if nt % self.n_segment != 0:
            raise RuntimeError(
                f"TSM input frames {nt} must be divisible by "
                f"n_segment={self.n_segment}"
            )

        b = nt // self.n_segment

        x = x.view(
            b,
            self.n_segment,
            c,
            h,
            w
        )

        fold = c // self.fold_div

        out = torch.zeros_like(x)

        # future -> current
        out[:, :-1, :fold] = x[:, 1:, :fold]

        # past -> current
        out[:, 1:, fold:2 * fold] = (
            x[:, :-1, fold:2 * fold]
        )

        # unchanged
        out[:, :, 2 * fold:] = (
            x[:, :, 2 * fold:]
        )

        return out.view(
            nt,
            c,
            h,
            w
        )


# ============================================================
# EfficientPhys attention mask
# ============================================================

class AttentionMask(nn.Module):
    def forward(self, x):
        denom = x.sum(
            dim=(2, 3),
            keepdim=True
        ) + 1e-6

        h = x.shape[2]
        w = x.shape[3]

        return (
            x / denom
            * h
            * w
            * 0.5
        )


# ============================================================
# EfficientPhys feature backbone
# ============================================================

class EfficientPhysFeatureBackbone(nn.Module):
    """
    EfficientPhys-Conv backbone.

    Original architecture idea:
        TSM
        Conv
        TSM
        Conv
        spatial attention
        pooling
        TSM
        Conv
        TSM
        Conv
        spatial attention
        pooling
        dense -> 128 feature

    Output:
        (B, T, 128)
    """

    def __init__(
        self,
        frame_depth=10,
        nb_filters1=32,
        nb_filters2=64,
        nb_dense=128,
        dropout_rate1=0.25,
        dropout_rate2=0.5,
    ):
        super().__init__()

        self.frame_depth = frame_depth
        self.nb_dense = nb_dense

        self.tsm1 = TSM(
            n_segment=frame_depth
        )

        self.tsm2 = TSM(
            n_segment=frame_depth
        )

        self.tsm3 = TSM(
            n_segment=frame_depth
        )

        self.tsm4 = TSM(
            n_segment=frame_depth
        )

        self.conv1 = nn.Conv2d(
            3,
            nb_filters1,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            nb_filters1,
            nb_filters1,
            kernel_size=3
        )

        self.conv3 = nn.Conv2d(
            nb_filters1,
            nb_filters2,
            kernel_size=3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            nb_filters2,
            nb_filters2,
            kernel_size=3
        )

        self.att1 = nn.Conv2d(
            nb_filters1,
            1,
            kernel_size=1
        )

        self.att2 = nn.Conv2d(
            nb_filters2,
            1,
            kernel_size=1
        )

        self.att_mask = AttentionMask()

        self.pool1 = nn.AvgPool2d(
            kernel_size=2
        )

        self.pool2 = nn.AvgPool2d(
            kernel_size=2
        )

        self.dropout1 = nn.Dropout(
            dropout_rate1
        )

        self.dropout2 = nn.Dropout(
            dropout_rate1
        )

        self.dropout3 = nn.Dropout(
            dropout_rate1
        )

        self.dropout4 = nn.Dropout(
            dropout_rate2
        )

        # EfficientPhys original img_size=36
        # flattened dimension = 3136
        self.fc1 = nn.Linear(
            3136,
            nb_dense
        )

    def forward(self, x):
        """
        x:
            (B,T,3,36,36)

        return:
            feature_seq (B,T,128)
        """

        B, T, C, H, W = x.shape

        if T % self.frame_depth != 0:
            raise RuntimeError(
                f"T={T} must be divisible by "
                f"frame_depth={self.frame_depth}"
            )

        # ----------------------------------------------------
        # Important:
        # TSM must see independent temporal blocks.
        #
        # B,T -> B*num_chunks, frame_depth
        # ----------------------------------------------------

        n_chunks = T // self.frame_depth

        x = x.reshape(
            B,
            n_chunks,
            self.frame_depth,
            C,
            H,
            W
        )

        x = x.reshape(
            B * n_chunks * self.frame_depth,
            C,
            H,
            W
        )

        x = self.tsm1(x)

        x = torch.tanh(
            self.conv1(x)
        )

        x = self.tsm2(x)

        x = torch.tanh(
            self.conv2(x)
        )

        g1 = torch.sigmoid(
            self.att1(x)
        )

        g1 = self.att_mask(g1)

        x = x * g1

        x = self.pool1(x)

        x = self.dropout1(x)

        x = self.tsm3(x)

        x = torch.tanh(
            self.conv3(x)
        )

        x = self.tsm4(x)

        x = torch.tanh(
            self.conv4(x)
        )

        g2 = torch.sigmoid(
            self.att2(x)
        )

        g2 = self.att_mask(g2)

        x = x * g2

        x = self.pool2(x)

        x = self.dropout3(x)

        x = x.flatten(1)

        if x.shape[1] != 3136:
            raise RuntimeError(
                f"EfficientPhys flatten dim expected 3136, "
                f"got {x.shape[1]}"
            )

        x = torch.tanh(
            self.fc1(x)
        )

        x = self.dropout4(x)

        x = x.reshape(
            B,
            T,
            self.nb_dense
        )

        return x


# ============================================================
# Model A: EfficientPhys
# ============================================================

class EfficientPhysBaseline(nn.Module):
    def __init__(
        self,
        frame_depth=10,
    ):
        super().__init__()

        self.backbone = (
            EfficientPhysFeatureBackbone(
                frame_depth=frame_depth
            )
        )

        self.head = nn.Linear(
            128,
            1
        )

    def forward(self, frames):
        feat = self.backbone(frames)

        pred = self.head(
            feat
        ).squeeze(-1)

        return pred


# ============================================================
# Model B: EfficientPhys + BiGRU
# ============================================================

class EfficientPhysBiGRU(nn.Module):
    def __init__(
        self,
        frame_depth=10,
        hidden_dim=128,
    ):
        super().__init__()

        self.backbone = (
            EfficientPhysFeatureBackbone(
                frame_depth=frame_depth
            )
        )

        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.head = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, frames):
        feat = self.backbone(frames)

        temporal, _ = self.bigru(
            feat
        )

        pred = self.head(
            temporal
        ).squeeze(-1)

        return pred


# ============================================================
# Model C: EfficientPhys + GRU (unidirectional)
# ============================================================

class EfficientPhysGRU(nn.Module):
    def __init__(
        self,
        frame_depth=10,
        hidden_dim=128,
    ):
        super().__init__()

        self.backbone = (
            EfficientPhysFeatureBackbone(
                frame_depth=frame_depth
            )
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        self.head = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, frames):
        feat = self.backbone(frames)

        temporal, _ = self.gru(
            feat
        )

        pred = self.head(
            temporal
        ).squeeze(-1)

        return pred


# ============================================================
# BUAA -> EfficientPhys input adapter
# ============================================================

def prepare_efficientphys_input(frames):
    """
    frames:
        (B,T,3,112,112)

    output:
        (B,T,3,36,36)

    EfficientPhys repository expects preprocessed
    temporal-difference input.

    Here BUAA RGB is converted to normalized temporal
    difference explicitly.
    """

    B, T, C, H, W = frames.shape

    flat = frames.reshape(
        B * T,
        C,
        H,
        W
    )

    flat = F.interpolate(
        flat,
        size=(36, 36),
        mode="bilinear",
        align_corners=False
    )

    x = flat.reshape(
        B,
        T,
        C,
        36,
        36
    )

    # temporal difference WITHIN each sequence
    diff = torch.zeros_like(x)

    diff[:, 1:] = (
        x[:, 1:] - x[:, :-1]
    )

    diff[:, 0] = diff[:, 1]

    # per-frame standardization
    mean = diff.mean(
        dim=(2, 3, 4),
        keepdim=True
    )

    std = diff.std(
        dim=(2, 3, 4),
        keepdim=True
    )

    diff = (
        diff - mean
    ) / (
        std + 1e-6
    )

    return diff


# ============================================================
# Train / validation
# ============================================================

def run_epoch(
    model,
    loader,
    device,
    optimizer=None
):
    train_mode = (
        optimizer is not None
    )

    if train_mode:
        model.train()
    else:
        model.eval()

    losses = []
    lux_losses = {}

    context = (
        torch.enable_grad()
        if train_mode
        else torch.no_grad()
    )

    with context:

        for batch in loader:

            frames = batch[
                "frames"
            ].to(device)

            target = batch[
                "bvp"
            ].to(device)

            lux_values = batch[
                "lux"
            ].cpu().numpy()

            x = prepare_efficientphys_input(
                frames
            )

            if train_mode:
                optimizer.zero_grad()

            pred = model(x)

            loss_each = (
                neg_pearson_loss_per_sample(
                    pred,
                    target
                )
            )

            loss = loss_each.mean()

            if train_mode:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

            cpu_losses = (
                loss_each.detach()
                .cpu()
                .numpy()
            )

            losses.extend(
                cpu_losses.tolist()
            )

            for lux, lv in zip(
                lux_values,
                cpu_losses
            ):
                key = round(
                    float(lux),
                    1
                )

                lux_losses.setdefault(
                    key,
                    []
                ).append(
                    float(lv)
                )

    mean_loss = float(
        np.mean(losses)
    )

    pearson = 1.0 - mean_loss

    lux_result = {
        k: float(np.mean(v))
        for k, v in lux_losses.items()
    }

    return (
        mean_loss,
        pearson,
        lux_result
    )


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        choices=[
            "efficientphys",
            "efficientphys_gru",
            "efficientphys_bigru"
        ],
        required=True
    )

    ap.add_argument(
        "--labels",
        default="./buaa_experiment/labels.csv"
    )

    ap.add_argument(
        "--train_subjects",
        default="./buaa_experiment/train_subjects.txt"
    )

    ap.add_argument(
        "--val_subjects",
        default="./buaa_experiment/val_subjects.txt"
    )

    ap.add_argument(
        "--window_size",
        type=int,
        default=150
    )

    ap.add_argument(
        "--stride",
        type=int,
        default=75
    )

    ap.add_argument(
        "--frame_depth",
        type=int,
        default=10
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=2
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    ap.add_argument(
        "--val_max_lux",
        type=float,
        default=4.0
    )

    ap.add_argument(
        "--max_train_windows",
        type=int,
        default=120
    )

    ap.add_argument(
        "--max_val_windows",
        type=int,
        default=80
    )

    ap.add_argument(
        "--hidden_dim",
        type=int,
        default=128
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    ap.add_argument(
        "--device",
        default="cuda"
    )

    ap.add_argument(
        "--split",
        default="A",
        help="split label used only for results CSV logging"
    )

    ap.add_argument(
        "--results_csv",
        default="results.csv"
    )

    ap.add_argument(
        "--roi",
        choices=["full", "multi"],
        default="full",
        help="full=전체 얼굴 프레임, multi=이마+양볼 ROI만 마스킹"
    )

    ap.add_argument(
        "--roi_boxes",
        default="buaa_experiment/roi_boxes.json"
    )

    args = ap.parse_args()

    set_seed(
        args.seed
    )

    if (
        args.window_size
        % args.frame_depth
        != 0
    ):
        raise ValueError(
            "window_size must be divisible "
            "by frame_depth"
        )

    device = torch.device(
        args.device
    )

    train_subjects = read_subjects(
        args.train_subjects
    )

    val_subjects = read_subjects(
        args.val_subjects
    )

    print("=" * 75)
    print("BUAA EfficientPhys A/B experiment")
    print("=" * 75)

    print(
        "model         :",
        args.model
    )

    print(
        "roi           :",
        args.roi
    )

    print(
        "window        :",
        args.window_size
    )

    print(
        "frame_depth   :",
        args.frame_depth
    )

    print(
        "train subject :",
        sorted(train_subjects)
    )

    print(
        "val subject   :",
        sorted(val_subjects)
    )

    print(
        "val max lux   :",
        args.val_max_lux
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    ds = BUAARppgDataset(
        args.labels,
        window_size=args.window_size,
        stride=args.stride,
        cache_dir="/tmp/buaa_dataset_fast_cache",
        roi_mode=args.roi,
        roi_boxes_path=args.roi_boxes if args.roi == "multi" else None
    )

    train_idx, train_lux = (
        select_windows(
            ds,
            train_subjects,
            max_lux=None
        )
    )

    val_idx, val_lux = (
        select_windows(
            ds,
            val_subjects,
            max_lux=args.val_max_lux
        )
    )

    print(
        "\ntrain available:",
        len(train_idx)
    )

    print(
        "val available:",
        len(val_idx)
    )

    print(
        "train lux:",
        train_lux
    )

    print(
        "val lux:",
        val_lux
    )

    # fixed subset
    rng = random.Random(
        args.seed
    )

    rng.shuffle(
        train_idx
    )

    rng.shuffle(
        val_idx
    )

    if args.max_train_windows:
        train_idx = train_idx[
            :args.max_train_windows
        ]

    if args.max_val_windows:
        val_idx = val_idx[
            :args.max_val_windows
        ]

    print(
        "\nUSED train:",
        len(train_idx)
    )

    print(
        "USED val:",
        len(val_idx)
    )

    train_loader = DataLoader(
        Subset(
            ds,
            train_idx
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        Subset(
            ds,
            val_idx
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if args.model == "efficientphys":

        model = EfficientPhysBaseline(
            frame_depth=args.frame_depth
        )

    elif args.model == "efficientphys_gru":

        model = EfficientPhysGRU(
            frame_depth=args.frame_depth,
            hidden_dim=args.hidden_dim
        )

    else:

        model = EfficientPhysBiGRU(
            frame_depth=args.frame_depth,
            hidden_dim=args.hidden_dim
        )

    model = model.to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nparameters:",
        n_params
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val = float(
        "inf"
    )

    best_epoch = None
    best_val_lux_result = {}

    checkpoint = (
        f"checkpoint_buaa_{args.model}_split{args.split}_roi{args.roi}_seed{args.seed}.pt"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        args.epochs
    ):

        t0 = time.time()

        (
            train_loss,
            train_p,
            _
        ) = run_epoch(
            model,
            train_loader,
            device,
            optimizer
        )

        (
            val_loss,
            val_p,
            val_lux_result
        ) = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None
        )

        print(
            "\n"
            + "=" * 75
        )

        print(
            f"[EPOCH "
            f"{epoch+1:02d}/"
            f"{args.epochs}]"
        )

        print(
            f"train loss="
            f"{train_loss:.4f} "
            f"pearson="
            f"{train_p:.4f}"
        )

        print(
            f"VAL loss="
            f"{val_loss:.4f} "
            f"pearson="
            f"{val_p:.4f}"
        )

        print(
            "val lux loss:",
            end=" "
        )

        for lux in sorted(
            val_lux_result
        ):

            l = val_lux_result[
                lux
            ]

            print(
                f"{lux:g}="
                f"{l:.3f}"
                f"(p={1-l:.3f})",
                end="  "
            )

        print()

        print(
            f"time="
            f"{time.time()-t0:.1f}s"
        )

        if val_loss < best_val:

            best_val = val_loss
            best_epoch = epoch + 1
            best_val_lux_result = val_lux_result

            torch.save(
                model.state_dict(),
                checkpoint
            )

            print(
                "[BEST] saved",
                checkpoint
            )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL MODEL:",
        args.model
    )

    print(
        "BEST VAL LOSS:",
        round(
            best_val,
            4
        )
    )

    print(
        "BEST VAL PEARSON:",
        round(
            1.0 - best_val,
            4
        )
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # Results CSV logging
    # (model, split, seed, lux, best_val_pearson, epoch)
    # one row per lux subset + one "overall" row
    # --------------------------------------------------------

    csv_exists = os.path.exists(
        args.results_csv
    )

    with open(
        args.results_csv,
        "a",
        newline=""
    ) as f:

        fcntl.flock(f, fcntl.LOCK_EX)

        writer = csv.writer(f)

        if not csv_exists:
            writer.writerow([
                "model",
                "split",
                "roi",
                "seed",
                "lux",
                "best_val_pearson",
                "epoch"
            ])

        writer.writerow([
            args.model,
            args.split,
            args.roi,
            args.seed,
            "overall",
            round(1.0 - best_val, 4),
            best_epoch
        ])

        for lux in sorted(best_val_lux_result):
            writer.writerow([
                args.model,
                args.split,
                args.roi,
                args.seed,
                lux,
                round(1.0 - best_val_lux_result[lux], 4),
                best_epoch
            ])


if __name__ == "__main__":
    main()

