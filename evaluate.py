import argparse
import csv
import os

import numpy as np
import torch

from torch.utils.data import DataLoader, Subset

from datasets.buaa_dataset_fast import BUAARppgDataset

from train import (
    read_subjects,
    select_windows,
    prepare_efficientphys_input,
    neg_pearson_loss_per_sample,
    EfficientPhysBaseline,
    EfficientPhysGRU,
    EfficientPhysBiGRU,
)


FPS = 30.0
HR_LOW_HZ = 0.7
HR_HIGH_HZ = 4.0
NFFT = 4096


def estimate_hr_bpm(signal, fps=FPS, nfft=NFFT):
    """
    signal: 1D array (T,) predicted or ground-truth waveform.

    0.7~4Hz(42~240bpm) 대역에서 zero-padded FFT의 최대 파워 주파수를
    심박수(bpm)로 변환한다.
    """
    signal = signal - np.mean(signal)

    spectrum = np.fft.rfft(signal, n=nfft)
    power = np.abs(spectrum) ** 2

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)

    band = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)

    band_power = power[band]
    band_freqs = freqs[band]

    peak_idx = np.argmax(band_power)
    peak_freq = band_freqs[peak_idx]

    return peak_freq * 60.0, freqs, power


def compute_snr_db(pred_signal, true_hr_bpm, fps=FPS, nfft=NFFT, tol_hz=0.1):
    """
    true_hr_bpm 주변(+-tol_hz)과 그 2차 하모닉 주변을 signal band로,
    0.7~4Hz 대역의 나머지를 noise band로 보고 SNR(dB)을 계산한다.
    """
    pred_signal = pred_signal - np.mean(pred_signal)

    spectrum = np.fft.rfft(pred_signal, n=nfft)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)

    band = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)

    true_hr_hz = true_hr_bpm / 60.0

    signal_mask = np.zeros_like(band)
    for harmonic in (1, 2):
        center = true_hr_hz * harmonic
        signal_mask |= (freqs >= center - tol_hz) & (freqs <= center + tol_hz)

    signal_mask &= band
    noise_mask = band & (~signal_mask)

    signal_power = power[signal_mask].sum() + 1e-12
    noise_power = power[noise_mask].sum() + 1e-12

    return 10.0 * np.log10(signal_power / noise_power)


def build_model(model_name, frame_depth, hidden_dim):
    if model_name == "efficientphys":
        return EfficientPhysBaseline(frame_depth=frame_depth)
    elif model_name == "efficientphys_gru":
        return EfficientPhysGRU(frame_depth=frame_depth, hidden_dim=hidden_dim)
    else:
        return EfficientPhysBiGRU(frame_depth=frame_depth, hidden_dim=hidden_dim)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True,
                     choices=["efficientphys", "efficientphys_gru", "efficientphys_bigru"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--labels", default="./buaa_experiment/labels.csv")
    ap.add_argument("--train_subjects", default="./buaa_experiment/train_subjects.txt")
    ap.add_argument("--val_subjects", default="./buaa_experiment/val_subjects.txt")
    ap.add_argument("--max_train_windows", type=int, default=120)
    ap.add_argument("--window_size", type=int, default=150)
    ap.add_argument("--stride", type=int, default=75)
    ap.add_argument("--frame_depth", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--val_max_lux", type=float, default=4.0)
    ap.add_argument("--max_val_windows", type=int, default=80)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="A")
    ap.add_argument("--roi", default="full")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results_csv", default="metrics_eval.csv")

    args = ap.parse_args()

    device = torch.device(args.device)

    import random
    random.seed(args.seed)

    train_subjects = read_subjects(args.train_subjects)
    val_subjects = read_subjects(args.val_subjects)

    ds = BUAARppgDataset(
        args.labels,
        window_size=args.window_size,
        stride=args.stride,
        cache_dir="/tmp/buaa_dataset_fast_cache",
    )

    # train.py와 완전히 동일한 순서로 RNG를 소비해야 train.py가 학습에 쓴
    # 것과 정확히 같은 val subset(및 순서)을 복원할 수 있다.
    train_idx, _ = select_windows(ds, train_subjects, max_lux=None)
    val_idx, _ = select_windows(ds, val_subjects, max_lux=args.val_max_lux)

    rng = random.Random(args.seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    if args.max_train_windows:
        train_idx = train_idx[:args.max_train_windows]

    if args.max_val_windows:
        val_idx = val_idx[:args.max_val_windows]

    loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = build_model(args.model, args.frame_depth, args.hidden_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    per_lux = {}

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            target = batch["bvp"].to(device)
            lux_values = batch["lux"].cpu().numpy()

            x = prepare_efficientphys_input(frames)
            pred = model(x)

            pearson_each = 1.0 - neg_pearson_loss_per_sample(pred, target)

            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            pearson_np = pearson_each.cpu().numpy()

            true_hr_np = batch["hr"].cpu().numpy()

            for i in range(pred_np.shape[0]):
                lux_key = round(float(lux_values[i]), 1)

                true_hr = float(np.mean(true_hr_np[i]))
                pred_hr, _, _ = estimate_hr_bpm(pred_np[i])
                snr_db = compute_snr_db(pred_np[i], true_hr)

                err = pred_hr - true_hr

                entry = per_lux.setdefault(lux_key, {
                    "pearson": [], "abs_err": [], "sq_err": [],
                    "pct_err": [], "acc5": [], "snr": [],
                })

                entry["pearson"].append(float(pearson_np[i]))
                entry["abs_err"].append(abs(err))
                entry["sq_err"].append(err ** 2)
                entry["pct_err"].append(abs(err) / max(true_hr, 1e-6) * 100.0)
                entry["acc5"].append(1.0 if abs(err) <= 5.0 else 0.0)
                entry["snr"].append(snr_db)

    def summarize(entry):
        return {
            "pearson": float(np.mean(entry["pearson"])),
            "hr_mae": float(np.mean(entry["abs_err"])),
            "hr_rmse": float(np.sqrt(np.mean(entry["sq_err"]))),
            "hr_mape": float(np.mean(entry["pct_err"])),
            "acc5bpm": float(np.mean(entry["acc5"]) * 100.0),
            "snr_db": float(np.mean(entry["snr"])),
        }

    all_entries = {"pearson": [], "abs_err": [], "sq_err": [], "pct_err": [], "acc5": [], "snr": []}
    for entry in per_lux.values():
        for k in all_entries:
            all_entries[k].extend(entry[k])

    overall = summarize(all_entries)

    print("=" * 75)
    print(f"model={args.model} split={args.split} roi={args.roi} seed={args.seed}")
    print("=" * 75)
    print(f"OVERALL  pearson={overall['pearson']:.4f}  "
          f"HR_MAE={overall['hr_mae']:.2f}bpm  HR_RMSE={overall['hr_rmse']:.2f}bpm  "
          f"HR_MAPE={overall['hr_mape']:.2f}%  Acc@5bpm={overall['acc5bpm']:.1f}%  "
          f"SNR={overall['snr_db']:.2f}dB")

    for lux in sorted(per_lux):
        s = summarize(per_lux[lux])
        print(f"  lux={lux:g}  pearson={s['pearson']:.4f}  "
              f"HR_MAE={s['hr_mae']:.2f}bpm  HR_RMSE={s['hr_rmse']:.2f}bpm  "
              f"HR_MAPE={s['hr_mape']:.2f}%  Acc@5bpm={s['acc5bpm']:.1f}%  "
              f"SNR={s['snr_db']:.2f}dB")

    csv_exists = os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="") as f:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["model", "split", "roi", "seed", "lux",
                              "pearson", "hr_mae", "hr_rmse", "hr_mape", "acc5bpm", "snr_db"])

        writer.writerow([args.model, args.split, args.roi, args.seed, "overall"] +
                         [round(overall[k], 4) for k in
                          ("pearson", "hr_mae", "hr_rmse", "hr_mape", "acc5bpm", "snr_db")])

        for lux in sorted(per_lux):
            s = summarize(per_lux[lux])
            writer.writerow([args.model, args.split, args.roi, args.seed, lux] +
                             [round(s[k], 4) for k in
                              ("pearson", "hr_mae", "hr_rmse", "hr_mape", "acc5bpm", "snr_db")])


if __name__ == "__main__":
    main()
