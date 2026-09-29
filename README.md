# Low-Light rPPG on BUAA-MIHR: EfficientPhys + GRU / BiGRU

BUAA-MIHR 저조도(1 ~ 4 lux) 영상에서 **EfficientPhys** 백본에 시간축 모델(GRU / BiGRU)을 붙였을 때
rPPG 성능이 어떻게 변하는지 비교하는 실험 코드.

| 모델 | 설명 |
|---|---|
| `efficientphys` | Baseline |
| `efficientphys_gru` | EfficientPhys + GRU |
| `efficientphys_bigru` | EfficientPhys + BiGRU |

세 모델은 동일한 백본, 동일한 학습 조건, 동일한 subject split에서 비교함.
Split 3종(A/B/C) × 시드 여러 개 × ROI 2종(`full` / `multi`)으로 반복 실험함.

---

## 결과 요약

Overall Pearson correlation (높을수록 좋음), HR MAE (bpm, 낮을수록 좋음), SNR (dB, 높을수록 좋음).
ROI = `full`, 시드 평균이며 표의 n은 시드 수. 원본은 [`metrics_eval.csv`](metrics_eval.csv) 임.

| Split | 모델 | n | Pearson | HR MAE | SNR |
|---|---|---|---|---|---|
| A | EfficientPhys | 6 | 0.132 | 35.4 | -6.06 |
| A | + GRU | 5 | 0.138 | 22.3 | -6.06 |
| A | + BiGRU | 5 | **0.156** | 23.7 | -5.90 |
| B | EfficientPhys | 5 | 0.063 | 32.8 | -5.73 |
| B | + GRU | 5 | 0.098 | 18.0 | -5.70 |
| B | + BiGRU | 5 | **0.109** | **12.6** | **-4.67** |
| C | EfficientPhys | 5 | 0.069 | 45.2 | -7.83 |
| C | + GRU | 5 | 0.084 | 25.7 | -7.65 |
| C | + BiGRU | 5 | **0.110** | **20.6** | **-7.04** |

- 모든 split에서 **BiGRU > GRU > Baseline** 순으로 Pearson이 높음.
- HR MAE는 Baseline 대비 GRU/BiGRU에서 크게 줄어듦 (split B: 32.8 → 12.6 bpm).
- 절대 성능은 낮음 (Pearson 0.06 ~ 0.16). 1 ~ 4 lux 극저조도이고 학습 데이터가 작은(120 windows) 설정이기 때문임.
  모델 간 상대 비교로 보는 것이 적절함.

---

## 저장소 구조

```text
.
├── train.py                 # 학습 / 검증 (모델 정의 포함)
├── evaluate.py              # 체크포인트 평가 (Pearson, HR MAE/RMSE/MAPE, Acc@5bpm, SNR)
├── precompute_roi.py        # 얼굴 ROI 박스 사전 계산 (MediaPipe face landmarker)
├── set_data_path.py         # labels.csv 의 video_path 를 내 데이터 경로로 변경
├── datasets/
│   ├── buaa_dataset_fast.py # BUAA-MIHR 데이터셋 로더
│   └── roi_utils.py         # ROI 유틸
├── buaa_experiment/
│   ├── roi_boxes.json       # 사전 계산된 ROI
│   ├── train_subjects*.txt  # split A(기본) / B / C
│   └── val_subjects*.txt
├── models/face_landmarker.task
├── scripts/                 # GPU별 실험 큐 스크립트 (run_*.sh)
├── results.csv              # 학습 중 best val Pearson (조도별)
├── results_roi.csv          # ROI 실험 결과
└── metrics_eval.csv         # evaluate.py 결과 (전체 지표)
```

체크포인트(`*.pt`)와 학습 로그는 용량 문제로 저장소에 포함하지 않았음.


---

## 실험 환경

| 항목 | 사양 |
|---|---|
| OS | Ubuntu 24.04 LTS |
| GPU | NVIDIA GeForce RTX 4090 (24GB) × 4 (병렬로 실험 큐 실행) |
| CUDA | CUDA 12.2 (PyTorch는 cu121 빌드) |
| Python | 3.11.15 (conda env `rppg`) |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| NumPy | 2.4.6 |
| OpenCV | opencv-contrib-python 5.0.0.93 |

```bash
conda create -n rppg python=3.11 -y && conda activate rppg
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 사용법

### 1. 데이터 경로 설정

`buaa_experiment/labels.csv` 가 준비되어 있어야 함.

BUAA-MIHR 데이터셋은 아래 구조여야 함.

```text
BUAA-MIHR/
├── Sub 01/
├── Sub 02/
└── ...
```

```bash
python set_data_path.py --data_root /path/to/BUAA-MIHR
```

### 2. 학습

`--model` 만 바꿔서 세 모델을 같은 조건으로 학습함.

```bash
python -u train.py \
  --model efficientphys_bigru \
  --window_size 150 --stride 75 --frame_depth 10 \
  --batch_size 2 --epochs 10 \
  --max_train_windows 120 --max_val_windows 80 \
  --val_max_lux 4.0 --seed 42 --device cuda
```

`--model` 은 `efficientphys` / `efficientphys_gru` / `efficientphys_bigru` 중 선택.

다른 split을 쓰려면 subject 파일을 지정함.

```bash
python -u train.py --model efficientphys_bigru ... \
  --split B \
  --train_subjects buaa_experiment/train_subjects_B.txt \
  --val_subjects   buaa_experiment/val_subjects_B.txt
```

여러 시드/split을 돌리는 예시는 `scripts/run_seed47to51_gpu0.sh` 등을 참고.

### 3. ROI 실험

```bash
python precompute_roi.py   # roi_boxes.json 생성
```

이후 `train.py` 의 ROI 옵션(`full` / `multi`)으로 학습함. 자세한 옵션은 `python train.py -h`.

---

## 실험 설정

| 항목 | 값 |
|---|---|
| window_size / stride | 150 / 75 |
| frame_depth | 10 |
| batch_size | 2 |
| epochs | 10 |
| max_train_windows / max_val_windows | 120 / 80 |
| val_max_lux | 4.0 (저조도만 검증) |
| loss | `1 - Pearson correlation` |

### Subject split

| Split | Val subjects |
|---|---|
| A (기본) | 07, 09 |
| B | 05, 11 |
| C | 03, 12 |

나머지 subject는 train으로 사용함. 정확한 목록은 `buaa_experiment/*_subjects*.txt` 를 참고.

---

## 출력

```text
train loss=...
VAL loss=... pearson=...
val lux loss: 1=... 1.6=... 2.5=... 4=...
FINAL MODEL: efficientphys_bigru
BEST VAL LOSS: ...
BEST VAL PEARSON: ...
```

체크포인트는 `checkpoint_buaa_<model>[...].pt` 로 저장됨.
