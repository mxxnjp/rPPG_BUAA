import csv
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    required=True,
    help="BUAA-MIHR dataset root"
)
parser.add_argument(
    "--labels",
    default="buaa_experiment/labels.csv"
)
args = parser.parse_args()


rows = []

with open(args.labels, "r", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        old_path = row["video_path"]

        # 기존 절대경로에서 BUAA-MIHR 이후의 상대경로만 추출
        marker = "BUAA-MIHR"

        if marker not in old_path:
            raise RuntimeError(
                f"BUAA-MIHR가 경로에 없습니다: {old_path}"
            )

        relative_path = old_path.split(marker, 1)[1].lstrip("/")

        row["video_path"] = os.path.join(
            os.path.abspath(args.data_root),
            relative_path
        )

        rows.append(row)


with open(args.labels, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


print("labels.csv 경로 변경 완료")
print("BUAA root:", os.path.abspath(args.data_root))
print("example:", rows[0]["video_path"])

