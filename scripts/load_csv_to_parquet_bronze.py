#!/usr/bin/env python3
"""
Convert local CSV into partitioned Parquet data for the HDFS Bronze layer.

The script writes files locally to a temporary directory and then copies partition directories
into the HDFS Bronze path using the namenode Docker container.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def partition_csv_to_parquet(csv_path: Path, output_dir: Path) -> None:
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["TransactionDate"], infer_datetime_format=True)
    if df.empty:
        raise RuntimeError("CSV file contains no rows")

    df["transactionDate"] = df["TransactionDate"].dt.date.astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing partitioned Parquet files to: {output_dir}")
    for date_value, partition in df.groupby("transactionDate"):
        partition_dir = output_dir / f"transactionDate={date_value}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_path = partition_dir / "data.parquet"
        partition.drop(columns=["transactionDate"]).to_parquet(file_path, index=False)
        print(f"  - Wrote partition {date_value}: {file_path}")

    print("Partitioned Parquet generation complete.")


def copy_to_hdfs(container_name: str, local_dir: Path, hdfs_target: str) -> None:
    print(f"Copying local Parquet partitions to container {container_name}")
    subprocess.run(["docker", "cp", str(local_dir), f"{container_name}:/tmp/bronze_parquet"], check=True)

    hdfs_cmd_base = "/opt/hadoop-3.2.1/bin/hdfs dfs"
    mkdir_cmd = f"{hdfs_cmd_base} -mkdir -p {hdfs_target}"
    put_cmd = f"{hdfs_cmd_base} -put -f /tmp/bronze_parquet/* {hdfs_target}/"

    print(f"Creating HDFS target path: {hdfs_target}")
    subprocess.run(["docker", "exec", container_name, "sh", "-lc", mkdir_cmd], check=True)
    print(f"Uploading partitions to HDFS: {hdfs_target}")
    subprocess.run(["docker", "exec", container_name, "sh", "-lc", put_cmd], check=True)
    print("Upload to HDFS Bronze complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CSV into HDFS Bronze as partitioned Parquet.")
    parser.add_argument("--csv", type=Path, default=Path("bank_transactions_data_2.csv"), help="Local CSV path")
    parser.add_argument("--local-output", type=Path, default=Path("./tmp/bronze_parquet"), help="Local parquet output directory")
    parser.add_argument("--hdfs-target", type=str, default="/data/bronze/transactions", help="HDFS Bronze target directory")
    parser.add_argument("--container", type=str, default="fraud_detection-namenode", help="Namenode Docker container name")
    parser.add_argument("--no-hdfs", action="store_true", help="Only write local parquet partitions, do not copy to HDFS")
    args = parser.parse_args()

    if args.local_output.exists():
        shutil.rmtree(args.local_output)

    partition_csv_to_parquet(args.csv, args.local_output)
    if not args.no_hdfs:
        copy_to_hdfs(args.container, args.local_output, args.hdfs_target)


if __name__ == "__main__":
    main()
