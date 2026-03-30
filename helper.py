"""
Helper module for VQA Challenge Runner.
Provides Solver base class and run() entry point.
"""
import csv
import json
import os
import time
from abc import ABC, abstractmethod


class Solver(ABC):
    """Abstract base class for VQA solvers."""

    @abstractmethod
    def solve(self, image_path: str, prompt: str) -> int:
        """
        Given an image and a prompt, return:
            1  = Yes
            0  = No
           -1  = Failed
        """
        ...

    @abstractmethod
    def model_info(self) -> dict:
        """Return model name and generation parameters."""
        ...


def run(solver: Solver, input_csv: str, output_txt: str, output_json: str):
    """
    Run the solver on every row in input_csv,
    write predictions to output_txt and model info to output_json.
    """
    # Read CSV
    rows = []
    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} images from {input_csv}")

    # Resolve image paths relative to the CSV directory
    csv_dir = os.path.dirname(os.path.abspath(input_csv))

    # Run solver
    results = []
    total = len(rows)
    for i, row in enumerate(rows):
        idx = int(row[0])
        img_path = row[1]
        prompt = row[2]

        full_img_path = os.path.join(csv_dir, img_path)
        if not os.path.exists(full_img_path):
            full_img_path = img_path  # fallback to as-is

        try:
            answer = solver.solve(full_img_path, prompt)
        except Exception as e:
            print(f"  [ERR] #{idx}: {e}")
            answer = -1

        results.append((idx, answer))

        if (i + 1) % 30 == 0 or (i + 1) == total:
            print(f"  Progress: {i + 1}/{total}")

    # Write predictions
    with open(output_txt, "w") as f:
        for idx, ans in results:
            f.write(f"{idx} {ans}\n")
    print(f"Predictions written to {output_txt}")

    # Write model info
    info = solver.model_info()
    with open(output_json, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Model info written to {output_json}")
