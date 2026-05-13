import csv
import math
from pathlib import Path


class QualityMetricLogger:
    def __init__(self, enabled=False, output_path=None, flush_every_frame=True):
        self.enabled = bool(enabled)
        self.output_path = Path(output_path) if output_path else None
        self.flush_every_frame = bool(flush_every_frame)
        self.rows = []
        self._csv_file = None
        self._csv_writer = None
        self._fieldnames = [
            "frame_idx",
            "q_cov_norm",
            "q_unc_norm",
            "q_res_norm",
            "q_cov",
            "q_unc",
            "q_res",
            "coverage_occupied_voxels",
            "coverage_total_voxels",
        ]

    def log_frame(self, frame_idx, q_norm, q_debug):
        if not self.enabled:
            return
        q_values = [float(x) for x in q_norm]
        row = {
            "frame_idx": int(frame_idx),
            "q_cov_norm": q_values[0],
            "q_unc_norm": q_values[1],
            "q_res_norm": q_values[2],
            "q_cov": float(q_debug["coverage"]),
            "q_unc": float(q_debug["uncertainty"]),
            "q_res": float(q_debug["residual"]),
            "coverage_occupied_voxels": int(q_debug["coverage_occupied_voxels"]),
            "coverage_total_voxels": int(q_debug["coverage_total_voxels"]),
        }
        self.rows.append(row)
        if self.flush_every_frame:
            self._write_csv_row(row)

    def _csv_output_path(self):
        if self.output_path is None:
            return None
        if self.output_path.suffix.lower() == ".csv":
            return self.output_path
        return self.output_path.with_suffix(".csv")

    def _write_csv_row(self, row):
        output_path = self._csv_output_path()
        if output_path is None:
            return
        if self._csv_writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = output_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._fieldnames
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def summary(self):
        if not self.rows:
            return {}

        def finite_values(key):
            return [
                float(row[key])
                for row in self.rows
                if math.isfinite(float(row[key]))
            ]

        metrics = {
            "num_quality_frames": len(self.rows),
            "first_frame_idx": int(self.rows[0]["frame_idx"]),
            "last_frame_idx": int(self.rows[-1]["frame_idx"]),
        }
        for key in [
            "q_cov_norm",
            "q_unc_norm",
            "q_res_norm",
            "q_cov",
            "q_unc",
            "q_res",
        ]:
            values = finite_values(key)
            if not values:
                metrics[f"{key}_mean"] = float("nan")
                metrics[f"{key}_min"] = float("nan")
                metrics[f"{key}_max"] = float("nan")
                continue
            metrics[f"{key}_mean"] = sum(values) / len(values)
            metrics[f"{key}_min"] = min(values)
            metrics[f"{key}_max"] = max(values)
        return metrics

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        if not self.enabled or self.output_path is None:
            return
        if not self.rows:
            return
        if self.output_path.suffix.lower() == ".csv":
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required to write quality metric parquet files"
            ) from exc

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self.rows)
        pq.write_table(table, self.output_path)
