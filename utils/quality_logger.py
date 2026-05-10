from pathlib import Path


class QualityMetricLogger:
    def __init__(self, enabled=False, output_path=None):
        self.enabled = bool(enabled)
        self.output_path = Path(output_path) if output_path else None
        self.rows = []

    def log_frame(self, frame_idx, q_norm, q_debug):
        if not self.enabled:
            return
        q_values = [float(x) for x in q_norm]
        self.rows.append(
            {
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
        )

    def close(self):
        if not self.enabled or self.output_path is None:
            return
        if not self.rows:
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
