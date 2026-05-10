import pyarrow.parquet as pq

from utils.quality_logger import QualityMetricLogger


def test_quality_metric_logger_writes_frame_metrics_to_parquet(tmp_path):
    output_path = tmp_path / "quality_metrics.parquet"
    logger = QualityMetricLogger(enabled=True, output_path=output_path)

    logger.log_frame(
        frame_idx=3,
        q_norm=[0.1, 0.2, 0.3],
        q_debug={
            "coverage": 0.4,
            "uncertainty": 0.5,
            "residual": 0.6,
            "coverage_occupied_voxels": 7,
            "coverage_total_voxels": 64,
        },
    )
    logger.close()

    table = pq.read_table(output_path)
    row = table.to_pylist()[0]
    assert row["frame_idx"] == 3
    assert row["q_cov_norm"] == 0.1
    assert row["q_unc_norm"] == 0.2
    assert row["q_res_norm"] == 0.3
    assert row["q_cov"] == 0.4
    assert row["q_unc"] == 0.5
    assert row["q_res"] == 0.6
    assert row["coverage_occupied_voxels"] == 7
    assert row["coverage_total_voxels"] == 64
