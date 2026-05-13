import math

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


def test_quality_metric_logger_flushes_each_frame_to_csv(tmp_path):
    output_path = tmp_path / "quality_metrics.csv"
    logger = QualityMetricLogger(enabled=True, output_path=output_path)

    logger.log_frame(
        frame_idx=4,
        q_norm=[0.2, 0.3, 0.4],
        q_debug={
            "coverage": 0.5,
            "uncertainty": 0.6,
            "residual": 0.7,
            "coverage_occupied_voxels": 8,
            "coverage_total_voxels": 64,
        },
    )

    assert output_path.exists()
    assert "frame_idx,q_cov_norm,q_unc_norm" in output_path.read_text()
    logger.close()


def test_quality_metric_logger_summary_aggregates_rows(tmp_path):
    logger = QualityMetricLogger(enabled=True, output_path=tmp_path / "quality.csv")
    for frame_idx, value in enumerate([0.1, 0.3]):
        logger.log_frame(
            frame_idx=frame_idx,
            q_norm=[value, 0.2, 0.4],
            q_debug={
                "coverage": value,
                "uncertainty": 0.2,
                "residual": 0.4,
                "coverage_occupied_voxels": 8,
                "coverage_total_voxels": 64,
            },
        )

    summary = logger.summary()

    assert summary["num_quality_frames"] == 2
    assert summary["first_frame_idx"] == 0
    assert summary["last_frame_idx"] == 1
    assert math.isclose(summary["q_cov_norm_mean"], 0.2)
    logger.close()
