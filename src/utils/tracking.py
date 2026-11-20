from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Experiment tracking abstraction supporting local MLflow and JSON fallbacks."""

    def __init__(self, experiment_name: str = "room2rug-multimodal", tracking_uri: str | None = None) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self._mlflow = None
        self._active_run = None
        self._local_run_data: dict[str, Any] = {"params": {}, "metrics": {}, "tags": {}}

        try:
            import mlflow

            self._mlflow = mlflow
            self._mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow.set_experiment(self.experiment_name)
        except ImportError:
            logger.info("MLflow not installed; tracker falling back to local JSON logger.")

    def start_run(self, run_name: str | None = None) -> None:
        self._local_run_data = {"run_name": run_name, "params": {}, "metrics": {}, "tags": {}}
        if self._mlflow is not None:
            self._active_run = self._mlflow.start_run(run_name=run_name)

    def log_param(self, key: str, value: Any) -> None:
        self._local_run_data["params"][key] = value
        if self._mlflow is not None and self._active_run is not None:
            self._mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._local_run_data["metrics"][key] = float(value)
        if self._mlflow is not None and self._active_run is not None:
            self._mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        if self._mlflow is not None and self._active_run is not None:
            self._mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def end_run(self, save_local_path: str | Path | None = None) -> dict[str, Any]:
        if save_local_path is not None:
            path = Path(save_local_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._local_run_data, f, indent=2)
        if self._mlflow is not None and self._active_run is not None:
            self._mlflow.end_run()
            self._active_run = None
        return self._local_run_data
