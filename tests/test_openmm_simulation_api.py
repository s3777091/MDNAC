from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


API_ROOT = Path(__file__).resolve().parents[1] / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


class TestOpenMMSimulationAPI(unittest.TestCase):
    def test_existing_predict_structure_route_still_accepts_json_body(self) -> None:
        from fastapi.testclient import TestClient
        from structure_predictor.server import create_app

        class DummyResult:
            def to_dict(self) -> dict[str, object]:
                return {
                    "job_id": "structure-job",
                    "structure_path": "structure.pdb",
                }

        class DummyRunner:
            def __init__(self, settings) -> None:
                self.settings = settings

            def predict(self, request) -> DummyResult:
                self.request = request
                return DummyResult()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_structure_config(root)
            with patch("structure_predictor.server.OpenFoldRunner", DummyRunner):
                client = TestClient(create_app(config_path=config_path, environment="local"))
                response = client.post(
                    "/predict-structure",
                    json={"sequence": "MPEPTIDE", "name": "candidate"},
                )

        self.assertEqual(200, response.status_code)
        self.assertEqual("structure-job", response.json()["job_id"])

    def test_create_local_simulation_job_enqueues_and_returns_immediately(self) -> None:
        from fastapi.testclient import TestClient
        from structure_predictor.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_structure_config(root)
            pdb_path = root / "structure.pdb"
            pdb_path.write_text("HEADER    TEST\nEND\n", encoding="utf-8")

            with patch("structure_predictor.server.enqueue_local_simulation_job") as enqueue:
                app = create_app(config_path=config_path, environment="local")
                client = TestClient(app)
                response = client.post(
                    "/predict/simulation",
                    json={
                        "structure_job_id": 102,
                        "pdb_path": str(pdb_path),
                        "run_target": "local",
                        "environment": {
                            "temperature_k": 300,
                            "ph": 7.4,
                            "salt_m": 0.15,
                            "steps": 50000,
                            "report_interval": 500,
                            "gpu_device": 0,
                        },
                    },
                )

            self.assertEqual(200, response.status_code)
            payload = response.json()
            self.assertEqual("queued", payload["status"])
            self.assertEqual("openmm_simulation", payload["task"])
            self.assertEqual("local", payload["run_target"])
            self.assertTrue(payload["job_id"])
            enqueue.assert_called_once()

            status_response = client.get(f"/predict/jobs/{payload['job_id']}")
            self.assertEqual(200, status_response.status_code)
            status = status_response.json()
            self.assertEqual("queued", status["status"])
            self.assertEqual({"current_step": 0, "total_steps": 50000}, status["progress"])

    def test_versioned_routes_are_available(self) -> None:
        from fastapi.testclient import TestClient
        from structure_predictor.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_structure_config(root)
            pdb_path = root / "structure.pdb"
            pdb_path.write_text("HEADER    TEST\nEND\n", encoding="utf-8")

            with patch("structure_predictor.server.enqueue_local_simulation_job"):
                client = TestClient(create_app(config_path=config_path, environment="local"))
                response = client.post(
                    "/api/v1/predict/simulation",
                    json={
                        "structure_job_id": "structure-102",
                        "pdb_path": str(pdb_path),
                        "run_target": "local",
                    },
                )

            self.assertEqual(200, response.status_code)
            job_id = response.json()["job_id"]
            self.assertEqual(200, client.get(f"/api/v1/predict/jobs/{job_id}").status_code)
            self.assertEqual(
                200,
                client.get(f"/api/v1/predict/jobs/{job_id}/result").status_code,
            )

    def test_cif_only_request_is_rejected(self) -> None:
        from fastapi.testclient import TestClient
        from structure_predictor.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_structure_config(root)
            client = TestClient(create_app(config_path=config_path, environment="local"))
            response = client.post(
                "/predict/simulation",
                json={
                    "structure_job_id": 102,
                    "cif_path": str(root / "structure.cif"),
                    "run_target": "local",
                },
            )

        self.assertEqual(400, response.status_code)
        self.assertEqual(
            "CIF simulation is not supported yet; please provide pdb_path.",
            response.json()["detail"],
        )

    def test_completed_result_response_uses_stored_job_result(self) -> None:
        from fastapi.testclient import TestClient
        from structure_predictor.server import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_structure_config(root)
            app = create_app(config_path=config_path, environment="local")
            store = app.state.simulation_jobs
            job_state = store.create_job(
                job_id="completed-job",
                payload={
                    "job_id": "completed-job",
                    "structure_job_id": "102",
                    "pdb_path": str(root / "structure.pdb"),
                    "run_target": "local",
                    "environment": {
                        "steps": 10,
                    },
                },
                run_target="local",
            )
            result = {
                "initial_pdb": "simulation_initial.pdb",
                "final_pdb": "simulation_final.pdb",
                "trajectory_dcd": "trajectory.dcd",
                "state_csv": "state.csv",
                "steps": 10,
                "temperature_k": 300,
                "ph": 7.4,
                "salt_m": 0.15,
                "gpu_device": 0,
            }
            store.update_job(job_state["job_id"], status="completed", result=result)

            response = TestClient(app).get("/predict/jobs/completed-job/result")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("completed", payload["status"])
        self.assertEqual(result, payload["result"])


def _write_structure_config(root: Path) -> Path:
    config_path = root / "config.structure.yaml"
    config_path.write_text(
        f"""
environment: local

environments:
  local:
    server:
      host: 127.0.0.1
      port: 8010
      reload: false
    runpod:
      enabled: false
    simulation:
      jobs_root: {root.as_posix()}/simulation_jobs
      rabbitmq_url: amqp://guest:guest@127.0.0.1:5672//
      queue_name: protein_simulation_queue
      worker_prefetch_multiplier: 1
      task_acks_late: true
      task_reject_on_worker_lost: true
      task_track_started: true
    openfold:
      repo_path: {root.as_posix()}/openfold
      output_root: {root.as_posix()}/structure_predictions
      template_mmcif_dir: {root.as_posix()}/mmcif
      openfold_checkpoint_path: {root.as_posix()}/seq_model_esm1b_ptm.pt
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


if __name__ == "__main__":
    unittest.main()
