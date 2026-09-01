import json
import math
import re
import tempfile
import unittest
from pathlib import Path

from scripts.run_report_dreamer_ab_campaign import aggregate, paired_deltas
from scripts.simlingo_dashboard import (
    HTML,
    parse_bench2drive_result,
    parse_dreamer_log,
    parse_report_trace_metrics,
    report_dreamer_pipeline_payload,
)
from scripts.finalize_report_native_trace import validate_trace_result_binding
from scripts.summarize_report_dreamer_run import bench2drive_summary


class ReportDashboardIntegrationTests(unittest.TestCase):
    def _log(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "run.log"
        path.write_text(text, encoding="utf-8")
        return path

    def test_dashboard_exposes_only_requested_dreamer_modes(self):
        select = re.search(
            r'<select id="dreamer_mode">(.*?)</select>', HTML, re.DOTALL
        )
        self.assertIsNotNone(select)
        self.assertEqual(
            re.findall(r'<option value="([^"]+)"', select.group(1)),
            ["off", "dreamer_ppo", "report_rssm_learned"],
        )
        self.assertNotIn("CarDreamer blend", HTML)
        self.assertIn(
            'new Set(["simlingo", "dreamer_ppo", "report_rssm_learned"])',
            HTML,
        )

    def test_dashboard_persists_the_effective_experiment_seed(self):
        self.assertIn('id="new_seed"', HTML)
        self.assertIn('function setExperimentSeed(value)', HTML)
        self.assertIn('setExperimentSeed(data.seed)', HTML)
        self.assertIn('localStorage.setItem("simlingoExperimentSeed", seed)', HTML)

    def test_report_runtime_records_exact_collision_events(self):
        source = Path(__file__).resolve().parents[1] / "scripts" / "simlingo_dashboard.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn(
            'report_collision_events = report_run_dir / "collision_events.jsonl"',
            text,
        )
        self.assertIn(
            '"SIMLINGO_COLLISION_EVENT_PATH": str(report_collision_events)',
            text,
        )

    def test_pipeline_status_uses_real_collection_audit_and_training_artifacts(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        matrix_id = "native_report12_v1"
        matrix = root / "data" / "report_dreamer" / "native" / "matrices" / matrix_id
        matrix.mkdir(parents=True)
        (matrix / "status.env").write_text(
            "state=complete\naccepted=12\ntotal=12\n",
            encoding="utf-8",
        )
        audit_dir = root / "checkpoints" / "report_aligned_dreamer" / ("audit_" + matrix_id)
        audit_dir.mkdir(parents=True)
        (audit_dir / "dataset_audit.json").write_text(
            json.dumps({"error": None, "accepted": 12, "rejected": 1}),
            encoding="utf-8",
        )
        split = {
            "episodes": 2,
            "transitions": 100,
            "collisions": 1,
            "towns": ["Town12"],
            "scenarios": ["Accident"],
            "policy_sources": ["simlingo_native"],
        }
        manifest = {
            "train": dict(split, episodes=8, transitions=600),
            "validation": dict(split, transitions=200),
            "test": dict(split),
            "seed_sets": {
                "train": ["1", "2"],
                "validation": ["3", "4"],
                "test": ["5", "6"],
            },
            "config": {
                "training": {"world_model_epochs": 30, "policy_epochs": 30}
            },
        }
        (audit_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        log = root / "logs" / "report_aligned_dreamer" / (matrix_id + "_training.log")
        log.parent.mkdir(parents=True)
        log.write_text(
            "[world-model] epoch=30 train=1.0 val=1.1\n"
            "[policy] epoch=4 actor=1.0 critic=1.0 val_objective=0.2\n",
            encoding="utf-8",
        )

        payload = report_dreamer_pipeline_payload(root, check_service=False)

        self.assertEqual(payload["current_phase"], "Offline training")
        self.assertEqual(payload["dataset"]["accepted_episodes"], 12)
        self.assertEqual(payload["dataset"]["transitions"], 900)
        self.assertTrue(payload["dataset"]["native_only"])
        self.assertEqual(payload["training"]["state"], "active")
        self.assertEqual(payload["training"]["policy_epoch"], 4)
        self.assertFalse(payload["checkpoints"]["candidate"]["available"])

    def test_report_d_log_is_attributed_and_parsed(self):
        path = self._log(
            "SIMLINGO_REPORT_DREAMER enabled: independent report-aligned "
            "ablation=D shadow=0 checkpoint=/tmp/candidate.pt\n"
            "SIMLINGO_REPORT_DREAMER_PROFILE step=40 ablation=D shadow=0 "
            "candidate=3 kind=actor alpha=0.250 applied=1 "
            "risk=0.700->0.200 progress=0.1000->0.1500 "
            "front=8.0 ttc=4.5 latency_ms=3.20\n"
        )
        info = parse_dreamer_log(path)
        self.assertEqual(info["group"], "report_rssm_learned")
        self.assertEqual(info["guard_rows"], 1)
        self.assertEqual(info["applied"], 1)
        self.assertEqual(info["candidate_ids"], {"3"})
        self.assertEqual(info["kinds"], {"actor"})
        self.assertAlmostEqual(info["risk_deltas"][0], 0.5)
        self.assertAlmostEqual(info["progress_deltas"][0], 0.05)
        self.assertAlmostEqual(info["min_ttc"], 4.5)
        self.assertEqual(info["latencies_ms"], [3.2])

    def test_report_log_exposes_exact_trace_path(self):
        path = self._log(
            "SIMLINGO_REPORT_DREAMER enabled: independent report-aligned "
            "ablation=D shadow=0 checkpoint=/tmp/candidate.pt "
            "trace=/tmp/report-runtime/trace.jsonl\n"
        )
        info = parse_dreamer_log(path)
        self.assertEqual(
            info["report_trace_path"],
            Path("/tmp/report-runtime/trace.jsonl"),
        )

    def test_report_shadow_is_separate_from_closed_loop_d(self):
        path = self._log(
            "SIMLINGO_REPORT_DREAMER enabled: independent report-aligned "
            "ablation=D shadow=1 checkpoint=/tmp/candidate.pt\n"
            "SIMLINGO_REPORT_DREAMER_PROFILE step=80 ablation=D shadow=1 "
            "candidate=4 kind=actor alpha=0.000 applied=0 "
            "risk=0.600->0.300 progress=0.1000->0.1200 "
            "front=12.0 ttc=99.0 latency_ms=2.10\n"
        )
        info = parse_dreamer_log(path)
        self.assertEqual(info["group"], "report_rssm_shadow")
        self.assertEqual(info["applied"], 0)
        self.assertIsNone(info["min_ttc"])

    def test_campaign_reports_dispersion_and_paired_deltas(self):
        rows = []
        for seed, native_score, candidate_score in (
            ("1", 70.0, 75.0),
            ("2", 80.0, 82.0),
        ):
            common = {
                "route": "57",
                "seed": seed,
                "weather": "day",
                "eligible": True,
                "route_completion": 100.0,
                "offroad_infractions": 0.0,
                "success": True,
            }
            rows.append(
                dict(common, condition="A", driving_score=native_score, collisions=1.0)
            )
            rows.append(
                dict(common, condition="D", driving_score=candidate_score, collisions=0.0)
            )
        summary = aggregate(rows, "D")
        self.assertEqual(summary["runs"], 2)
        self.assertAlmostEqual(summary["driving_score_mean"], 78.5)
        self.assertGreater(summary["driving_score_std"], 0.0)
        self.assertEqual(summary["collisions_total"], 0.0)
        deltas = paired_deltas(rows, "D")
        self.assertEqual(deltas["paired_runs"], 2)
        self.assertEqual(deltas["driving_score"]["values"], [5.0, 2.0])
        self.assertTrue(math.isfinite(deltas["driving_score"]["std"]))

    def test_missing_score_is_not_silently_replaced_by_zero(self):
        rows = [
            {
                "condition": "D",
                "eligible": True,
                "driving_score": None,
                "route_completion": 50.0,
                "collisions": 0.0,
                "offroad_infractions": 0.0,
                "success": False,
            }
        ]
        summary = aggregate(rows, "D")
        self.assertEqual(summary["driving_score_count"], 0)
        self.assertIsNone(summary["driving_score_mean"])
        self.assertIsNone(summary["driving_score_std"])

    def test_incomplete_bench2drive_result_has_null_metrics(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "result.json"
        path.write_text(
            '{"_checkpoint":{"progress":[0,1],"records":[],"global_record":'
            '{"status":"Started"}},"entry_status":"Started","eligible":true}',
            encoding="utf-8",
        )
        summary = bench2drive_summary(path)
        self.assertFalse(summary["complete_result"])
        self.assertIsNone(summary["collisions"])
        self.assertIsNone(summary["offroad_infractions"])
        self.assertIsNone(summary["driving_score"])

    def test_dashboard_rejects_incomplete_result_even_with_scores(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "results_route_57_seed_101.json"
        path.write_text(
            json.dumps(
                {
                    "entry_status": "Started",
                    "eligible": True,
                    "_checkpoint": {
                        "progress": [0, 1],
                        "records": [
                            {
                                "status": "Started",
                                "scores": {
                                    "score_route": 100.0,
                                    "score_composed": 100.0,
                                },
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        self.assertIsNone(parse_bench2drive_result(path))

    def test_dashboard_accepts_only_finished_completed_result(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "results_route_57_seed_101.json"
        path.write_text(
            json.dumps(
                {
                    "entry_status": "Finished",
                    "eligible": True,
                    "_checkpoint": {
                        "progress": [1, 1],
                        "records": [
                            {
                                "status": "Completed",
                                "scores": {
                                    "score_route": 100.0,
                                    "score_composed": 92.0,
                                    "score_penalty": 0.92,
                                },
                                "infractions": {
                                    "collisions_pedestrian": [],
                                    "collisions_vehicle": [],
                                    "collisions_layout": [],
                                    "red_light": [],
                                    "stop_infraction": [],
                                    "outside_route_lanes": [],
                                    "vehicle_blocked": [],
                                    "scenario_timeouts": [],
                                    "route_timeout": [],
                                    "min_speed_infractions": [],
                                },
                                "meta": {"route_length": 1000.0},
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        result = parse_bench2drive_result(path)
        self.assertIsNotNone(result)
        self.assertEqual(result["route_score"], 100.0)
        self.assertEqual(result["driving_score"], 92.0)

    def test_dashboard_does_not_turn_missing_finished_metrics_into_zero(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "results_route_57_seed_102.json"
        path.write_text(
            json.dumps(
                {
                    "entry_status": "Finished",
                    "eligible": True,
                    "_checkpoint": {
                        "progress": [1, 1],
                        "records": [
                            {
                                "status": "Completed",
                                "scores": {"score_route": 100.0},
                                "infractions": {},
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        self.assertIsNone(parse_bench2drive_result(path))

    def test_report_trace_metrics_use_candidate_rollouts_without_fake_dqi(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.jsonl"
        rows = [
            {
                "candidate_features": [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                "candidate_utilities": [0.0, 0.0],
                "selected_index": 1,
                "applied": True,
                "alpha": 0.2,
                "native_predicted_risk": 0.7,
                "selected_predicted_risk": 0.3,
                "native_predicted_progress": 0.1,
                "selected_predicted_progress": 0.2,
                "inference_latency_ms": 4.0,
            },
            {
                "candidate_features": [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                "candidate_utilities": [1.0, 0.0],
                "selected_index": 0,
                "applied": False,
                "alpha": 0.0,
                "native_predicted_risk": 0.4,
                "selected_predicted_risk": 0.4,
                "native_predicted_progress": 0.2,
                "selected_predicted_progress": 0.2,
                "inference_latency_ms": 6.0,
            },
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        metrics = parse_report_trace_metrics(path)
        self.assertEqual(metrics["ticks"], 2)
        self.assertEqual(metrics["mean_candidates_per_decision"], 2.0)
        self.assertEqual(metrics["proposal_rate"], 0.5)
        self.assertEqual(metrics["intervention_rate"], 0.5)
        self.assertAlmostEqual(metrics["alpha_mean"], 0.1)
        self.assertAlmostEqual(metrics["predicted_risk_gain"], 0.2)
        self.assertIn("dreaming_quality_index", metrics["unavailable_metrics"])

    def test_native_trace_is_bound_to_its_fresh_result(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        trace = root / "trace.jsonl"
        result = root / "result.json"
        trace.write_text(
            json.dumps(
                {"result_path": str(result), "route_id": "57", "seed": "101"}
            )
            + "\n",
            encoding="utf-8",
        )
        result.write_text("{}", encoding="utf-8")
        metadata = validate_trace_result_binding(trace, result.resolve())
        self.assertEqual(metadata["route_id"], "57")
        other = root / "other.json"
        other.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            validate_trace_result_binding(trace, other.resolve())


if __name__ == "__main__":
    unittest.main()
