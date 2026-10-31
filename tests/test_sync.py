"""sync_mlflow: um run feito num SQLite 'do Colab' chega inteiro num segundo MLflow (também SQLite)."""
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

import sync_mlflow


def test_sync_copies_run_once(tmp_path):
    colab_artifacts = tmp_path / "colab" / "artifacts"
    src = MlflowClient(tracking_uri=f"sqlite:///{(tmp_path / 'colab' / 'mlflow.db').as_posix()}")
    exp_id = src.create_experiment("clef-tickets-priority",
                                   artifact_location=(colab_artifacts / "clef-tickets-priority").as_posix())
    run = src.create_run(exp_id, run_name="clef-flash-lora-teste")
    run_id = run.info.run_id
    src.log_batch(run_id, params=[Param("lora_r", "32")],
                  metrics=[Metric("val/accuracy", 0.5 + step / 10, 1000 + step, step) for step in (1, 2)])
    report = tmp_path / "train_report.json"
    report.write_text('{"ok": true}')
    src.log_artifact(run_id, str(report))
    src.set_terminated(run_id, "FINISHED")

    target_uri = f"sqlite:///{(tmp_path / 'local' / 'mlflow.db').as_posix()}"
    dst = MlflowClient(tracking_uri=target_uri)
    dst.create_experiment("clef-tickets-priority", artifact_location=(tmp_path / "local" / "artifacts").as_posix())

    imported = sync_mlflow.sync(tmp_path / "colab" / "mlflow.db", colab_artifacts, target_uri,
                                colab_root=colab_artifacts.as_posix())
    assert [old for old, _ in imported] == [run_id]
    new = dst.get_run(imported[0][1])
    assert new.info.run_name == "clef-flash-lora-teste"
    assert new.info.status == "FINISHED"
    assert new.data.params["lora_r"] == "32"
    assert [(m.step, m.value) for m in dst.get_metric_history(new.info.run_id, "val/accuracy")] == [(1, 0.6), (2, 0.7)]
    assert [a.path for a in dst.list_artifacts(new.info.run_id)] == ["train_report.json"]

    assert sync_mlflow.sync(tmp_path / "colab" / "mlflow.db", colab_artifacts, target_uri,
                            colab_root=colab_artifacts.as_posix()) == []
