from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_DASK_CLUSTER_CONFIG: dict[str, Any] = {
    "name": "local",
    "args": {
        "n_workers": 2,
        "threads_per_worker": 1,
        "memory_limit": "1GB",
        "dashboard_address": None,
    },
}


def dask_cluster_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the configured Dask cluster settings with local defaults."""
    configured = config.get("dask_cluster")
    if not configured:
        return {
            "name": DEFAULT_DASK_CLUSTER_CONFIG["name"],
            "args": dict(DEFAULT_DASK_CLUSTER_CONFIG["args"]),
        }

    cluster_config = dict(configured)
    cluster_config.setdefault("name", DEFAULT_DASK_CLUSTER_CONFIG["name"])
    if cluster_config["name"] == "local":
        args = dict(DEFAULT_DASK_CLUSTER_CONFIG["args"])
        args.update(cluster_config.get("args", {}) or {})
        cluster_config["args"] = args
    else:
        cluster_config["args"] = dict(cluster_config.get("args", {}) or {})
    return cluster_config


def create_dask_cluster(cluster_config: dict[str, Any], logs_dir: Path | None = None) -> Any:
    """Create a Dask cluster from local or SLURM configuration."""
    executor_name = str(cluster_config.get("name", "local")).lower()
    args = dict(cluster_config.get("args", {}) or {})

    LOGGER.info("Setting up Dask executor: %s", executor_name)

    if executor_name == "local":
        from dask.distributed import LocalCluster

        LOGGER.info("LocalCluster started with args=%s", args)
        return LocalCluster(**args)

    if executor_name == "slurm":
        try:
            from dask_jobqueue import SLURMCluster
        except ImportError as exc:
            msg = "dask-jobqueue is required for dask_cluster.name='slurm'."
            raise RuntimeError(msg) from exc

        instance_cfg = dict(args.get("instance", {}) or {})
        scale_cfg = dict(args.get("scale", {}) or {})

        if logs_dir:
            logs_dir.mkdir(parents=True, exist_ok=True)
            extra_directives = list(instance_cfg.get("job_extra_directives", []))
            extra_directives.extend(
                [
                    f"--output={logs_dir}/slurm-%j.out",
                    f"--error={logs_dir}/slurm-%j.err",
                ]
            )
            instance_cfg["job_extra_directives"] = extra_directives

        processes = int(instance_cfg.get("processes", 1))
        min_jobs = int(scale_cfg.get("minimum_jobs", 0))
        max_jobs = int(scale_cfg.get("maximum_jobs", 0) or 0)
        n_workers_init = min_jobs * processes

        LOGGER.info(
            "SLURM job template: cores=%s, processes=%s, memory=%s, queue=%s, account=%s",
            instance_cfg.get("cores"),
            processes,
            instance_cfg.get("memory"),
            instance_cfg.get("queue"),
            instance_cfg.get("account"),
        )
        LOGGER.info(
            "Initial SLURM submit: minimum_jobs=%d -> n_workers=%d",
            min_jobs,
            n_workers_init,
        )

        cluster = SLURMCluster(n_workers=n_workers_init, **instance_cfg)
        LOGGER.info("SLURMCluster started with instance args=%s", instance_cfg)

        if max_jobs > 0:
            cluster.adapt(minimum_jobs=min_jobs, maximum_jobs=max_jobs)
            LOGGER.info("Adaptive ceiling enabled: maximum_jobs=%d", max_jobs)

        return cluster

    LOGGER.warning("Unknown Dask executor '%s'. Falling back to a minimal LocalCluster.", executor_name)
    from dask.distributed import LocalCluster

    return LocalCluster(n_workers=1, threads_per_worker=1, memory_limit="1GB")


@contextmanager
def dask_client_context(cluster_config: dict[str, Any], logs_dir: Path | None = None) -> Iterator[Any]:
    """Create and close a Dask Client for the configured cluster."""
    from dask.distributed import Client

    cluster = create_dask_cluster(cluster_config, logs_dir=logs_dir)
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()
