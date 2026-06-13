import pytest

from redshift_catalog_curation_assistant.executor import (
    DaskClusterConfigError,
    create_dask_cluster,
    dask_cluster_config,
)


def test_dask_cluster_config_defaults_to_small_local_cluster():
    """Ensure default Dask executor settings are suitable for local use."""
    config = dask_cluster_config({})

    assert config["name"] == "local"
    assert config["args"]["n_workers"] == 3
    assert config["args"]["threads_per_worker"] == 1
    assert config["args"]["memory_limit"] == "2GB"
    assert config["args"]["dashboard_address"] is None


def test_dask_cluster_config_merges_local_overrides():
    """Ensure local cluster overrides preserve unspecified defaults."""
    config = dask_cluster_config({"dask_cluster": {"name": "local", "args": {"n_workers": 4}}})

    assert config["name"] == "local"
    assert config["args"]["n_workers"] == 4
    assert config["args"]["threads_per_worker"] == 1
    assert config["args"]["memory_limit"] == "2GB"
    assert config["args"]["dashboard_address"] is None


def test_dask_cluster_config_preserves_slurm_sections():
    """Ensure SLURM cluster sections pass through for advanced users."""
    cluster = {
        "name": "slurm",
        "logs_dir": "logs",
        "args": {
            "instance": {
                "cores": 4,
                "processes": 2,
                "memory": "8GB",
                "queue": "debug",
            },
            "scale": {
                "minimum_jobs": 1,
                "maximum_jobs": 3,
            },
        },
    }

    config = dask_cluster_config({"dask_cluster": cluster})

    assert config == cluster


def test_dask_cluster_config_rejects_incomplete_slurm_config():
    """Ensure SLURM configs fail before creating an unusable cluster."""
    with pytest.raises(DaskClusterConfigError, match="args.instance.*args.scale.minimum_jobs"):
        dask_cluster_config({"dask_cluster": {"name": "slurm"}})


def test_dask_cluster_config_rejects_unknown_executor():
    """Ensure typos in executor names fail instead of silently using local Dask."""
    with pytest.raises(DaskClusterConfigError, match="Unsupported Dask executor.*slrum"):
        dask_cluster_config({"dask_cluster": {"name": "slrum"}})


def test_create_dask_cluster_rejects_unknown_executor():
    """Ensure direct cluster creation also rejects unknown executor names."""
    with pytest.raises(DaskClusterConfigError, match="Unsupported Dask executor.*unknown"):
        create_dask_cluster({"name": "unknown"})


def test_dask_cluster_config_rejects_slurm_without_initial_jobs():
    """Ensure SLURM configs require at least one initial job."""
    cluster = {
        "name": "slurm",
        "args": {
            "instance": {
                "cores": 4,
                "memory": "8GB",
            },
            "scale": {
                "minimum_jobs": 0,
            },
        },
    }

    with pytest.raises(DaskClusterConfigError, match="args.scale.minimum_jobs"):
        dask_cluster_config({"dask_cluster": cluster})
