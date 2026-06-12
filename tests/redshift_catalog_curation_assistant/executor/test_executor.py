from redshift_catalog_curation_assistant.executor import dask_cluster_config


def test_dask_cluster_config_defaults_to_small_local_cluster():
    """Ensure default Dask executor settings are suitable for local use."""
    config = dask_cluster_config({})

    assert config["name"] == "local"
    assert config["args"]["n_workers"] == 2
    assert config["args"]["threads_per_worker"] == 1
    assert config["args"]["memory_limit"] == "1GB"
    assert config["args"]["dashboard_address"] is None


def test_dask_cluster_config_merges_local_overrides():
    """Ensure local cluster overrides preserve unspecified defaults."""
    config = dask_cluster_config({"dask_cluster": {"name": "local", "args": {"n_workers": 4}}})

    assert config["name"] == "local"
    assert config["args"]["n_workers"] == 4
    assert config["args"]["threads_per_worker"] == 1
    assert config["args"]["memory_limit"] == "1GB"
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
