from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.test_utils.launcher_common_tests import (
    BatchedSweeperTestSuite,
    IntegrationTestSuite,
    LauncherTestSuite,
)
from hydra.test_utils.test_utils import TSweepRunner
from hydra_plugins.experiment_sweeper_plugin.experiment_sweeper import ExperimentSweeper
from pytest import mark


def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking at
    # the Sweeper plugins
    plugin_names = [x.__name__ for x in Plugins.instance().discover(Sweeper)]
    assert ExperimentSweeper.__name__ in plugin_names


def test_cmd_line_overrides_launch_jobs(hydra_sweep_runner: TSweepRunner) -> None:
    sweep = hydra_sweep_runner(
        calling_file=None,
        calling_module="hydra.test_utils.a_module",
        config_path="configs",
        config_name="compose.yaml",
        task_function=None,
        overrides=["hydra/sweeper=experiment", "hydra/launcher=basic", "foo=1,2"],
    )
    with sweep:
        assert sweep.returns is not None
        job_ret = sweep.returns[0]
        assert len(job_ret) == 2
        assert job_ret[0].overrides == ["foo=1"]
        assert job_ret[0].cfg == {"foo": 1, "bar": 100}
        assert job_ret[1].overrides == ["foo=2"]
        assert job_ret[1].cfg == {"foo": 2, "bar": 100}


def test_configured_overrides_launch_jobs(hydra_sweep_runner: TSweepRunner) -> None:
    sweep = hydra_sweep_runner(
        calling_file=__file__,
        calling_module=None,
        config_path="configs",
        config_name="overrides.yaml",
        task_function=None,
        overrides=None,
    )
    with sweep:
        job_ret = sweep.returns[0]
        all_cfgs = set([frozenset(j.cfg.items()) for j in job_ret])

        expected_cfgs = [
            {"foo": "a", "bar": 1},
            {"foo": 3, "bar": 1},
            {"foo": True, "bar": 1},
            {"foo": "a", "bar": True},
            {"foo": 3, "bar": True},
            {"foo": True, "bar": True},
        ]
        assert all_cfgs == set([frozenset(cfg.items()) for cfg in expected_cfgs])


def test_cmd_line_overrides_overwrite_configured_overrides(
    hydra_sweep_runner: TSweepRunner,
) -> None:
    sweep = hydra_sweep_runner(
        calling_file=__file__,
        calling_module=None,
        config_path="configs",
        config_name="overrides.yaml",
        task_function=None,
        overrides=["foo=False"],
    )
    with sweep:
        job_ret = sweep.returns[0]
        assert len(job_ret) == 2
        all_cfgs = set([frozenset(j.cfg.items()) for j in job_ret])

        expected_cfgs = [
            {"foo": False, "bar": 1},
            {"foo": False, "bar": True},
        ]
        assert all_cfgs == set([frozenset(cfg.items()) for cfg in expected_cfgs])


def test_accepts_ranges_in_config(
    hydra_sweep_runner: TSweepRunner,
) -> None:
    sweep = hydra_sweep_runner(
        calling_file=__file__,
        calling_module=None,
        config_path="configs",
        config_name="ranges.yaml",
        task_function=None,
        overrides=None,
    )
    with sweep:
        job_ret = sweep.returns[0]
        assert len(job_ret) == 3
        all_cfgs = set([frozenset(j.cfg.items()) for j in job_ret])

        expected_cfgs = [{"foo": 1.5}, {"foo": 2.0}, {"foo": 2.5}]
        assert all_cfgs == set([frozenset(cfg.items()) for cfg in expected_cfgs])


# Run launcher test suite with the basic launcher and this sweeper
@mark.parametrize(
    ("launcher_name", "overrides"), [("basic", ["hydra/sweeper=experiment"])]
)
class TestExperimentSweeper(LauncherTestSuite):
    pass


# Many sweepers are batching jobs in groups.
# This test suite verifies that the spawned jobs are not overstepping the directories of one another.
@mark.parametrize(
    ("launcher_name", "overrides"),
    # This will cause the sweeper to split batches to at most 2 jobs each, which is what
    # the tests in BatchedSweeperTestSuite are expecting.
    [("basic", ["hydra/sweeper=experiment", "hydra.sweeper.max_batch_size=2"])],
)
class TestExperimentSweeperWithBatching(BatchedSweeperTestSuite):
    pass


# Run integration test suite with the basic launcher and this sweeper
@mark.parametrize(
    ("task_launcher_cfg", "extra_flags"),
    [({}, ["-m", "hydra/sweeper=experiment", "hydra/launcher=basic"])],
)
class TestExperimentSweeperIntegration(IntegrationTestSuite):
    pass
