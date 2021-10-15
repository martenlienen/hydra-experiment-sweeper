import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from more_itertools import chunked
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@dataclass
class SweeperConfig:
    _target_: str = (
        "hydra_plugins.experiment_sweeper_plugin.experiment_sweeper.ExperimentSweeper"
    )
    max_batch_size: Optional[int] = None


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="experiment",
    node=SweeperConfig,
    provider="hydra-experiment-sweeper",
)


class ExperimentSweeper(Sweeper):
    """A hydra sweeper with configurable overrides for reproducible experiments."""

    def __init__(self, max_batch_size: Optional[int]):
        super().__init__()

        self.max_batch_size = max_batch_size

        self.config: Optional[DictConfig] = None
        self.launcher: Optional[Launcher] = None
        self.hydra_context: Optional[HydraContext] = None

    def __repr__(self):
        return f"ExperimentSweeper(max_batch_size={self.max_batch_size!r})"

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.hydra_context = hydra_context
        self.config = config
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: List[str]) -> List[Sequence[JobReturn]]:
        assert self.config is not None
        assert self.launcher is not None
        log.info(f"{self!s} sweeping")
        log.info(f"Sweep output dir : {self.config.hydra.sweep.dir}")

        self.save_sweep_config()

        jobs = self.generate_jobs(arguments)

        # Validate all jobs once in the beginning to avoid failing halfway through
        self.validate_batch_is_legal(jobs)

        returns = []
        initial_job_idx = 0
        for batch in chunked(jobs, self.batch_size(jobs)):
            results = self.launcher.launch(batch, initial_job_idx=initial_job_idx)
            initial_job_idx += len(batch)
            returns.append(results)
        return returns

    def save_sweep_config(self):
        # Save sweep run config in top level sweep working directory
        sweep_dir = Path(self.config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")

    def generate_jobs(self, arguments):
        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        lists = []
        for override in parsed:
            if override.is_sweep_override():
                sweep_choices = override.sweep_string_iterator()
                key = override.get_key_element()
                sweep = [f"{key}={val}" for val in sweep_choices]
                lists.append(sweep)
            else:
                key = override.get_key_element()
                value = override.get_value_element_as_str()
                lists.append([f"{key}={value}"])
        jobs = list(itertools.product(*lists))
        return jobs

    def batch_size(self, jobs: Iterable) -> int:
        if self.max_batch_size is None or self.max_batch_size == -1:
            return len(jobs)
        else:
            return self.max_batch_size
