import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    Override,
    OverrideType,
    ValueType,
)
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
    overrides: Dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="experiment",
    node=SweeperConfig,
    provider="hydra-experiment-sweeper",
)


class ExperimentSweeper(Sweeper):
    """A hydra sweeper with configurable overrides for reproducible experiments."""

    def __init__(
        self, max_batch_size: Optional[int], overrides: Dict[str, Union[str, List[Any]]]
    ):
        super().__init__()

        self.max_batch_size = max_batch_size
        self.overrides = overrides

        self.config: Optional[DictConfig] = None
        self.launcher: Optional[Launcher] = None
        self.hydra_context: Optional[HydraContext] = None

    def __repr__(self):
        return (
            f"ExperimentSweeper(max_batch_size={self.max_batch_size!r}, "
            f"overrides={self.overrides!r})"
        )

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
        configured_overrides = self.parse_configured_overrides(parser)
        cmd_overrides = parser.parse_overrides(arguments)

        override_choices = []
        overriden = set()
        for override in reversed(configured_overrides + cmd_overrides):
            key = override.get_key_element()
            if key in overriden:
                # The key is already overriden by a higher-priority override, e.g. command
                # line setting overriding the config file.
                continue
            else:
                overriden.add(key)

            if override.is_sweep_override():
                values = override.sweep_string_iterator()
            else:
                values = [override.get_value_element_as_str()]

            override_choices.append([f"{key}={value}" for value in values])

        # Put the overrides in the same order as they were read from the configuration and
        # command line arguments. This is expected by the BasicSweeper tests that we
        # re-use.
        override_choices.reverse()

        jobs = list(itertools.product(*override_choices))
        return jobs

    def parse_configured_overrides(self, parser: OverridesParser):
        def parse(key, value):
            if isinstance(value, str):
                return parser.parse_override(f"{key}={value}")
            else:
                return Override(
                    type=OverrideType.CHANGE,
                    key_or_group=key,
                    value_type=ValueType.CHOICE_SWEEP,
                    _value=ChoiceSweep(list=value),
                )

        return [parse(key, value) for key, value in self.overrides.items()]

    def batch_size(self, jobs: Iterable) -> int:
        if self.max_batch_size is None or self.max_batch_size == -1:
            return len(jobs)
        else:
            return self.max_batch_size
