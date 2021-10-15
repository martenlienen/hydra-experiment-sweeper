# Hydra Experiment Sweeper

[hydra](https://hydra.cc/) is a great library to configure model trainings and numerical
experiments. Via its `--multirun` facility, you can launch multiple jobs at a time and
evaluate multiple combinations of settings such as learning rates and seeds easily.
However, the built-in sweepers only accept sweep ranges on the command line, which means
that reproducing a set of experiments for a paper or similar relies on you to remember or
write down the exact invocation of hydra that produced the results in the first place.

`hydra-experiment-sweeper` lets you configure experiments via YAML. This way colleagues
and yourself a year from now can re-run the same jobs with exactly the same parameters.

```yaml
defaults:
  - override hydra/sweeper: experiment

hydra:
  sweeper:
    overrides:
      learning_rate: 1e-5,1e-6,1e-7
      # Any other overrides such as seeds, model, dataset, etc.
```

## Installation

```sh
pip install hydra-experiment-sweeper
```

## How to Use

The sweeper accepts any override syntax that hydra accepts on the command line as well as
YAML lists for long values such as seeds or paths.

```yaml
defaults:
  - override hydra/sweeper: experiment

hydra:
  sweeper:
    max_batch_size: 2
    overrides:
      model: linreg,svm
      learning_rate: range(0.1, 0.3, 0.1)
      seed:
        - 123
        - 42
```

Running [this experiment](example/config/experiment.yaml) with `python example/train.py
--config-name experiment -m` launches the configured jobs.

<details>
  <summary>Experiment invocation and output</summary>

```
$ python example/train.py -m
[2021-10-15 17:57:18,241][HYDRA] ExperimentSweeper(max_batch_size=2, overrides={'model': 'linreg,svm', 'learning_rate': 'range(0.1, 0.3, 0.1)', 'seed': [123, 42]}) sweeping
[2021-10-15 17:57:18,242][HYDRA] Sweep output dir : multirun/2021-10-15/17-57-18
[2021-10-15 17:57:18,796][HYDRA] Launching 2 jobs locally
[2021-10-15 17:57:18,796][HYDRA] 	#0 : model=linreg learning_rate=0.1 seed=123
model:
  model_type: linear_regression
seed: 123
learning_rate: 0.1

[2021-10-15 17:57:18,888][HYDRA] 	#1 : model=linreg learning_rate=0.1 seed=42
model:
  model_type: linear_regression
seed: 42
learning_rate: 0.1

[2021-10-15 17:57:18,982][HYDRA] Launching 2 jobs locally
[2021-10-15 17:57:18,982][HYDRA] 	#2 : model=linreg learning_rate=0.2 seed=123
model:
  model_type: linear_regression
seed: 123
learning_rate: 0.2

[2021-10-15 17:57:19,079][HYDRA] 	#3 : model=linreg learning_rate=0.2 seed=42
model:
  model_type: linear_regression
seed: 42
learning_rate: 0.2

[2021-10-15 17:57:19,171][HYDRA] Launching 2 jobs locally
[2021-10-15 17:57:19,171][HYDRA] 	#4 : model=svm learning_rate=0.1 seed=123
model:
  model_type: support_vector_machine
seed: 123
learning_rate: 0.1

[2021-10-15 17:57:19,290][HYDRA] 	#5 : model=svm learning_rate=0.1 seed=42
model:
  model_type: support_vector_machine
seed: 42
learning_rate: 0.1

[2021-10-15 17:57:19,405][HYDRA] Launching 2 jobs locally
[2021-10-15 17:57:19,405][HYDRA] 	#6 : model=svm learning_rate=0.2 seed=123
model:
  model_type: support_vector_machine
seed: 123
learning_rate: 0.2

[2021-10-15 17:57:19,497][HYDRA] 	#7 : model=svm learning_rate=0.2 seed=42
model:
  model_type: support_vector_machine
seed: 42
learning_rate: 0.2
```
</details>

You can still override parameters on the command line. Let's say you want to run the same
experiment but instead of sweeping over the configured learning rates, you want to try a
different order of magnitude. `python example/train.py --config-name experiment -m
learning_rate=1e-6,5e-5,1e-5` works just as you would expect with the built-in sweeper and
overwrites the configured override.
