# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Logging utils for AlphaEvolve."""

import abc
import collections
from collections.abc import Mapping, Sequence
import textwrap
from typing import Any

from components import evolution_database


class StepTrace:
  """Hooks that `step_loop` calls at each stage of one step.

  `step_loop` gets one trace per step from `BaseLogger.begin_step` and calls
  the methods below in the order the step runs. The default implementation
  does nothing, so loggers that only count steps (e.g. `LocalLogger`) are not
  affected. `verbose_logging.VerboseFileLogger` returns a trace that records
  every stage, with timestamps, to a log file per coroutine.
  """

  def sampled(
      self,
      root_individual: evolution_database.Program,
      previous_individuals: Sequence[evolution_database.Program],
  ) -> None:
    """Called right after `database.sample_programs()`."""

  def prompt_built(self, prompt: str) -> None:
    """Called right after the prompt is built, before awaiting the LLM."""

  def llm_returned(self, llm_sample: str | None) -> None:
    """Called right after `await llm.generate_async(...)` returns."""

  def diffs_applied(self, code: str) -> None:
    """Called after the LLM diffs are applied, before awaiting the evaluator."""

  def evaluated(
      self, scores: Mapping[str, float | None], output: Any | None
  ) -> None:
    """Called right after `await evaluator.run_async(...)` returns."""

  def registered(self, program: evolution_database.Program) -> None:
    """Called after the child is registered in the database and logged."""

  def failed(self, error: Exception, traceback: str | None) -> None:
    """Called when the step fails with a recoverable error."""

  def cancelled(self) -> None:
    """Called when the step is cancelled while awaiting the LLM or evaluator."""


class BaseLogger(abc.ABC):
  """Base class for logging programs and scores."""

  def __init__(self):
    self._step_stats = collections.defaultdict(int)
    self._error_stats = collections.defaultdict(int)
    self._best_scores = collections.defaultdict(lambda: float('-inf'))

  @property
  def total_steps(self) -> int:
    return sum(self._step_stats.values())

  @property
  def step_stats(self) -> Mapping[str, int]:
    return dict(self._step_stats)

  @property
  def error_stats(self) -> Mapping[str, int]:
    return dict(self._error_stats)

  def log_step(
      self,
      program: evolution_database.Program,
      prompt: str | None,
      llm_sample: str | None,
  ):
    """Logs a step and updates the best scores."""
    del prompt, llm_sample  # Unused.
    self._step_stats['successful_steps'] += 1
    # Update best scores:
    for metric, score in program.scores.items():
      metric_key = f'scores_to_optimize/{metric}'
      if score is not None and score > self._best_scores[metric_key]:
        self._best_scores[metric_key] = score

  def log_error(
      self,
      program: str | None,
      error: Exception,
      traceback: str | None,
      prompt: str | None,
      llm_sample: str | None,
  ):
    """Logs an error and updates the best scores."""
    del program, traceback, prompt, llm_sample  # Unused.
    self._step_stats['failed_steps'] += 1
    self._error_stats[type(error).__name__] += 1

  def close(self) -> None:
    """Closes the logger and flushes the logs."""
    # No action needed by default.
    pass

  def log_run_start(self, *, config: Any, database: Any, **modules: Any):
    """Called once, after the initial program is evaluated, before the loops.

    Args:
      config: The `configuration.Configuration` of the run.
      database: The (shared) program database, already holding the seed.
      **modules: The other modules of the run (llm, evaluator, ...).
    """
    # No action needed by default.
    del config, database, modules  # Unused.

  def begin_step(self, coroutine_index: int) -> StepTrace:
    """Called by `step_loop` at the start of every step.

    Args:
      coroutine_index: Which copy of `step_loop` runs the step (0-based).

    Returns:
      A `StepTrace` that `step_loop` notifies as the step progresses.
    """
    del coroutine_index  # Unused.
    return StepTrace()

  def log_loop_end(self, coroutine_index: int) -> None:
    """Called when a `step_loop` exits because the step budget is used up."""
    # No action needed by default.
    del coroutine_index  # Unused.

  def get_state(self) -> Mapping[str, Any]:
    return {
        'step_stats': dict(self._step_stats),
        'error_stats': dict(self._error_stats),
        'best_scores': dict(self._best_scores),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self._step_stats.update(state['step_stats'])
    self._error_stats.update(state['error_stats'])
    self._best_scores.update(state['best_scores'])


class LocalLogger(BaseLogger):
  """Logs programs and scores to the console."""

  def log_step(
      self,
      program: evolution_database.Program,
      prompt: str | None,
      llm_sample: str | None,
  ):
    """Logs a step and updates the best scores."""
    super().log_step(program, prompt, llm_sample)
    output = program.output
    output_str = (
        str(output) if len(str(output)) < 100 else f'{str(output)[:100]}...'
    )
    print(textwrap.dedent(f"""\
        --------------------------------------------------
        Step:        {self.total_steps}
        Program:     \n{textwrap.indent(str(program.text), ' ' * 8)}
        Output:      {output_str}
        Scores:      {program.scores}
        Best Scores: {dict(self._best_scores)}
        --------------------------------------------------
        """))

  def log_error(
      self,
      program: str | None,
      error: Exception,
      traceback: str | None,
      prompt: str | None,
      llm_sample: str | None,
  ):
    """Logs an error and updates the best scores."""
    super().log_error(program, error, traceback, prompt, llm_sample)
    print(f'step: {self.total_steps}\nerror: {error!r}\ntraceback: {traceback}')
