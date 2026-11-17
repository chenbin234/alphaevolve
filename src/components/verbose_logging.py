"""Verbose Markdown logs of an AlphaEvolve run, one file per coroutine.

`VerboseFileLogger` does everything `LocalLogger` does (console output, step
and error counters, the step budget) and in addition writes one directory per
run:

    logs/<date>_<time>_<config name>/
    ├── overview.md      configuration, seed program, one row per finished step
    │                    (linking to its full record), final database, summary
    ├── coroutine_0.md   every step run by coroutine 0, in order
    ├── coroutine_1.md   ...
    └── ...
    logs/latest          symlink to the most recent run directory

For every step, the coroutine's file shows the program database as the step
found it, what was sampled from it, the exact prompt, the raw LLM response,
each SEARCH/REPLACE edit and the child program it produced, the evaluation,
where the child was stored, and when each phase started and how long it took.

The files are Markdown, so their headings form an outline you can jump around
in (VS Code: Ctrl+Shift+O or the Outline panel), and links between the files
work in a Markdown preview. They also read fine as plain text (`less`, `grep`).

The logger only observes. Every hook catches its own exceptions and reports
them in the log, so a bug in here can never stop or change the evolution.

Select it in the problem config with `logger: VerboseFileLogger`. The log
directory defaults to `<repo>/logs` and can be changed with the environment
variable `ALPHAEVOLVE_LOG_DIR`.
"""

import atexit
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import difflib
import functools
import json
import math
import os
import pathlib
import re
import sys
import time
import traceback as traceback_lib
from typing import Any

from components import evolution_database
from components import logging as base_logging
from components import parsing_utils
from components import prompting

_DEFAULT_LOG_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'logs'
_LOG_ROOT_ENV_VAR = 'ALPHAEVOLVE_LOG_DIR'
_OVERVIEW_FILE = 'overview.md'


def _coroutine_file(coroutine_index: int) -> str:
  return f'coroutine_{coroutine_index}.md'


# ==============================================================================
# Formatting helpers
# ==============================================================================


def _now() -> datetime.datetime:
  return datetime.datetime.now()


def _clock(t: datetime.datetime) -> str:
  """Wall-clock time with milliseconds, e.g. '10:31:15.442'."""
  return t.strftime('%H:%M:%S.%f')[:-3]


def _stamp(t: datetime.datetime) -> str:
  """Date and wall-clock time, e.g. '2026-11-17 10:31:15.442'."""
  return t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def _dur(seconds: float) -> str:
  """Human-readable duration: '412 µs', '3.1 ms', '23.41 s', '2 min 03.4 s'."""
  if seconds < 1e-3:
    return f'{seconds * 1e6:.0f} µs'
  if seconds < 1:
    return f'{seconds * 1e3:.1f} ms'
  if seconds < 60:
    return f'{seconds:.2f} s'
  minutes, secs = divmod(seconds, 60)
  if minutes < 60:
    return f'{int(minutes)} min {secs:04.1f} s'
  hours, minutes = divmod(minutes, 60)
  return f'{int(hours)} h {int(minutes):02d} min'


def _score(x: Any) -> str:
  if x is None:
    return 'None'
  if isinstance(x, bool):
    return str(x)
  if isinstance(x, (int, float)):
    return f'{x:.8g}'
  return str(x)


def _delta(new: Any, old: Any) -> str:
  try:
    d = float(new) - float(old)
  except (TypeError, ValueError):
    return '—'
  return f'{d:+.6g}' if math.isfinite(d) else '—'


def _as_number(x: Any) -> float:
  """Score as a float for comparisons (None / NaN / junk sort lowest)."""
  try:
    x = float(x)
  except (TypeError, ValueError):
    return -math.inf
  return -math.inf if math.isnan(x) else x


def _plural(n: int, word: str) -> str:
  return f'{n} {word}' if n == 1 else f'{n} {word}s'


def _oneline(text: str, limit: int = 90) -> str:
  text = ' '.join(text.split())
  return text if len(text) <= limit else text[: limit - 1] + '…'


def _truncate(text: str, limit: int) -> str:
  if len(text) <= limit:
    return text
  return (
      text[:limit]
      + f'\n… [{len(text) - limit:,} more characters not shown in this log]'
  )


def _code(text: str) -> str:
  """Inline code span, safe for text that contains backticks."""
  text = text.replace('\n', ' ')
  if not text:
    return '(empty)'
  longest = max((len(run) for run in re.findall('`+', text)), default=0)
  ticks = '`' * (longest + 1)
  pad = ' ' if '`' in text else ''
  return f'{ticks}{pad}{text}{pad}{ticks}'


def _fence(text: str | None, lang: str = '') -> str:
  """Fenced code block that is never closed early by fences in `text`.

  Prompts and LLM replies contain their own python code fences, so the fence is
  made one backtick longer than the longest backtick run inside `text`.

  Args:
    text: The verbatim content.
    lang: Info string for syntax highlighting.

  Returns:
    The fenced block.
  """
  text = '(empty)' if not text else text.rstrip()
  longest = max((len(run) for run in re.findall('`{3,}', text)), default=0)
  ticks = '`' * max(3, longest + 1)
  return f'{ticks}{lang}\n{text}\n{ticks}'


def _quote(text: str) -> str:
  return '\n'.join(f'> {line}' if line else '>' for line in text.splitlines())


def _cell(value: Any) -> str:
  return str(value).replace('\n', ' ').replace('|', r'\|')


def _table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], max_pad: int = 48
) -> str:
  """Markdown table, padded so that it is also aligned as plain text."""
  grid = [[_cell(h) for h in headers]] + [[_cell(c) for c in r] for r in rows]
  widths = [
      min(max_pad, max(len(r[i]) for r in grid)) for i in range(len(headers))
  ]

  def line(cells):
    return '| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, widths)) + ' |'

  rule = '|' + '|'.join('-' * (max(w, 3) + 2) for w in widths) + '|'
  return '\n'.join([line(grid[0]), rule] + [line(r) for r in grid[1:]])


def _bullets(lines: Sequence[str]) -> str:
  return '\n'.join(f'- {line}' for line in lines)


def _wait_stats(waits: Sequence[float]) -> str:
  if not waits:
    return '—'
  return (
      f'{len(waits)} × · total {_dur(sum(waits))} · mean'
      f' {_dur(sum(waits) / len(waits))} · min {_dur(min(waits))} · max'
      f' {_dur(max(waits))}'
  )


class _MarkdownFile:
  """Append-only Markdown file, flushed after every write."""

  def __init__(self, path: pathlib.Path):
    self.path = path
    self._f = open(path, 'w', encoding='utf-8')
    self._after_line = False

  def blocks(self, *parts: str | None) -> None:
    """Appends Markdown blocks, separated by blank lines."""
    text = '\n\n'.join(p.strip('\n') for p in parts if p)
    if not text or self._f.closed:
      return
    self._f.write(('\n' if self._after_line else '') + text + '\n\n')
    self._f.flush()
    self._after_line = False

  def line(self, text: str) -> None:
    """Appends a single line (e.g. a table row) without a blank line after."""
    if self._f.closed:
      return
    self._f.write(text + '\n')
    self._f.flush()
    self._after_line = True

  def close(self) -> None:
    if not self._f.closed:
      self._f.close()


# ==============================================================================
# Program IDs and database snapshots
# ==============================================================================


@dataclasses.dataclass
class _ProgramInfo:
  """What the logger knows about one distinct program (one source text)."""

  pid: str  # 'P0000', 'P0001', ... in order of creation.
  origin: str  # 'seed', the step that made it (e.g. 'c2-003'), or 'unknown'.
  file: str | None  # Log file showing its full source.
  parent: str | None
  created_at: datetime.datetime
  scores: dict[str, Any] | None = None
  status: str = 'waiting for evaluation'

  @property
  def link(self) -> str:
    """Markdown link to the place where its full source is logged."""
    if self.file is None:
      return self.pid
    return f'[{self.pid}]({self.file}#{self.pid.lower()})'

  @property
  def origin_link(self) -> str:
    if self.origin in ('seed', 'unknown') or self.file is None:
      return self.origin
    return f'[{self.origin}]({self.file}#step-{self.origin})'


class _ProgramRegistry:
  """Assigns IDs to programs, keyed by source text.

  `Program` objects are frozen and the database stores modified copies of them
  (`dataclasses.replace`), so object identity is lost. The source text is what
  the database itself uses to tell programs apart, so it is used here too:
  identical code always gets the same ID.
  """

  def __init__(self):
    self._by_text: dict[str, _ProgramInfo] = {}
    self.by_pid: dict[str, _ProgramInfo] = {}
    self.all: list[_ProgramInfo] = []

  def get(self, text: str) -> _ProgramInfo | None:
    return self._by_text.get(text)

  def add(self, text: str, **kwargs: Any) -> _ProgramInfo:
    info = _ProgramInfo(pid=f'P{len(self.all):04d}', **kwargs)
    self._by_text[text] = info
    self.by_pid[info.pid] = info
    self.all.append(info)
    return info

  def of(self, program: evolution_database.Program) -> _ProgramInfo:
    info = self._by_text.get(program.text)
    if info is None:  # Not expected: every program is seen when it is made.
      info = self.add(
          program.text,
          origin='unknown',
          file=None,
          parent=None,
          created_at=_now(),
          scores=dict(program.scores),
          status='found in the database',
      )
    return info


@dataclasses.dataclass(frozen=True)
class _Entry:
  """One program as stored in the database at snapshot time."""

  pid: str
  scores: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class _DbView:
  """Immutable snapshot of the database (the live lists keep changing)."""

  available: bool
  hall_of_fame: Mapping[str, _Entry]
  islands: Sequence[Sequence[_Entry]] | None
  last_reset_time: datetime.datetime | None


# ==============================================================================
# Prompt and diff analysis (read-only re-computations for the log)
# ==============================================================================


def _prompt_recipe(prompt: str) -> list[tuple[str, str, float]]:
  """Finds which random template options `TemplatePromptSampler` picked."""
  try:
    options_by_name = json.loads(prompting.PROMPT_FORMAT_ARGS)
  except (AttributeError, ValueError):
    return []
  task_part = prompt.rsplit('# Task', 1)[-1]
  recipe = []
  for name, options in options_by_name.items():
    non_empty = sorted(
        (o for o in options if o[0].strip()),
        key=lambda o: len(o[0]),
        reverse=True,
    )
    chosen = next((o for o in non_empty if o[0].strip() in task_part), None)
    if chosen is None:
      chosen = next((o for o in options if not o[0].strip()), None)
    if chosen is not None:
      recipe.append((name, chosen[0], chosen[1]))
  return recipe


@dataclasses.dataclass(frozen=True)
class _Edit:
  search: str
  replace: str
  applied: bool


def _analyze_edits(
    parent_text: str, llm_sample: str
) -> tuple[int, list[_Edit]]:
  """Re-runs the steps of `parsing_utils.apply_diffs`, recording each edit.

  Args:
    parent_text: Source of the parent program.
    llm_sample: The raw LLM reply.

  Returns:
    The number of python code blocks in the reply, and for each SEARCH/REPLACE
    edit whether its SEARCH text matched (i.e. whether it changed anything).
  """
  code_blocks = re.findall(
      parsing_utils.code_block_pattern('(.*?)'), llm_sample, re.DOTALL
  )
  pairs = []
  for block in code_blocks:
    pairs += parsing_utils._find_edit_blocks(block)  # pylint: disable=protected-access
  mutable = [
      block.code
      for block in parsing_utils._parse_evolve_blocks(  # pylint: disable=protected-access
          parent_text,
          block_start=parsing_utils._EVOLVE_BLOCK_START,  # pylint: disable=protected-access
          block_end=parsing_utils._EVOLVE_BLOCK_END,  # pylint: disable=protected-access
          evolve_entire_file=False,
      )
      if block.is_mutable
  ]
  edits = []
  for search, replace in pairs:
    applied = False
    for i, block_code in enumerate(mutable):
      if search in block_code:
        mutable[i] = block_code.replace(search, replace, 1)
        applied = True
    edits.append(_Edit(search, replace, applied))
  return len(code_blocks), edits


def _first_line(text: str) -> str:
  return next((line.strip() for line in text.splitlines() if line.strip()), '')


def _n_lines(text: str) -> int:
  return len(text.splitlines())


# ==============================================================================
# One step
# ==============================================================================


def _guarded(method):
  """Makes a logging hook swallow and report its own exceptions."""

  @functools.wraps(method)
  def wrapper(self, *args, **kwargs):
    try:
      return method(self, *args, **kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
      self._report_bug(method.__name__, traceback_lib.format_exc())  # pylint: disable=protected-access
      return None

  return wrapper


# (start mark, end mark, label) of each timed phase of a step. The numbers match
# the section numbers in the step's log.
_PHASES = (
    ('begin', 'sampled', '2 · sample programs'),
    ('sampled', 'prompt', '3 · build prompt'),
    ('prompt', 'llm', '4 · ⏳ wait for LLM'),
    ('llm', 'diff', '5 · apply diffs'),
    ('diff', 'eval', '6 · ⏳ wait for evaluator'),
    ('eval', 'end', '7 · register + log'),
)


@dataclasses.dataclass
class _CoroutineStats:
  iterations: int = 0
  ok: int = 0
  failed: int = 0
  cancelled: int = 0
  unfinished: int = 0
  llm_waits: list[float] = dataclasses.field(default_factory=list)
  eval_waits: list[float] = dataclasses.field(default_factory=list)
  step_seconds: float = 0.0
  ended: str = 'still running when the log was closed'


class _VerboseStepTrace(base_logging.StepTrace):
  """Records one step of one coroutine into that coroutine's log file.

  `step_loop` calls the hooks in order; each hook stamps the time and writes
  the section for the phase that just finished, so the file can be followed
  live (e.g. `tail -f`).
  """

  def __init__(self, logger: 'VerboseFileLogger', coroutine_index: int):
    self._logger = logger
    self._stats = logger._stats[coroutine_index]  # pylint: disable=protected-access
    self._stats.iterations += 1
    logger._steps_started += 1  # pylint: disable=protected-access

    self.coroutine_index = coroutine_index
    self.iteration = self._stats.iterations
    self.global_index = logger._steps_started  # pylint: disable=protected-access
    self.step_id = f'c{coroutine_index}-{self.iteration:03d}'
    self.file_name = _coroutine_file(coroutine_index)
    self._file = logger._file_for(coroutine_index)  # pylint: disable=protected-access

    # State of the run when the step starts.
    now = time.perf_counter()
    self.steps_done_at_start = logger.total_steps
    self.ok_before, self.failed_before = self._stats.ok, self._stats.failed
    self.others_in_flight = [
        f'{t.step_id} {t.phase} ({_dur(now - t.phase_since)} so far)'
        for t in logger._open_traces.values()  # pylint: disable=protected-access
    ]
    self.db_before = logger._db_view()  # pylint: disable=protected-access
    self.registrations_at_start = len(logger._registrations)  # pylint: disable=protected-access

    # Filled in as the step progresses.
    self.marks: dict[str, tuple[float, datetime.datetime]] = {}
    self.phase, self.phase_since = 'sampling', now
    self.root: evolution_database.Program | None = None
    self.parent: _ProgramInfo | None = None
    self.source = '?'
    self.llm_sample: str | None = None
    self.child: _ProgramInfo | None = None
    self.child_is_new = False
    self.scores: dict[str, Any] | None = None
    self.db_before_register: _DbView | None = None
    self.result = ''

    self._mark('begin')
    logger._open_traces[coroutine_index] = self  # pylint: disable=protected-access

  # ---------------------------------------------------------------- helpers --

  def _mark(self, name: str, next_phase: str | None = None) -> None:
    self.marks[name] = (time.perf_counter(), _now())
    if next_phase is not None:
      self.phase, self.phase_since = next_phase, self.marks[name][0]

  def _seconds(self, start: str, end: str) -> float:
    return self.marks[end][0] - self.marks[start][0]

  def _time(self, mark: str) -> str:
    return _clock(self.marks[mark][1])

  def _main(self, scores: Mapping[str, Any] | None) -> str:
    if scores is None:
      return '—'
    return _score(scores.get(self._logger._main_metric))  # pylint: disable=protected-access

  def _report_bug(self, hook: str, tb: str) -> None:
    try:
      self._logger._logger_bugs.append(f'{self.step_id}: {hook}()')  # pylint: disable=protected-access
      self._file.blocks(
          f'> ⚠ **VerboseFileLogger bug in `{hook}()`.** The run is not'
          ' affected; only this part of the log is missing.',
          _fence(tb, 'text'),
      )
      print(
          f'[VerboseFileLogger] internal error in {hook}() (the run'
          f' continues):\n{tb}',
          file=sys.stderr,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  # ------------------------------------------------------------------ hooks --

  @_guarded
  def sampled(
      self,
      root_individual: evolution_database.Program,
      previous_individuals: Sequence[evolution_database.Program],
  ) -> None:
    self._mark('sampled', 'building the prompt')
    lg = self._logger
    self.root = root_individual
    self.parent = lg._programs.of(root_individual)  # pylint: disable=protected-access
    self.source = lg._source_label(root_individual)  # pylint: disable=protected-access
    n = lg._num_coroutines  # pylint: disable=protected-access

    header_rows = [
        (
            'coroutine',
            f'{self.coroutine_index}, one of {n} copies of `step_loop`'
            f' (indices 0–{n - 1})',
        ),
        (
            'iteration',
            f'{self.iteration} of this coroutine (before it:'
            f' {self.ok_before} ✅ · {self.failed_before} ❌)',
        ),
        (
            'global step',
            f'#{self.global_index} in start order across all coroutines',
        ),
        (
            'budget when started',
            f'{self.steps_done_at_start} / {lg._budget} steps finished'  # pylint: disable=protected-access
            ' (`logger.total_steps`)',
        ),
        (
            'other steps in flight',
            ' · '.join(self.others_in_flight) or 'none',
        ),
        ('started', _stamp(self.marks['begin'][1])),
    ]
    self._file.blocks(
        '---',
        f'## Step {self.step_id}',
        f'**coroutine {self.coroutine_index} · iteration {self.iteration} ·'
        f' global step #{self.global_index} · {self.source} · started'
        f' {self._time("begin")}**',
        _table(['field', 'value'], header_rows),
        '### 1 · Database before sampling',
        'The shared database exactly as `sample_programs()` saw it. Other'
        ' coroutines may change it while this step waits for the LLM and'
        ' the evaluator.',
        lg._render_db(self.db_before),  # pylint: disable=protected-access
        f'### 2 · Sampling · `database.sample_programs()` → {self.source}',
        lg._strategy_text(root_individual, self.db_before),  # pylint: disable=protected-access
        self._sampling_table(previous_individuals),
        f'Took {_dur(self._seconds("begin", "sampled"))}.',
    )

  def _sampling_table(
      self, previous_individuals: Sequence[evolution_database.Program]
  ) -> str:
    lg = self._logger
    main = lg._main_metric  # pylint: disable=protected-access
    root_render = self.root.render()
    rows = [(
        'parent · `root_individual`',
        self.parent.link,
        self._main(self.root.scores),
        self.parent.origin_link,
        'yes, as "Current program": the LLM edits this one',
    )]
    seen = set()
    for i, program in enumerate(previous_individuals):
      info = lg._programs.of(program)  # pylint: disable=protected-access
      # Same filter as TemplatePromptSampler.sample_prompt.
      rendered = program.render()
      if rendered == root_render:
        in_prompt = 'no: same code and scores as the parent'
      elif rendered in seen:
        in_prompt = 'no: duplicate of an earlier inspiration'
      else:
        in_prompt = 'yes, under "Prior programs" (context only)'
        seen.add(rendered)
      rows.append((
          f'inspiration · `previous_individuals[{i}]`',
          info.link,
          self._main(program.scores),
          info.origin_link,
          in_prompt,
      ))
    table = _table(['role', 'program', f'{main} ★', 'made by', 'in prompt?'],
                   rows)
    if not previous_individuals:
      table += (
          '\n\nNo inspirations were sampled, so the "Prior programs" section'
          ' of the prompt is empty.'
      )
    return table

  @_guarded
  def prompt_built(self, prompt: str) -> None:
    self._mark('prompt', 'waiting for the LLM')
    recipe = _prompt_recipe(prompt)
    recipe_rows = [
        (
            f'`{name}`',
            f'“{_oneline(value, 100)}”' if value.strip() else '(empty)',
            f'{prob:g}',
        )
        for name, value, prob in recipe
    ]
    self._file.blocks(
        '### 3 · Prompt · `prompt_sampler.sample_prompt()`',
        recipe
        and 'Random parts of the template chosen for this prompt (from'
        ' `prompting.PROMPT_FORMAT_ARGS`):',
        recipe and _table(['placeholder', 'chosen', 'probability'],
                          recipe_rows),
        f'**{len(prompt):,} characters · {_n_lines(prompt)} lines** · built in'
        f' {_dur(self._seconds("sampled", "prompt"))}. Exactly as sent to the'
        ' LLM:',
        _fence(prompt, 'text'),
        f'⏳ **{self._time("prompt")} · sent to the LLM.** This coroutine now'
        ' awaits `llm.generate_async(prompt)`; meanwhile the event loop runs'
        ' the other coroutines.',
    )

  @_guarded
  def llm_returned(self, llm_sample: str | None) -> None:
    self._mark('llm', 'applying diffs')
    waited = self._seconds('prompt', 'llm')
    self._stats.llm_waits.append(waited)
    self.llm_sample = llm_sample
    parts = [
        f'### 4 · LLM response · ⏳ waited {_dur(waited)}',
        f'⏳ Waited **{_dur(waited)}** for the LLM: sent'
        f' {self._time("prompt")} → received {self._time("llm")}.'
        f' {self._logger._model_text}'.rstrip(),  # pylint: disable=protected-access
    ]
    if llm_sample is None:
      parts.append(
          'The LLM returned no text (`None`), so there is nothing to apply.'
      )
    else:
      n_blocks = len(
          re.findall(
              parsing_utils.code_block_pattern('(.*?)'), llm_sample, re.DOTALL
          )
      )
      n_search = llm_sample.count(parsing_utils.SEARCH)
      parts += [
          f'**{len(llm_sample):,} characters · {_n_lines(llm_sample)} lines'
          f' · {_plural(n_blocks, "python code block")} ·'
          f' {_plural(n_search, "SEARCH marker")}**. Raw text:',
          _fence(llm_sample, 'text'),
      ]
    self._file.blocks(*parts)

  @_guarded
  def diffs_applied(self, code: str) -> None:
    self._mark('diff', 'waiting for the evaluator')
    lg = self._logger
    n_blocks, edits = _analyze_edits(self.root.text, self.llm_sample or '')
    n_applied = sum(e.applied for e in edits)

    self.child = lg._programs.get(code)  # pylint: disable=protected-access
    if self.child is None:
      self.child_is_new = True
      self.child = lg._programs.add(  # pylint: disable=protected-access
          code,
          origin=self.step_id,
          file=self.file_name,
          parent=self.parent.pid,
          created_at=self.marks['diff'][1],
      )
    unchanged = code == self.root.text
    title = f'{self.child.pid} (unchanged)' if unchanged else self.child.pid

    parts = [
        f'### 5 · Applying diffs · `parsing_utils.apply_diffs()` → {title}'
    ]
    if edits:
      parts.append(
          f'The reply has {_plural(n_blocks, "python code block")} with'
          f' {_plural(len(edits), "SEARCH/REPLACE edit")}: **{n_applied}'
          f' applied, {len(edits) - n_applied} without a match**. Edits are'
          ' applied in order, only to the EVOLVE-BLOCK code of the parent'
          f' {self.parent.pid} (never to the inspirations). Each one replaces'
          ' the first occurrence of its SEARCH text; a SEARCH text that'
          ' matches nothing is skipped without an error.'
      )
    else:
      parts.append(
          f'The reply has {_plural(n_blocks, "python code block")} and no'
          ' SEARCH/REPLACE edit, so there is nothing to apply.'
      )
    if edits:
      parts.append(
          _table(
              ['edit', 'SEARCH (first line)', 'lines', 'result'],
              [
                  (
                      i + 1,
                      _code(_oneline(_first_line(e.search), 70)),
                      f'{_n_lines(e.search)} → {_n_lines(e.replace)}',
                      '✅ applied' if e.applied else '❌ no match: skipped',
                  )
                  for i, e in enumerate(edits)
              ],
          )
      )
    if unchanged:
      if not edits:
        why = 'the reply contains no edit'
      elif not n_applied:
        why = 'none of its edits matched'
      else:
        why = 'the edits that matched did not change the text'
      parts.append(
          f'⚠ **No change.** The child is identical to its parent'
          f' {self.parent.pid}, because {why}. It is still evaluated and'
          ' still uses one step of the budget.'
      )
    else:
      diff_lines = list(
          difflib.unified_diff(
              self.root.text.splitlines(keepends=True),
              code.splitlines(keepends=True),
              fromfile=f'{self.parent.pid} (parent)',
              tofile=f'{self.child.pid} (child)',
          )
      )
      added = sum(
          1 for d in diff_lines if d.startswith('+') and not d.startswith('+++')
      )
      removed = sum(
          1 for d in diff_lines if d.startswith('-') and not d.startswith('---')
      )
      if self.child_is_new:
        parts += [
            f'#### {self.child.pid}',
            f'**New program {self.child.pid}** · child of {self.parent.link} ·'
            f' made by {self.step_id} at {self._time("diff")} ·'
            f' {_plural(added, "line")} added, {removed} removed.',
        ]
      else:
        parts.append(
            f'The child is identical to the existing program'
            f' {self.child.link} (made by {self.child.origin_link}), so it'
            ' keeps that ID.'
        )
      parts += [
          f'What changed (unified diff, parent {self.parent.pid} → child'
          f' {self.child.pid}):',
          _fence(''.join(diff_lines), 'diff'),
      ]
      if self.child_is_new:
        parts += [
            '<details>\n<summary>Full source of'
            f' {self.child.pid} ({_n_lines(code)} lines)</summary>',
            _fence(code, 'python'),
            '</details>',
        ]
    parts.append(
        f'⏳ **{self._time("diff")} · sent to the evaluator.** This coroutine'
        ' now awaits `evaluator.run_async(code, parent_output)`'
        f' ({lg._evaluator_name}); meanwhile the event loop runs the other'  # pylint: disable=protected-access
        ' coroutines.'
    )
    self._file.blocks(*parts)

  @_guarded
  def evaluated(
      self, scores: Mapping[str, float | None], output: Any | None
  ) -> None:
    self._mark('eval', 'registering')
    lg = self._logger
    waited = self._seconds('diff', 'eval')
    self._stats.eval_waits.append(waited)
    self.scores = dict(scores)
    # Taken right before `register_program` (no `await` in between), so the
    # registration section can show exactly what that call changed.
    self.db_before_register = lg._db_view()  # pylint: disable=protected-access
    if self.child_is_new:
      self.child.scores = dict(scores)
      self.child.status = 'evaluated'

    parent_scores = self.root.scores
    rows = []
    for metric in dict.fromkeys([*scores, *parent_scores]):
      record = self.db_before_register.hall_of_fame.get(metric)
      record_text = (
          f'{_score(record.scores.get(metric))} ({record.pid})'
          if record
          else '—'
      )
      rows.append((
          f'{metric} ★' if metric == lg._main_metric else metric,  # pylint: disable=protected-access
          _score(parent_scores.get(metric)),
          _score(scores.get(metric)),
          _delta(scores.get(metric), parent_scores.get(metric)),
          record_text,
      ))
    if output is None:
      output_text = '`output_artifacts`: None (`evaluate()` returned none).'
    else:
      output_text = (
          '`output_artifacts` (passed to the children of this program as'
          ' `PARENT_OUTPUT`):\n\n'
          + _fence(_truncate(repr(output), lg._max_output_chars), 'text')  # pylint: disable=protected-access
      )
    self._file.blocks(
        f'### 6 · Evaluation · ⏳ waited {_dur(waited)}',
        f'⏳ Waited **{_dur(waited)}** for the evaluator: sent'
        f' {self._time("diff")} → returned {self._time("eval")}. This'
        ' includes waiting for a free worker thread, starting the sandbox and'
        ' running `evaluate()`.',
        _table(
            [
                'metric',
                f'parent {self.parent.pid}',
                f'child {self.child.pid}',
                'Δ vs parent',
                'record before this step',
            ],
            rows,
        ),
        '★ = main metric (`main_metric_name`): it picks the best program at'
        ' the end of the run.',
        output_text,
    )

  @_guarded
  def registered(self, program: evolution_database.Program) -> None:
    del program  # Already known: same text and scores as `self.child`.
    self._mark('end')
    lg = self._logger
    after = lg._db_view()  # pylint: disable=protected-access
    lines, stored = self._registration_changes(self.db_before_register, after)

    others = [
        f'{pid} ({step_id})'
        for step_id, pid in lg._registrations[self.registrations_at_start:]  # pylint: disable=protected-access
    ]
    lg._registrations.append((self.step_id, self.child.pid))  # pylint: disable=protected-access
    if others:
      lines.append(
          f'While this step was in flight (since {self._time("sampled")}),'
          f' other coroutines registered {_plural(len(others), "program")}:'
          f' {", ".join(others)}. They were not available to this step\'s'
          ' prompt.'
      )
    else:
      lines.append(
          'No other coroutine registered a program while this step was in'
          ' flight.'
      )
    lines.append(
        f'Budget: **{lg.total_steps} / {lg._budget}** steps finished after'  # pylint: disable=protected-access
        ' this one (`logger.total_steps`).'
    )
    parts = [
        f'### 7 · Registration · `database.register_program({self.child.pid})`',
        _bullets(lines),
    ]
    if after.last_reset_time != self.db_before_register.last_reset_time:
      parts += ['The islands after the reset:', lg._render_db(after)]  # pylint: disable=protected-access

    self._stats.ok += 1
    self.result = '✅ ' + (' · '.join(stored) if stored else 'not stored')
    places = [s for s in stored if not s.startswith('🔄')]
    headline = 'stored in ' + ', '.join(places) if places else 'not stored'
    if len(places) < len(stored):
      headline += ' · 🔄 islands reset'
    self._finish(parts, f'✅ Step {self.step_id} finished · {headline}')

  def _registration_changes(
      self, before: _DbView, after: _DbView
  ) -> tuple[list[str], list[str]]:
    """Describes what `register_program` changed, and where the child went."""
    lines, stored = [], []
    reset = before.last_reset_time != after.last_reset_time
    if reset:
      hours = getattr(self._logger._database, '_island_reset_interval_hours',  # pylint: disable=protected-access
                      None)
      if self.root.sampled_from == 'islands':
        appended = (
            f'The child was first appended to island {self.root.island_index};'
            ' then '
        )
      else:
        appended = 'Then '
      survivors = [
          f'island {i}'
          for i, island in enumerate(after.islands or [])
          if any(e.pid == self.child.pid for e in island)
      ]
      fate = (
          f'{self.child.pid} survived the reset and is now in'
          f' {", ".join(survivors)}.'
          if survivors
          else f'{self.child.pid} was not among the programs kept.'
      )
      lines.append(
          f'🔄 **The islands were reset** during this call. {appended}'
          '`_maybe_reset_islands()` found that'
          + (f' {hours:g} h had' if hours else ' enough time had')
          + ' passed since the last reset, emptied every island and reseeded'
          ' them with survivors of the old islands: each old island\'s best'
          ' program plus up to 4 random others (shown below).'
          f' {fate}'
      )
      if survivors:
        stored.append(', '.join(survivors))
      stored.append('🔄 islands reset')
    elif before.islands is not None and after.islands is not None:
      for i, (old, new) in enumerate(zip(before.islands, after.islands)):
        for entry in new[len(old):]:
          lines.append(
              f'✅ Appended {entry.pid} to **island {i}** (now'
              f' {_plural(len(new), "program")}). A child of an island'
              ' program always joins that island, whatever its score.'
          )
          stored.append(f'island {i}')
    for metric, entry in after.hall_of_fame.items():
      old = before.hall_of_fame.get(metric)
      if old is not None and (old.pid, old.scores.get(metric)) == (
          entry.pid,
          entry.scores.get(metric),
      ):
        continue
      old_text = (
          f'{old.pid} ({_score(old.scores.get(metric))})' if old else 'nothing'
      )
      lines.append(
          f'🏆 Hall of fame `{metric}`: {old_text} →'
          f' **{entry.pid} ({_score(entry.scores.get(metric))})**, a new'
          ' record.'
      )
      stored.append(f'🏆 {metric}')
    if not stored:
      if self.root.sampled_from == 'best_programs_per_metric':
        reason = (
            'the parent came from the hall of fame, so the child could only'
            ' be kept by beating a metric record, and it did not'
        )
      else:
        reason = 'it did not beat any metric record'
      lines.append(
          f'∅ **Not stored anywhere**: {reason}. No other coroutine can'
          ' sample it; it lives on only in this log.'
      )
    return lines, stored

  @_guarded
  def failed(self, error: Exception, traceback: str | None) -> None:
    self._mark('end')
    lg = self._logger
    self._stats.failed += 1
    message = _truncate(str(error), lg._max_error_chars)  # pylint: disable=protected-access
    error_name = type(error).__name__
    if 'llm' not in self.marks:
      waited = self._seconds('prompt', 'end')
      self._stats.llm_waits.append(waited)
      where = 'while waiting for the LLM'
      heading = f'### 4 · LLM response · ❌ failed after {_dur(waited)}'
    elif 'diff' not in self.marks:
      where = 'while applying diffs'
      heading = '### 5 · Applying diffs · ❌ failed'
    else:
      waited = self._seconds('diff', 'end')
      self._stats.eval_waits.append(waited)
      where = 'during evaluation'
      heading = f'### 6 · Evaluation · ❌ failed after {_dur(waited)}'
      if self.child_is_new:
        self.child.status = f'evaluation failed ({error_name})'
    parts = [
        heading,
        f'**{error_name}** {where}:',
        _fence(message, 'text'),
        'Nothing is stored in the database. The failure still counts as one'
        f' step of the budget (now **{lg.total_steps} / {lg._budget}**), and'  # pylint: disable=protected-access
        ' the loop starts its next step right away.',
        '<details>\n<summary>Traceback</summary>',
        _fence(traceback, 'text'),
        '</details>',
    ]
    self.result = f'❌ {error_name} ({where.split()[-1]})'
    self._finish(parts, f'❌ Step {self.step_id} failed · {error_name} {where}')

  @_guarded
  def cancelled(self) -> None:
    phase, since = self.phase, self.phase_since
    self._mark('end')
    waited = self.marks['end'][0] - since
    self._stats.cancelled += 1
    self._stats.ended = (
        f'cancelled at {self._time("end")} while {phase} (step'
        f' {self.step_id})'
    )
    if self.child is not None and self.child_is_new:
      self.child.status = 'cancelled before evaluation finished'
    lg = self._logger
    if lg.total_steps >= lg._budget:  # pylint: disable=protected-access
      why = (
          'Another coroutine found the budget used up, so `async_run`'
          ' cancelled the remaining coroutines'
      )
    else:
      why = (
          f'The run was stopped before the budget was used up'
          f' ({lg.total_steps} / {lg._budget}, e.g. Ctrl+C or a crash in'  # pylint: disable=protected-access
          ' another coroutine)'
      )
    parts = [
        f'⛔ **Cancelled at {self._time("end")}**, after {phase} for'
        f' {_dur(waited)}. {why}: `CancelledError` was raised at this'
        ' coroutine\'s `await`. The step is discarded, nothing is registered'
        ' and it does not count toward the budget. (The worker thread is not'
        ' stopped; it runs until its LLM request or evaluation ends.)'
    ]
    self.result = f'⛔ cancelled ({phase})'
    self._finish(
        parts, f'⛔ Step {self.step_id} cancelled · while {phase}', counted=False
    )

  def never_finished(self) -> None:
    """Called by `close()` for a step that neither finished nor was cancelled."""
    phase = self.phase
    self._mark('end')
    self._stats.unfinished += 1
    self.result = f'⚠ never finished ({phase})'
    self._finish(
        [
            f'⚠ The log was closed at {self._time("end")} while this step was'
            f' still {phase}: the run ended without finishing or cancelling'
            ' it (e.g. it crashed or was interrupted).'
        ],
        f'⚠ Step {self.step_id} never finished · {phase}',
        counted=False,
    )

  def _finish(
      self, parts: list[str], headline: str, counted: bool = True
  ) -> None:
    """Writes the end-of-step summary and the overview timeline row."""
    lg = self._logger
    lg._open_traces.pop(self.coroutine_index, None)  # pylint: disable=protected-access
    total = self._seconds('begin', 'end')
    self._stats.step_seconds += total

    rows = []
    for start, stop, label in _PHASES:
      if start not in self.marks:
        break
      partial = stop not in self.marks
      took = self.marks['end' if partial else stop][0] - self.marks[start][0]
      rows.append((
          label + (' · interrupted' if partial else ''),
          self._time(start),
          _dur(took),
          f'{100 * took / total:.1f} %' if total else '—',
      ))
      if partial:
        break
    rows.append(('**total**', self._time('begin'), f'**{_dur(total)}**',
                 '100 %'))

    s = self._stats
    iterations = _plural(s.iterations, 'iteration')
    so_far = (
        f'Coroutine {self.coroutine_index} so far: {iterations} ({s.ok} ✅ ·'
        f' {s.failed} ❌ · {s.cancelled} ⛔) · waited'
        f' {_dur(sum(s.llm_waits))} for the LLM and'
        f' {_dur(sum(s.eval_waits))} for the evaluator in total.'
    )
    budget = (
        f'Budget: {lg.total_steps} / {lg._budget} steps finished.'  # pylint: disable=protected-access
        if counted
        else ''
    )
    self._file.blocks(
        *parts,
        f'### {headline} · {_dur(total)}',
        _table(['phase', 'started', 'took', 'share of step'], rows),
        f'{so_far} {budget}'.strip(),
    )
    lg._add_timeline_row(self, counted)  # pylint: disable=protected-access


# ==============================================================================
# The logger
# ==============================================================================


class VerboseFileLogger(base_logging.LocalLogger):
  """`LocalLogger` plus one verbose Markdown log file per coroutine.

  See the module docstring for the layout of the log directory.
  """

  def __init__(
      self,
      log_root: str | os.PathLike[str] | None = None,
      max_output_chars: int = 5_000,
      max_error_chars: int = 20_000,
  ):
    """Initializes the logger.

    Args:
      log_root: Directory that receives one sub-directory per run. Defaults to
        $ALPHAEVOLVE_LOG_DIR, or else `logs/` at the root of the repository.
      max_output_chars: `output_artifacts` longer than this are truncated in
        the log (they can be large arrays).
      max_error_chars: Error messages longer than this are truncated in the log
        (a failed evaluation includes the container's whole output).
    """
    super().__init__()
    self._log_root = pathlib.Path(
        log_root or os.environ.get(_LOG_ROOT_ENV_VAR) or _DEFAULT_LOG_ROOT
    )
    self._max_output_chars = max_output_chars
    self._max_error_chars = max_error_chars

    self.run_dir: pathlib.Path | None = None
    self._started = False
    self._closed = False
    self._files: dict[int, _MarkdownFile] = {}
    self._overview: _MarkdownFile | None = None
    self._programs = _ProgramRegistry()
    self._stats: dict[int, _CoroutineStats] = collections.defaultdict(
        _CoroutineStats
    )
    self._open_traces: dict[int, _VerboseStepTrace] = {}
    self._registrations: list[tuple[str, str]] = []  # (step id, program id)
    self._steps_started = 0
    self._logger_bugs: list[str] = []
    self._loop_ends: list[str] = []

    # Set by `log_run_start`.
    self._config = None
    self._database = None
    self._t0 = _now()
    self._budget = 0
    self._num_coroutines = 0
    self._main_metric = ''
    self._model_text = ''
    self._evaluator_name = 'evaluator'
    self._timeline_widths: list[int] = []

  # ------------------------------------------------------------- run start --

  def log_run_start(self, *, config: Any, database: Any, **modules: Any):
    try:
      self._start(config, database, modules)
    except Exception:  # pylint: disable=broad-exception-caught
      self._report_bug('log_run_start', traceback_lib.format_exc())

  def _start(self, config, database, modules: Mapping[str, Any]) -> None:
    self._config, self._database = config, database
    self._t0 = _now()
    self._budget = config.budget.num_llm_calls
    self._num_coroutines = config.budget.num_coroutines
    self._main_metric = config.problem_spec.main_metric_name

    llm = modules.get('llm')
    evaluator = modules.get('evaluator')
    prompt_sampler = modules.get('prompt_sampler')
    weights = list(getattr(llm, '_model_weights', None) or [])
    if len(weights) == 1:
      self._model_text = f'Model: `{weights[0][0]}`.'
    elif weights:
      self._model_text = (
          'Model drawn at random inside `generate()` from: '
          + ', '.join(f'`{m}` ({w:g})' for m, w in weights)
          + '.'
      )
    self._evaluator_name = type(evaluator).__name__

    self.run_dir = self._make_run_dir(config.name)
    self._overview = _MarkdownFile(self.run_dir / _OVERVIEW_FILE)
    for c in range(self._num_coroutines):
      self._file_for(c)

    # Seed programs: P0000, ...
    seeds = list(getattr(database, '_initial_population', None) or [])
    if not seeds:
      seeds = [database.get_best_program()]
    for seed in seeds:
      if self._programs.get(seed.text) is None:
        self._programs.add(
            seed.text,
            origin='seed',
            file=_OVERVIEW_FILE,
            parent=None,
            created_at=self._t0,
            scores=dict(seed.scores),
            status='evaluated',
        )

    self._write_overview_header(config, database, seeds, llm, evaluator,
                                prompt_sampler)
    for c in range(self._num_coroutines):
      self._write_coroutine_header(c)

    self._started = True
    atexit.register(self.close)  # Still write the summary on Ctrl+C / crash.
    print(f'[VerboseFileLogger] Writing verbose logs to {self.run_dir}/')

  def _make_run_dir(self, name: str) -> pathlib.Path:
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name) or 'run'
    base = f'{self._t0:%Y-%m-%d_%H-%M-%S}_{safe_name}'
    run_dir = self._log_root / base
    suffix = 1
    while run_dir.exists():
      suffix += 1
      run_dir = self._log_root / f'{base}_{suffix}'
    run_dir.mkdir(parents=True)
    latest = self._log_root / 'latest'
    try:  # Convenience only: `logs/latest` -> this run.
      if latest.is_symlink() or latest.exists():
        latest.unlink()
      latest.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
      pass
    return run_dir

  def _file_for(self, coroutine_index: int) -> _MarkdownFile:
    if coroutine_index not in self._files:
      self._files[coroutine_index] = _MarkdownFile(
          self.run_dir / _coroutine_file(coroutine_index)
      )
    return self._files[coroutine_index]

  def _file_links(self, exclude: str | None = None) -> str:
    names = [_OVERVIEW_FILE] + [
        _coroutine_file(c) for c in range(self._num_coroutines)
    ]
    return ' · '.join(
        f'**{n}** (this file)' if n == exclude else f'[{n}]({n})'
        for n in names
    )

  def _write_overview_header(
      self, config, database, seeds, llm, evaluator, prompt_sampler
  ) -> None:
    n = self._num_coroutines
    db_text = type(database).__name__
    num_islands = getattr(database, '_num_islands', None)
    reset_hours = getattr(database, '_island_reset_interval_hours', None)
    if num_islands is not None:
      db_text += f' · {num_islands} islands'
    if reset_hours is not None:
      db_text += f' · islands reset every {reset_hours:g} h'
    evaluator_text = type(evaluator).__name__
    for attr, fmt in (
        ('_image', 'image `{}`'),
        ('_timeout_seconds', 'timeout {} s'),
        ('_mem_limit', 'memory limit {}'),
    ):
      value = getattr(evaluator, attr, None)
      if value is not None:
        evaluator_text += ' · ' + fmt.format(value)
    weights = getattr(llm, '_model_weights', None) or []
    llm_text = type(llm).__name__ + ''.join(
        f' · `{m}` (weight {w:g})' for m, w in weights
    )
    config_rows = [
        ('config name', f'`{config.name}`'),
        (
            'main metric',
            f'`{self._main_metric}` (picks the best program at the end)',
        ),
        (
            'budget',
            f'{self._budget} steps (= LLM calls) in total, shared by all'
            ' coroutines',
        ),
        ('coroutines', f'{n} copies of `step_loop` (indices 0–{n - 1})'),
        ('database', db_text),
        ('prompt sampler', type(prompt_sampler).__name__),
        ('LLM', llm_text),
        ('evaluator', evaluator_text),
        ('logger', f'{type(self).__name__} (+ LocalLogger console output)'),
        ('process', f'Python {sys.version.split()[0]} · PID {os.getpid()}'),
    ]
    parts = [
        f'# AlphaEvolve run · `{config.name}`',
        f'Started **{_stamp(self._t0)}** · log directory `{self.run_dir}`',
        'Files: ' + self._file_links(exclude=_OVERVIEW_FILE),
        'This is the entry point of the log: the configuration, the seed'
        ' program, a timeline with one row per step (each linking to its'
        ' full record in a coroutine file) and, once the run ends, a summary'
        ' with the final database and every program that was created.',
        '## Configuration',
        _table(['setting', 'value'], config_rows),
        '## Problem description',
        _quote(config.problem_spec.problem_description),
        '## Seed program',
        'Read from `program_path` and evaluated once before the loops start.'
        ' Only the code between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END`'
        ' is shown to and edited by the LLM.',
    ]
    for seed in seeds:
      info = self._programs.get(seed.text)
      parts += [
          f'### {info.pid}',
          f'Scores: `{dict(seed.scores)}` · output_artifacts:'
          f' {"None" if seed.output is None else "yes"} ·'
          f' {_n_lines(seed.text)} lines',
          _fence(seed.text, 'python'),
      ]
    parts += [
        '## Database at the start',
        'Every island and the hall of fame start with the seed.',
        self._render_db(self._db_view()),
        '## Step timeline',
        'One row per step, added when the step ends, so rows are in'
        ' *finishing* order. The step link opens its full record. ✅'
        ' evaluated and registered · ❌ failed (still counts toward the'
        ' budget) · ⛔ cancelled when the run ended (does not count).',
    ]
    self._overview.blocks(*parts)

    headers = [
        'done', 'finished', 'step', 'sampled from', 'parent', 'child',
        f'{self._main_metric} ★', 'Δ parent', 'LLM wait', 'eval wait',
        'result',
    ]
    self._timeline_widths = [4, 12, 36, 13, 6, 6, 12, 10, 9, 9, 0]
    self._timeline_widths = [
        max(w, len(h)) for w, h in zip(self._timeline_widths, headers)
    ]
    self._overview.line(self._timeline_row(headers))
    self._overview.line(
        '|' + '|'.join('-' * (max(w, 3) + 2) for w in self._timeline_widths)
        + '|'
    )

  def _timeline_row(self, cells: Sequence[str]) -> str:
    return (
        '| '
        + ' | '.join(
            _cell(c).ljust(w) for c, w in zip(cells, self._timeline_widths)
        )
        + ' |'
    )

  def _write_coroutine_header(self, c: int) -> None:
    n = self._num_coroutines
    legend = [
        (
            '1 · Database before sampling',
            '`database.get_state()`',
            'the shared database as this step found it',
        ),
        (
            '2 · Sampling',
            '`database.sample_programs()`',
            'where the programs came from (hall of fame or island i), the'
            ' parent and the inspirations',
        ),
        (
            '3 · Prompt',
            '`prompt_sampler.sample_prompt()`',
            'the random template choices and the exact prompt text',
        ),
        (
            '4 · LLM response',
            '`await llm.generate_async(prompt)`',
            'how long this coroutine waited, and the raw reply',
        ),
        (
            '5 · Applying diffs',
            '`parsing_utils.apply_diffs()`',
            'each SEARCH/REPLACE edit and whether it matched; the child as a'
            ' diff and in full',
        ),
        (
            '6 · Evaluation',
            '`await evaluator.run_async(code)`',
            'how long this coroutine waited; scores vs. parent and record',
        ),
        (
            '7 · Registration',
            '`database.register_program(child)`',
            'where the child was stored, if anywhere',
        ),
        (
            '✅ / ❌ / ⛔ summary',
            '',
            'how the step ended, with a timing table of its phases',
        ),
    ]
    conventions = [
        '**Program IDs.** `P0000` is the seed; every new piece of code gets'
        ' the next ID, across the whole run. Identical code always has the'
        ' same ID. In the database tables, `P0007 1.23 ←P0003` means program'
        ' P0007, main-metric score 1.23, child of P0003. In a Markdown'
        ' preview, program links jump to the step that created the program.',
        '**Indices and counts.** Coroutine and island indices are 0-based, as'
        ' in the code. Counts start at 1: `c2-003` is the 3rd iteration of'
        ' coroutine 2. "Global step #9" is the 9th step to *start* in the'
        ' whole run; "budget 6 / 100" counts the steps that have *finished*'
        ' (`logger.total_steps`), successful or not.',
        '**Time.** Timestamps are local wall-clock time (HH:MM:SS.mmm). ⏳'
        ' marks an `await`: while this coroutine waits there, the event loop'
        ' runs the other coroutines, which is why their steps interleave.',
        '**★** marks the main metric.',
    ]
    self._file_for(c).blocks(
        f'# Coroutine {c} · AlphaEvolve run `{self._config.name}`',
        f'This file follows **coroutine {c}**, one of the {n} copies of'
        f' `step_loop` that `async_run` starts (indices 0–{n - 1}). They all'
        ' share one program database and one budget of'
        f' {self._budget} steps, so a child registered by any coroutine can'
        f' be sampled by all the others. Started {_stamp(self._t0)}.',
        'Files: ' + self._file_links(exclude=_coroutine_file(c)),
        '## How to read this file',
        f'Each pass through `step_loop` is one **`## Step c{c}-NNN`**'
        ' section. Jump between steps with the outline (VS Code:'
        ' Ctrl+Shift+O or the Outline panel), or list them with'
        f' `grep -n "^## Step" {_coroutine_file(c)}`. Every step has the same'
        ' parts, in the order the code runs them:',
        _table(['section', 'code', 'shows'], legend, max_pad=36),
        _bullets(conventions),
    )

  # ------------------------------------------------------------ step hooks --

  def begin_step(self, coroutine_index: int) -> base_logging.StepTrace:
    if not self._started or self._closed:
      return base_logging.StepTrace()
    try:
      return _VerboseStepTrace(self, coroutine_index)
    except Exception:  # pylint: disable=broad-exception-caught
      self._report_bug('begin_step', traceback_lib.format_exc())
      return base_logging.StepTrace()

  @_guarded
  def log_loop_end(self, coroutine_index: int) -> None:
    if not self._started or self._closed:
      return
    now = _now()
    text = (
        f'left `step_loop` at {_clock(now)}: budget used up'
        f' ({self.total_steps} / {self._budget})'
    )
    self._stats[coroutine_index].ended = text
    self._loop_ends.append(f'coroutine {coroutine_index} {text}')
    self._file_for(coroutine_index).blocks(
        '---',
        f'## Coroutine {coroutine_index} left `step_loop`',
        f'At {_clock(now)} the loop condition `logger.total_steps <'
        f' num_llm_calls` was false ({self.total_steps} / {self._budget}),'
        ' so this coroutine returned. That makes'
        ' `asyncio.wait(..., return_when=FIRST_COMPLETED)` in `async_run`'
        ' return, and the coroutines still in flight are cancelled.',
    )

  def _add_timeline_row(self, trace: _VerboseStepTrace, counted: bool) -> None:
    child = trace.child.pid if trace.child else '—'
    llm_wait = (
        _dur(trace._seconds('prompt', 'llm')) if 'llm' in trace.marks else '—'  # pylint: disable=protected-access
    )
    eval_wait = (
        _dur(trace._seconds('diff', 'eval')) if 'eval' in trace.marks else '—'  # pylint: disable=protected-access
    )
    parent_scores = trace.root.scores if trace.root else None
    self._overview.line(
        self._timeline_row([
            str(self.total_steps) if counted else '—',
            trace._time('end'),  # pylint: disable=protected-access
            f'[{trace.step_id}]({trace.file_name}#step-{trace.step_id})',
            trace.source,
            trace.parent.pid if trace.parent else '—',
            child,
            trace._main(trace.scores),  # pylint: disable=protected-access
            _delta(
                (trace.scores or {}).get(self._main_metric),
                (parent_scores or {}).get(self._main_metric),
            ),
            llm_wait,
            eval_wait,
            trace.result,
        ])
    )

  # -------------------------------------------------------- database views --

  def _db_view(self) -> _DbView:
    state = self._database.get_state() if self._database is not None else {}

    def entry(program):
      return _Entry(self._programs.of(program).pid, dict(program.scores))

    islands = state.get('islands')
    return _DbView(
        available=bool(state),
        hall_of_fame={
            metric: entry(p)
            for metric, p in state.get('best_programs_per_metric', {}).items()
        },
        islands=None
        if islands is None
        else [[entry(p) for p in island] for island in islands],
        last_reset_time=state.get('last_reset_time'),
    )

  def _entry_text(self, entry: _Entry) -> str:
    """'P0007 1.23 ←P0003': id, main score and parent of a stored program."""
    text = f'{entry.pid} {_score(entry.scores.get(self._main_metric))}'
    info = self._programs.by_pid.get(entry.pid)
    if info is not None and info.parent:
      text += f' ←{info.parent}'
    return text

  def _render_db(self, view: _DbView) -> str:
    """Renders a database snapshot (hall of fame + islands) as Markdown."""
    if not view.available:
      return (
          f'`{type(self._database).__name__}.get_state()` exposes nothing to'
          ' show.'
      )
    parts = [
        '**Hall of fame** · `_best_programs_per_metric`: the best program for'
        ' each metric',
        _table(
            ['metric', 'program', 'score'],
            [
                (
                    f'{m} ★' if m == self._main_metric else m,
                    e.pid,
                    _score(e.scores.get(m)),
                )
                for m, e in view.hall_of_fame.items()
            ],
        ),
    ]
    if view.islands is not None:
      rows = []
      for i, island in enumerate(view.islands):
        best = max(
            island,
            key=lambda e: _as_number(e.scores.get(self._main_metric)),
            default=None,
        )
        rows.append((
            i,
            len(island),
            self._entry_text(best).split(' ←')[0] if best else '—',
            ' · '.join(self._entry_text(e) for e in island) or '(empty)',
        ))
      parts += [
          f'**Islands** · `_islands`: {len(view.islands)} lists of programs,'
          ' oldest first, shown as `id score ←parent`',
          _table(['island', 'size', 'best ★', 'programs'], rows, max_pad=60),
      ]
    distinct = {e.pid for e in view.hall_of_fame.values()}
    for island in view.islands or []:
      distinct.update(e.pid for e in island)
    footer = f'{_plural(len(distinct), "distinct program")} in the database.'
    if view.last_reset_time is not None:
      footer += f' Islands last reset at {_clock(view.last_reset_time)}'
      hours = getattr(self._database, '_island_reset_interval_hours', None)
      if hours is not None:
        due = view.last_reset_time + datetime.timedelta(hours=hours)
        footer += (
            f'; next reset at the first `register_program` after'
            f' {_clock(due)}'
        )
      footer += '.'
    parts.append(footer)
    return '\n\n'.join(parts)

  def _source_label(self, root: evolution_database.Program) -> str:
    if root.sampled_from == 'islands' and root.island_index is not None:
      return f'island {root.island_index}'
    if root.sampled_from == 'best_programs_per_metric':
      return 'hall of fame'
    return 'best per metric'

  def _strategy_text(
      self, root: evolution_database.Program, view: _DbView
  ) -> str:
    if root.sampled_from == 'islands' and root.island_index is not None:
      i = root.island_index
      n = len(view.islands) if view.islands is not None else '?'
      size = (
          len(view.islands[i])
          if view.islands is not None and i < len(view.islands)
          else '?'
      )
      return (
          f'`sample_programs()` took the **island** branch (probability'
          f' 0.5), picked **island {i}** uniformly from {n} islands, then'
          f' temperature-sampled up to 5 of its {size} programs (T = 0.1, on a'
          ' randomly chosen metric: higher scores are much more likely). The'
          f' child will be appended to island {i}, whatever its score.'
      )
    if root.sampled_from == 'best_programs_per_metric':
      k = len({e.pid for e in view.hall_of_fame.values()})
      return (
          '`sample_programs()` took the **hall of fame** branch (probability'
          ' 0.5) and drew up to 5 programs uniformly from the best program per'
          f' metric ({_plural(k, "distinct program")} there). The child will'
          ' be kept only if it beats a metric record.'
      )
    return (
        f'`{type(self._database).__name__}.sample_programs()` drew up to 5'
        ' programs uniformly from the best program per metric.'
    )

  # ------------------------------------------------------------------ close --

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      if self._started:
        self._write_final_summary()
    except Exception:  # pylint: disable=broad-exception-caught
      self._report_bug('close', traceback_lib.format_exc())
    finally:
      for f in self._files.values():
        f.close()
      if self._overview is not None:
        self._overview.close()
      super().close()
    if self._started:
      print(f'[VerboseFileLogger] Logs written to {self.run_dir}/')

  def _write_final_summary(self) -> None:
    for trace in list(self._open_traces.values()):
      trace.never_finished()
    now = _now()
    final = self._db_view()

    # Per coroutine: a closing section with its statistics and the database.
    for c, f in self._files.items():
      s = self._stats[c]
      waited = sum(s.llm_waits) + sum(s.eval_waits)
      share = (
          f'LLM {100 * sum(s.llm_waits) / s.step_seconds:.1f} % · evaluator'
          f' {100 * sum(s.eval_waits) / s.step_seconds:.1f} % · everything'
          f' else {100 * (s.step_seconds - waited) / s.step_seconds:.1f} %'
          if s.step_seconds
          else '—'
      )
      f.blocks(
          '---',
          f'## End of coroutine {c}',
          _table(
              ['', ''],
              [
                  ('log closed', _stamp(now)),
                  ('how it ended', s.ended),
                  (
                      'iterations',
                      f'{s.iterations}: {s.ok} ✅ · {s.failed} ❌ ·'
                      f' {s.cancelled} ⛔'
                      + (f' · {s.unfinished} ⚠' if s.unfinished else ''),
                  ),
                  ('time in steps', _dur(s.step_seconds)),
                  ('waiting for the LLM', _wait_stats(s.llm_waits)),
                  ('waiting for the evaluator', _wait_stats(s.eval_waits)),
                  ('share of step time', share),
              ],
          ),
          '### Database at the end of the run',
          self._render_db(final),
      )

    # Overview: run summary, per-coroutine table, final database, programs.
    ok = self.step_stats.get('successful_steps', 0)
    failed = self.step_stats.get('failed_steps', 0)
    errors = ', '.join(f'{k} × {v}' for k, v in self.error_stats.items())
    cancelled = sum(s.cancelled + s.unfinished for s in self._stats.values())
    summary_rows = [
        ('finished', f'{_stamp(now)} ({_dur((now - self._t0).total_seconds())}'
         ' after the start)'),
        (
            'steps finished',
            f'{self.total_steps} / {self._budget}: {ok} ✅ · {failed} ❌'
            + (f' ({errors})' if errors else ''),
        ),
        ('steps cancelled at the end', str(cancelled)),
        ('LLM calls started', str(self._steps_started)),
    ]
    try:
      best = self._database.get_best_program()
      info = self._programs.of(best)
      summary_rows.append((
          f'best {self._main_metric} ★',
          f'{_score(best.scores.get(self._main_metric))} · {info.link}, made'
          f' by {info.origin_link}',
      ))
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    summary_rows.append(
        ('programs created', str(sum(
            1 for p in self._programs.all if p.origin not in ('seed',)
        )))
    )
    if self._loop_ends:
      summary_rows.append(('loop ends', ' · '.join(self._loop_ends)))
    if self._logger_bugs:
      summary_rows.append(
          ('⚠ logger bugs', ', '.join(self._logger_bugs) + ' (see stderr)')
      )

    coroutine_rows = []
    for c in sorted(self._files):
      s = self._stats[c]
      coroutine_rows.append((
          f'[{c}]({_coroutine_file(c)})',
          s.iterations,
          s.ok,
          s.failed,
          s.cancelled + s.unfinished,
          f'{_dur(sum(s.llm_waits))} total · '
          + (_dur(sum(s.llm_waits) / len(s.llm_waits)) if s.llm_waits else '—')
          + ' mean',
          f'{_dur(sum(s.eval_waits))} total · '
          + (_dur(sum(s.eval_waits) / len(s.eval_waits))
             if s.eval_waits else '—')
          + ' mean',
          s.ended,
      ))

    places = collections.defaultdict(list)
    for metric, e in final.hall_of_fame.items():
      places[e.pid].append(f'🏆 {metric}')
    for i, island in enumerate(final.islands or []):
      for pid in dict.fromkeys(e.pid for e in island):
        places[pid].append(f'island {i}')
    program_rows = [
        (
            p.link,
            p.origin_link,
            _clock(p.created_at),
            p.parent or '—',
            _score((p.scores or {}).get(self._main_metric))
            if p.scores is not None
            else p.status,
            ', '.join(places.get(p.pid, [])) or '—',
        )
        for p in self._programs.all
    ]

    self._overview.blocks(
        '## Final summary',
        _table(['', ''], summary_rows),
        '### Per coroutine',
        _table(
            ['coroutine', 'iterations', '✅', '❌', '⛔', 'LLM wait',
             'eval wait', 'how it ended'],
            coroutine_rows,
        ),
        '### Database at the end of the run',
        self._render_db(final),
        '### All programs',
        'Every distinct program seen in this run, in order of creation,'
        ' including those that were never stored or failed evaluation.',
        _table(
            ['program', 'made by', 'at', 'parent', f'{self._main_metric} ★',
             'in the database at the end'],
            program_rows,
        ),
    )

  # ----------------------------------------------------------------- errors --

  def _report_bug(self, where: str, tb: str) -> None:
    self._logger_bugs.append(where)
    print(
        f'[VerboseFileLogger] internal error in {where} (the run continues):'
        f'\n{tb}',
        file=sys.stderr,
    )
