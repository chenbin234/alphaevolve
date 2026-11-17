# How this AlphaEvolve repository works

This guide covers how the repository is laid out, how its components connect so
that an LLM takes part in the search loop, how the program database chooses
what the LLM sees, and how and why the code uses `asyncio`. File references point at the code as it stands on `main`.

---

## 1. Repository layout

| Path | What it is | Runs the loop? |
|---|---|---|
| `src/` | A minimal, working AlphaEvolve: the evolutionary loop, LLM client, Docker sandbox, program database. | **Yes**, the only executable part. |
| `src/data/` | A small example problem: circle packing with 28 circles. The config is `problem_config/circle_packing.yaml` and the seed program is `initial_programs/circle_packing.py`. | Input to `src/` |
| `maths/<problem>/` | Each problem from the paper has a `config.yaml` and an `initial_program.py` that `src/run.py` can use, plus a Colab `verification.ipynb` that checks the published result. | Input to `src/` |
| `omega/` | The matrix multiplication exponent result: an evolved JAX optimiser (`level_3/final_program.py`), a verifier and a rigorous verification notebook. | No, it's standalone. |
| `matmul_tensor_decompositions/` | Notebooks containing the discovered tensor decompositions, with verifiers. | No, it's standalone. |

The rest of this guide covers `src/`, which is the system itself.

```
src/
├── run.py                     CLI entry point
├── alpha_evolve.py            the evolutionary loop (async)
├── requirements.txt           packages installed into the Docker sandbox image
└── components/
    ├── configuration.py       YAML → typed Configuration dataclasses
    ├── evolution_database.py  Program record + population/selection strategies
    ├── prompting.py           builds the prompt sent to the LLM
    ├── llm_client.py          Gemini client (sync generate + async wrapper)
    ├── parsing_utils.py       EVOLVE-BLOCK parsing + SEARCH/REPLACE diff application
    ├── evaluator_client.py    runs candidate programs in a locked-down Docker container
    ├── logging.py             step/error counters, best scores, console output
    └── exceptions.py          recoverable error types the loop catches
```

---

## 2. The big picture

AlphaEvolve is an **evolutionary algorithm whose mutation operator is an LLM**.
The four classic parts of an evolutionary algorithm map onto modules like this:

| Evolutionary concept | Module in this repo |
|---|---|
| Population + selection | `evolution_database` (`GreedyDatabase`, `GreedyAndIslandsDatabase`) |
| Variation (mutation / crossover) | `prompting` + `llm_client` + `parsing_utils`. The LLM reads parent programs and proposes edits. |
| Fitness function | `evaluator_client` runs the program's own `evaluate()` in Docker. |
| Bookkeeping / budget | `logging` |

### 2.1 Four loops, one database

`src/data/problem_config/circle_packing.yaml` sets the budget:

```yaml
budget:
  num_llm_calls: 100   # total steps across ALL loops, not 100 per loop
  num_coroutines: 4    # how many copies of step_loop run at the same time
```

`async_run` starts **4 copies of the same loop**, `step_loop`
([alpha_evolve.py:122](src/alpha_evolve.py#L122)). Each copy does one step at a
time, and the four copies run side by side. They share one program database and
one step counter:

```
┌───────────────── Program database: ONE, shared by all 4 loops ─────────────────┐
│  best program for each metric ("hall of fame")     + 5 islands of programs     │
└─┬─────────────▲──────┬─────────────▲──────┬─────────────▲──────┬─────────────▲─┘
  │ parent      │      │ parent      │      │ parent      │      │ parent      │
  │ + insp.     │      │ + insp.     │      │ + insp.     │      │ + insp.     │
  ▼       child │      ▼       child │      ▼       child │      ▼       child │
┌── coroutine 1 ┴─┐  ┌── coroutine 2 ┴─┐  ┌── coroutine 3 ┴─┐  ┌── coroutine 4 ┴─┐
│ 1 sample        │  │ 1 sample        │  │ 1 sample        │  │ 1 sample        │
│ 2 build prompt  │  │ 2 build prompt  │  │ 2 build prompt  │  │ 2 build prompt  │
│ 3 await LLM     │  │ 3 await LLM     │  │ 3 await LLM     │  │ 3 await LLM     │
│ 4 apply diffs   │  │ 4 apply diffs   │  │ 4 apply diffs   │  │ 4 apply diffs   │
│ 5 await Docker  │  │ 5 await Docker  │  │ 5 await Docker  │  │ 5 await Docker  │
│ 6 register      │  │ 6 register      │  │ 6 register      │  │ 6 register      │
│   repeat from 1 │  │   repeat from 1 │  │   repeat from 1 │  │   repeat from 1 │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │                    │
┌────────┴────────────────────┴────────────────────┴────────────────────┴────────┐
│ logger.total_steps, also shared: one pass through 1-6 is ONE step (one LLM     │
│ call) and adds 1, whether it succeeded or failed. Each loop checks it before   │
│ every step and stops at 100.                                                   │
└────────────────────────────────────────────────────────────────────────────────┘
```

How to read it:

- **▼ (sample):** one `sample_programs()` call returns a parent plus up to 4
  inspirations. All of them go into **one** prompt, which gets **one** LLM call
  and produces **one** child. The inspirations are context only: the diff is
  applied to the parent alone.
- **▲ (register):** `register_program(child)` puts the child into the shared
  database. Any loop, not just the one that made it, can pick it on its next ▼.
- **Inside a box, everything is serial.** A loop never has two steps in flight.
  It also keeps no memory between steps: it has no lineage and no island of its
  own, and every ▼ is a fresh draw from the database as it is at that moment.
- **Across boxes, steps overlap.** Up to 4 steps are in flight at once, so up to
  4 Gemini requests and 4 Docker containers can be running at the same time.
- **The counter is shared.** The 100 steps are the total for all four loops.
  Faster loops simply do more of them.

### 2.2 Inside the program database

The top box in 2.1 is a single Python object, `GreedyAndIslandsDatabase`, held
in memory. It stores `Program` records:

```
One entry in the database = one Program record (evolution_database.py:31):
┌────────────────────────────────────────────────────────────────────────────┐
│ text          the full source code, including evaluate()                   │
│ scores        {"sum_of_radii": 1.41}, from scores_to_maximize              │
│ output        output_artifacts from evaluate(), or None                    │
│ sampled_from  ┐ both None while stored. sample_programs() fills them in    │
│ island_index  ┘ on the copies it returns, as the child's return address.   │
└────────────────────────────────────────────────────────────────────────────┘
```

Here is a made-up snapshot of a circle-packing run after ten successful steps.
It continues the worked example in section 5.7. The letters are names for this
example, and `<-A` means "made from parent A".

```
┌──────────── GreedyAndIslandsDatabase: one Python object, in memory ────────────┐
│                                                                                │
│ _best_programs_per_metric: the "hall of fame", a dict with one key per metric  │
│ ┌──────────────────────────────────────────────────────────────────────────┐   │
│ │ "sum_of_radii" ──► F 1.41                                                │   │
│ └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│ _islands: a list of 5 lists. Children are only ever appended.                  │
│ ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐   │
│ │ island 0     │ island 1     │ island 2     │ island 3     │ island 4     │   │
│ ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤   │
│ │ S 0.96       │ S 0.96       │ S 0.96       │ S 0.96       │ S 0.96       │   │
│ │ G 1.02 <-S   │              │ A 1.20 <-S   │ H -inf <-S   │ D 1.05 <-S   │   │
│ │              │              │ C 0.50 <-A   │ I 1.10 <-S   │ J 1.12 <-D   │   │
│ │              │              │ E 1.33 <-A   │              │              │   │
│ │              │              │ F 1.41 <-E   │              │              │   │
│ └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘   │
│                                                                                │
│ _last_reset_time: when the run started, or the last reset. Two hours later     │
│ the islands are rebuilt.                                                       │
└────────────────────────────────────────────────────────────────────────────────┘
  Not stored anywhere: B 1.10 <-A (a hall-of-fame step that did not beat the
  record) and every failed step.
```

Reading the snapshot:

- **The letters and the `<-` marks are only for reading.** The database stores
  no names, ids or parent links. It knows only each program's text, scores and
  output.
- **`S` is on every island.** The constructor put a copy of the seed on each
  one. Island 1 hasn't produced a child yet, so the seed is all it has.
- **Island 2 has been the busiest.** It has the most programs and produced the
  current champion `F`. Its lineage runs S → A → E → F, so `F` is the seed's
  great-grandchild.
- **`H` scored `-inf` (an invalid packing) and is still kept.** Islands never
  reject a child. They just rarely pick a weak one.
- **`F` is stored twice:** on island 2, because its parent came from there, and
  in the hall of fame, because it set a record. The earlier record holders `S`,
  `A` and `E` lost their hall-of-fame slot but still live on their islands.
- **`B` is nowhere.** It came from a hall-of-fame step and didn't beat the
  record, so it was thrown away.

Each step reads the database once and writes to it once:

```
READ: sample_programs()                  WRITE: register_program(child)
───────────────────────                  ──────────────────────────────
coin flip                                the child carries its parent's tag
├─ heads: hall of fame ──► [F]           ├─ tag "island 2":
│    parent F, no inspirations           │    append to island 2, and also
└─ tails: 1 of the 5 islands, at random  │    copy to hall of fame if > 1.41
     island 2 ──► [F, E, A, S, C]        └─ tag "hall of fame":
       parent F 66%, E 27%, A 7%              copy to hall of fame if > 1.41,
       (the other 4 are inspirations)         otherwise throw it away
     island 1 ──► [S]
       parent S, no inspirations
```

The odds for island 2 come from running the real `_temperature_sample` on these
five scores. With only one metric, the hall of fame always holds exactly one
program, so a heads step never has inspirations. Section 5 covers the rules in
full.

### 2.3 The four loops over time

LLM replies and evaluations take different amounts of time, so the loops soon
drift out of phase:

```
time ────────────────────────────────────────────────────────────────────────────►
coroutine 1  S P[ LLM ····· ]A[ Docker ······· ]R S P[ LLM ········ ]A[ Docker ···
coroutine 2  S P[ LLM ········· ]A[ Docker ·········· ]R S P[ LLM ··· ]A[ Docker ·
coroutine 3  S P[ LLM ··· ]A[ Docker ························ ]R S P[ LLM ········
coroutine 4  S P[ LLM ····· ]A S P[ LLM ······· ]A[ Docker ········ ]R S P[ LLM ··
                             ▲ b                ▲ a
```

`S` = sample, `P` = build prompt, `A` = apply diffs, `R` = register. These are
plain Python calls on the event-loop thread and finish almost instantly. The
bracketed parts are `await`s. While one loop waits there, the event loop runs
the others.

- **a:** coroutine 1 registers its child. The next `S` of coroutines 2 and 3
  comes after this point, so either may get that child as a parent or as an
  inspiration. Coroutine 4's second `S` came before it, so it could not.
- **b:** coroutine 4's diff did not apply. The step counts as failed, nothing is
  stored, and the loop starts its next step at once.
- **At the very start**, all four loops sample a database that holds only the
  seed program. The first four steps are therefore four independent edits of the
  same parent, with no inspirations.
- **At the end**, the run stops as soon as one loop finds the counter at 100.
  The other loops are cancelled mid-step and their unfinished children are
  thrown away (section 7.3).

### 2.4 Zoom in: one step inside one loop

Each box in 2.1 runs this pipeline once per step:

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                   Program database                       │
                 │  best-per-metric programs  +  N islands of programs      │
                 └───────────────┬──────────────────────────────▲───────────┘
            sample_programs()    │ parent + up to 4              │ register_program(child)
                                 │ "inspiration" programs        │
                                 ▼                               │
                 ┌──────────────────────────────┐                │
                 │ Prompt sampler               │                │
                 │ problem description          │                │
                 │ + prior programs & scores    │                │
                 │ + current program            │                │
                 │ + SEARCH/REPLACE rules       │                │
                 │ + a randomly chosen task     │                │
                 └───────────────┬──────────────┘                │
                         prompt  │                               │
                                 ▼                               │
                 ┌──────────────────────────────┐                │
                 │ LLM (Gemini)                 │  ◄── await (thread)
                 │ returns SEARCH/REPLACE diffs │                │
                 └───────────────┬──────────────┘                │
                    llm_sample   │                               │
                                 ▼                               │
                 ┌──────────────────────────────┐                │
                 │ apply_diffs(parent, sample)  │                │
                 │ edits only EVOLVE-BLOCK code │                │
                 └───────────────┬──────────────┘                │
                    child code   │   + parent's output_artifacts │
                                 ▼                               │
                 ┌──────────────────────────────┐                │
                 │ Docker evaluator             │  ◄── await (thread)
                 │ runs evaluate() in a sandbox │                │
                 └───────────────┬──────────────┘                │
                                 │ scores_to_maximize,           │
                                 │ output_artifacts              │
                                 └───────────────────────────────┘
```

Section 5 explains how the database picks and stores programs. Section 7
explains the async machinery.

---

## 3. Startup: from the command line to the loop

**`src/run.py`** ([main](src/run.py#L29))

1. It parses `--problem_config`, which defaults to
   `src/data/problem_config/circle_packing.yaml`.
2. It resolves the config path relative to your current directory first, then
   relative to `src/`. After that it runs **`os.chdir(src/)`**
   ([run.py:54](src/run.py#L54)), because the paths inside the configs are
   relative to `src/`:
   - `program_path: "data/initial_programs/circle_packing.py"`
   - `program_path: "../maths/cap_sets/initial_program.py"`
   - `DockerEvaluator(requirements_path='requirements.txt')` → `src/requirements.txt`
3. It calls `configuration.from_yaml(...)` and then
   `asyncio.run(alpha_evolve.async_run(config))` ([run.py:60](src/run.py#L60)).
   This is the single point where synchronous code hands over to the event loop.

**`components/configuration.py`** ([from_yaml](src/components/configuration.py#L120))
turns the YAML into frozen dataclasses:

```yaml
name: "circle_packing"
problem_spec:
  program_path: ...          # read from disk and stored as problem_spec.initial_program
  problem_description: ...   # natural-language task, put at the top of every prompt
  main_metric_name: ...      # which score picks the final "best program"
budget:
  num_llm_calls: 100         # total steps (successful + failed)
  num_coroutines: 4          # how many loops run concurrently
modules:                     # each name must be a value of an Enum in configuration.py
  database: GreedyAndIslandsDatabase
  prompt_sampler: TemplatePromptSampler
  llm: GenAILLMClient
  evaluator: DockerEvaluator
  logger: LocalLogger
```

**`alpha_evolve.async_run`** ([alpha_evolve.py:94](src/alpha_evolve.py#L94))

1. It looks up each class in the registries `_DATABASES`, `_PROMPT_SAMPLERS`,
   `_LLM_CLIENTS`, `_EVALUATORS` and `_LOGGERS`
   ([alpha_evolve.py:31](src/alpha_evolve.py#L31)) and instantiates it. This is
   the **plug-in mechanism**. To add a new LLM provider, you subclass
   `BaseLLMClient`, add a value to `LLMType` and add an entry to `_LLM_CLIENTS`.
2. It evaluates the **initial program synchronously**
   ([alpha_evolve.py:105](src/alpha_evolve.py#L105)). This call isn't inside a
   `try`, so a broken seed program aborts the run straight away.
3. It seeds the database with that single scored program.
4. It starts `num_coroutines` copies of `step_loop` in an `asyncio.TaskGroup` and
   stops when the first one finishes (see section 7.3).
5. It returns `database.get_best_program()`, which is the best program by
   `main_metric_name`.

---

## 4. The components and how they connect

### 4.1 `Program`, the unit of evolution ([evolution_database.py](src/components/evolution_database.py#L31))

```python
Program(text,          # full source code of the candidate
        scores,        # dict metric -> float, from evaluate()["scores_to_maximize"]
        output,        # evaluate()["output_artifacts"], e.g. the best solution found
        sampled_from,  # 'islands' | 'best_programs_per_metric' | None
        island_index)  # which island the parent came from
```

`Program.render()` ([line 53](src/components/evolution_database.py#L53)) is
**what the LLM actually sees** for each program. It produces
`Obtained scores: {...}` followed by a fenced block that contains **only the code
between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`**. Everything outside
those markers, including `evaluate()` and the scoring function, stays hidden from
the LLM. That's why the `maths/*/config.yaml` files repeat the scoring function
inside `problem_description`.

### 4.2 Database: population and selection ([evolution_database.py](src/components/evolution_database.py))

- **`GreedyDatabase`** keeps one program per metric, namely the best one seen.
  `sample_programs(k=5)` draws up to 5 distinct programs at random from that set.
- **`GreedyAndIslandsDatabase`**, the default, combines two strategies:
  - With probability 0.5, it samples from the **best-per-metric** set, which
    favours exploitation.
  - Otherwise, it picks a random **island** (5 by default) and draws programs
    from it with a **temperature softmax** over a randomly chosen metric
    (temperature 0.1). Scores are min-max normalised first.
  - A child is appended to the island its parent came from. That's why
    `sampled_from` and `island_index` are copied from parent to child in
    `step_loop`. Islands evolve independently, which keeps the population
    diverse.
  - Every `island_reset_interval_hours` (2 hours by default), all islands are
    rebuilt from each old island's best program plus a few random survivors
    (section 5.6).
  - Every child is also offered to the best-per-metric set, and it enters only if
    it beats the current best on some metric. Island membership has no such
    test (section 5.5).

`sample_programs` returns a list. `step_loop` treats the first element as the
**parent** ("current program") and the rest as **inspirations** ("prior
programs").

Section 5 walks through the default database step by step, with worked
numbers.

### 4.3 Prompt sampler: turning the population into a prompt ([prompting.py](src/components/prompting.py#L146))

`TemplatePromptSampler.sample_prompt` fills `PROMPT_TEMPLATE` with:

1. `problem_description` from the config,
2. **Prior programs**: the rendered inspirations, each showing its scores and
   with duplicates removed,
3. **Current program**: the rendered parent,
4. the **SEARCH/REPLACE** format rules plus two examples,
5. three **randomly sampled** phrases from `PROMPT_FORMAT_ARGS`:
   - `task_instruction`: one of 5 styles, each with probability 0.2, ranging
     from "combine the strengths of all the programs above" to "Make it crazy
     with a capital C",
   - `focus_sentence`: "Focus on simplifying the code…" with probability 0.2,
   - `trigger_chain_of_thought`: with probability 0.5, it asks the LLM to explain
     its reasoning before writing code.

The random phrasing is a cheap source of **diversity**. Identical prompts would
produce similar mutations.

### 4.4 LLM client ([llm_client.py](src/components/llm_client.py))

- `BaseLLMClient.generate(prompt)` is synchronous and abstract.
- `BaseLLMClient.generate_async(prompt)` is the async wrapper used by the loop
  (section 7).
- `GenAILLMClient` calls Gemini through `google-genai` and needs
  `GEMINI_API_KEY`. It supports a **weighted model ensemble** through
  `model_weights`: each call samples a model, so you can mix a fast, cheap model
  with a strong, slow one. The default is `(("gemini-3.8-flash", 1.0),)`.
- Errors from `genai.errors.APIError` and `ValueError` are turned into
  `LLMInferenceError`, which the loop treats as a failed step.

### 4.5 Diff parsing: turning LLM text into a new program ([parsing_utils.py](src/components/parsing_utils.py#L241))

`apply_diffs(parent_code, llm_sample)` works in four steps:

1. It splits the parent into alternating **immutable** and **mutable** blocks
   using the `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers.
2. It extracts every ```` ```python ... ``` ```` fence from the LLM response.
3. Inside each fence, it finds every
   `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` triple and turns it into an
   `(old, new)` pair.
4. It applies every pair to **every mutable block** with `str.replace(old, new, 1)`.
   Immutable blocks are never touched, so the LLM can't change `evaluate()` and
   can't game the score by editing the scorer.

### 4.6 Evaluator: the fitness function ([evaluator_client.py](src/components/evaluator_client.py))

`DockerEvaluator`:

- **Building the image:** on first use, it builds `alphaevolve-sandbox:<hash of
  requirements.txt>` from `python:3.11` plus `src/requirements.txt`, and caches
  it by that hash.
- **Wrapping the script** ([_build_script](src/components/evaluator_client.py#L238)):
  1. It defines `PARENT_OUTPUT` from the parent's `output_artifacts`, which are
     JSON-serialised.
  2. It pastes in the candidate code.
  3. It calls `evaluate()`.
  4. It prints a separator line followed by `json.dumps(result)`.
- **Sandbox settings** ([_run](src/components/evaluator_client.py#L160)):
  - no network, read-only root filesystem, every Linux capability dropped,
    `no-new-privileges`
  - 512 MB of RAM, 100 PIDs, 1 CPU, a 100 MB `/tmp` tmpfs
  - single-threaded BLAS
  - a `timeout --signal=KILL` wrapper plus `container.wait(timeout=...)`
- **Reading the result:** it parses whatever follows the last separator. The
  result must be a mapping that contains `scores_to_maximize`. Any failure is
  turned into `CodeExecutionError`.

The evaluator returns `(scores_to_maximize, output_artifacts)`.

### 4.7 Logger ([logging.py](src/components/logging.py))

The logger counts `successful_steps` and `failed_steps`, counts errors by type,
tracks the best score for each metric and prints every step. **Its
`total_steps` property is also the loop's budget counter**: `step_loop` keeps
going while `logger.total_steps < num_llm_calls`.

---

## 5. The program database in depth

Think of the database as the search's **memory and its coach**. The LLM has no
memory: every call starts from scratch and knows only what's in the prompt. The
database remembers every program worth keeping and, at each step, chooses which
few programs the LLM gets to look at. What the LLM sees shapes what it writes
next, so the database steers the whole search.

This section explains
[`GreedyAndIslandsDatabase`](src/components/evolution_database.py#L141), which
every config in this repo uses. It covers the simpler `GreedyDatabase` at the
end.

### 5.1 The problem it solves: exploitation vs. exploration

Any search has to balance two goals:

- **Exploitation:** keep improving the best thing you have. This is fast, but
  you can get stuck on a "local optimum", a solution that small changes can't
  improve even though a much better one exists elsewhere.
- **Exploration:** keep trying other directions, including some that look worse
  right now, because one of them might lead somewhere better.

The database handles each goal with its own structure:

```
┌─────────────────────── GreedyAndIslandsDatabase ───────────────────────┐
│                                                                        │
│  Hall of fame (exploitation)        Islands (exploration)              │
│  _best_programs_per_metric          _islands: 5 separate lists         │
│  ┌──────────────────────────┐       ┌────────┐┌────────┐   ┌────────┐  │
│  │ "sum_of_radii" → prog A  │       │island 0││island 1│...│island 4│  │
│  │ (one record per metric)  │       │ S, B, D││ S, C   │   │ S      │  │
│  └──────────────────────────┘       └────────┘└────────┘   └────────┘  │
│                                                                        │
│  _last_reset_time: a clock used to mix the islands every 2 hours       │
└────────────────────────────────────────────────────────────────────────┘
```

- **The hall of fame** is a dictionary from each metric to the best program
  found for it. It's like a trophy cabinet: one record holder per category.
- **The islands** are 5 independent populations, like 5 research teams that each
  keep their own notebook and don't share ideas. Because they evolve separately,
  they tend to find different approaches, which keeps the search diverse.

### 5.2 The start: one seed everywhere

`async_run` creates the database like this
([alpha_evolve.py:108](src/alpha_evolve.py#L108)):

```python
database_cls = _DATABASES[config.modules.database]   # class named in the YAML
database = database_cls(
    initial_population=[
        evolution_database.Program(
            text=config.problem_spec.initial_program,  # source of the seed program
            scores=initial_scores,                     # from the evaluation just before
            output=output,                             # its output_artifacts
        )
    ],
    main_metric_name=config.problem_spec.main_metric_name,
)
```

The starting population is a single program, the seed `S`. The constructor
([evolution_database.py:144](src/components/evolution_database.py#L144)) puts
it in:
- the hall of fame, where it's the record holder for every metric since it's the
  only program,
- every one of the 5 islands.

So all 5 teams start from the same program. They drift apart as the LLM makes
different random changes on each island. The seed also guarantees the database
is never empty. That matters because `step_loop` unpacks the first sampled
program as the parent, and island sampling raises an error if any island is
empty.

The seed must be scored before it goes in, because the database compares and
samples programs by score. `main_metric_name` (for example `sum_of_radii`)
doesn't change how the search runs: the database tracks the best program for
*every* metric. The name only decides which program `get_best_program()`
returns at the end.

### 5.3 Picking programs for a step: `sample_programs()`

Every step starts with a **coin flip** (`random.random() < 0.5`) in
[`sample_programs`](src/components/evolution_database.py#L189).

**Heads (50%): use the hall of fame.**
It takes up to 5 distinct programs from the hall of fame, uniformly at random.
Each is tagged `sampled_from='best_programs_per_metric'`.

**Tails (50%): use an island.**
1. It picks one of the 5 islands uniformly at random.
2. It draws up to 5 programs from that island with **temperature sampling**
   (section 5.4), which favours higher scores.
3. Each program is tagged `sampled_from='islands'` and
   `island_index=<that island>`.

`step_loop` treats the first program returned as the **parent**, which the LLM
is asked to modify. The rest are **inspirations**, shown in the prompt as
"Prior programs".

The tags work like a **return address**. The child inherits its parent's tags,
so when it comes back from evaluation, the database knows where it belongs.

### 5.4 Temperature sampling: picking good programs without always picking the best

[`_temperature_sample`](src/components/evolution_database.py#L221) uses a
softmax with temperature. Here it is with numbers. Say an island holds four
programs with these scores:

| Program | Score |
|---|---|
| P1 | 2.0 |
| P2 | 2.3 |
| P3 | 2.5 |
| P4 | 2.6 |

**Step 1: normalise to the range 0–1.** Subtract the minimum (2.0) and divide
by the largest difference (0.6). The scores become `[0, 0.5, 0.83, 1.0]`, so
the worst is 0 and the best is 1.

**Step 2: divide by the temperature** (T = 0.1). This gives logits
`[0, 5, 8.3, 10]`.

**Step 3: apply the softmax**, which turns the logits into probabilities:
pᵢ = e^(logitᵢ) / Σ e^(logit).

The result depends a lot on the temperature:

| | P1 (2.0) | P2 (2.3) | P3 (2.5) | P4 (2.6) |
|---|---|---|---|---|
| **T = 0.1** (used here) | ≈ 0% | 0.6% | 16% | **84%** |
| T = 1 (for comparison) | 13% | 22% | 30% | 36% |

Temperature acts as a **greediness knob**:
- A **low temperature** almost always picks the best program.
- A **high temperature** picks nearly at random.

At 0.1 this code is fairly greedy, but the runner-up still becomes the parent
about 1 time in 6. That keeps some exploration going.

Three details:
- **Draws are without replacement**, up to 5 programs. On a small island like
  this one, all 4 programs end up in the prompt. The probabilities mainly decide
  **which one is first**, and the first one becomes the parent.
- **Every program gets a tiny extra probability** (`+ 1e-6`). Even a broken
  program with a score of −∞ can, very rarely, be picked.
- **Multiple metrics:** if `evaluate()` returns several metrics, one is chosen
  at random for each draw. This optimises all metrics in turn. Circle packing
  has only one metric, so this doesn't apply.

### 5.5 Placing a child: `register_program(child)`

**No new islands are ever created.** The database is built with 5 islands
(`num_islands=5`,
[evolution_database.py:148](src/components/evolution_database.py#L148)) and
keeps exactly 5 for the whole run. This section is about when a *program* is
added to an island, and when it enters the hall of fame. The two follow
different rules. "Beat the record" is the hall-of-fame rule; islands have no
score check at all.

Only children that evaluated without an error get this far. A failed step (an
LLM error, an unusable diff, a crash in Docker) never reaches the database: the
loop logs the error and moves on.

[`register_program`](src/components/evolution_database.py#L166) does three
things in order:

```python
def register_program(self, program):
    self._register_to_best_programs_per_metric(program)            # A: hall of fame
    if program.sampled_from == 'islands' and program.island_index is not None:
        self._register_to_islands(program, program.island_index)   # B: island
    self._maybe_reset_islands()                                     # C: reset (5.6)
```

**A. Hall of fame: beat the record**
([lines 172-180](src/components/evolution_database.py#L172)).
- For each metric, the child takes over that metric's slot only if its score is
  **strictly higher** than the current record holder's. A tie isn't enough, and
  a `None` score is skipped.
- A record on **any** metric is enough, not just `main_metric_name`. With
  several metrics, the hall of fame holds up to one program per metric.
- It doesn't matter where the parent came from. Every child gets this chance.

**B. Island: no score check**
([lines 182-186](src/components/evolution_database.py#L182)).
- The child goes to an island only if its parent was sampled from an island.
  `step_loop` copies the parent's `sampled_from` and `island_index` onto the
  child ([alpha_evolve.py:87](src/alpha_evolve.py#L87)). This is the return
  address from section 5.3.
- The child is then appended to **its parent's island, whatever it scored.** A
  worse child is kept. So is an exact copy of the parent, which happens when the
  LLM's edits changed nothing (gotcha 2 in section 8).
- An island has no size limit and no duplicate check. Only a reset shrinks it.

So where the child ends up depends only on these two questions:

| Parent sampled from | Child beats a record? | Where the child ends up |
|---|---|---|
| island *i* | yes | hall of fame **and** island *i* |
| island *i* | no | island *i* only |
| hall of fame | yes | hall of fame only |
| hall of fame | no | **nowhere**: the child is thrown away |

In the snapshot in section 2.2, `F` is the first row, `C` and `H` are the
second, and `B` is the fourth.

Why it's designed this way:
- **Each island is a separate lineage.** A child always stays on its parent's
  island, so the 5 islands evolve independently and don't converge on the same
  solution. Programs move between islands only at a reset.
- **Islands keep weak programs on purpose,** for diversity. They never delete
  anyone between resets. Weak programs stay but are rarely chosen as parents,
  because temperature sampling at T = 0.1 favours strong ones (section 5.4).
  Selection pressure on islands comes from *who gets picked*, not from *who
  gets removed*.
- **The hall of fame is strict hill climbing:** beat the champion or be
  forgotten. Half of all steps start from it.

**Seeing it in the logs.** With `logger: VerboseFileLogger`, every step in
`logs/latest/coroutine_K.md` records this decision:
- **section 2 · Sampling** says whether the parent came from an island or the
  hall of fame,
- **section 7 · Registration** says where the child went (for example
  "Appended P0012 to island 3", a new hall-of-fame record, or "Not stored
  anywhere" with the reason),
- **section 1** is the database snapshot before the step, so you can watch the
  islands grow.

### 5.6 Island reset: mixing the islands every 2 hours

After every registration,
[`_maybe_reset_islands()`](src/components/evolution_database.py#L252) checks
the clock. Once `island_reset_interval_hours` (2 by default) have passed,
[`_reset_islands`](src/components/evolution_database.py#L259) does three things:

1. From each island, it keeps up to 5 survivors, using temperature sampling at
   T = 1e-6. The first pick is always the island's best. The other picks are
   random (see below).
2. It merges all survivors and removes duplicates.
3. It deals them back into 5 fresh islands round-robin, like dealing cards.

```
before:  island 0: 40 programs   island 1: 35 programs   ...
           ↓ keep ≤5 each          ↓ keep ≤5 each
after:   each new island gets a mix of survivors from different old islands
```

This is called **migration** in evolutionary algorithms. Most programs are
removed, so the islands stay small, and each island's best spreads to the
others, while each island still starts its next phase from varied material.

**What a reset really keeps.** The docstring of `_reset_islands` says "top k
programs per island", but only the first survivor is chosen by score. I ran the
real `_temperature_sample` 20,000 times on one island of 10 programs scored 0
to 9, drawing 5 at T = 1e-6:

| Program's score | 9 (best) | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 (worst) |
|---|---|---|---|---|---|---|---|---|---|---|
| How often it was kept | **100%** | 45% | 44% | 45% | 44% | 45% | 44% | 44% | 45% | 44% |

The cause is the `+ 1e-6` on
[line 239](src/components/evolution_database.py#L239):
- At T = 1e-6, the softmax gives the best program essentially all the
  probability and every other program essentially 0.
- The `+ 1e-6` then lifts all of those near-zero values to the same tiny
  number.
- Draws are without replacement. Once the best program is drawn, the rest all
  have equal odds, so each of the other 9 has a 4-in-9 (≈44%) chance.

So a reset keeps **each island's best plus 4 random programs**, not its top 5.
That might be intended, since it keeps diversity through the reset, or it might
be a bug.

### 5.7 A short worked example (circle packing, one metric)

The scores below are made up for illustration.

| Step | Coin flip | Parent / inspirations | Child's score | What happens |
|---|---|---|---|---|
| start | | | | Hall of fame: `S (0.96)`. Islands 0–4: `[S]` each. |
| 1 | island 2 | parent `S`, none | `A = 1.20` | `A` is appended to island 2 and **becomes the champion** (1.20 > 0.96). |
| 2 | hall of fame | parent `A`, none | `B = 1.10` | It didn't beat 1.20, so `B` is **thrown away**. |
| 3 | island 2 | island 2 is `[S, A]`, so the parent is `A` (≈100%) and `S` is the inspiration | `C = 0.50` | `C` is appended to island 2 anyway, but it will rarely be picked. |
| 4 | island 4 | parent `S`, none | `D = 1.05` | `D` is appended to island 4. Island 4 is now heading in its own direction. |

### 5.8 Things that surprise people

- **With one metric, the hall of fame holds exactly one program.** For circle
  packing, every heads step sends the LLM *only the champion* with no
  inspirations, which makes it pure hill climbing. Prompts that show several
  programs side by side come only from island steps.
- **The reset may never happen in short runs.** The default budget is 100 LLM
  calls over 4 parallel loops. With fast evaluations like the `src/data`
  example, the run probably ends well before 2 hours, so the islands are never
  mixed. The `maths/` problems have 100–1000 s searches, so their runs last long
  enough for resets.
- **Parallel loops read the same database while it changes.** The four loops run
  concurrently, so a child registered by one loop can be sampled by another on
  its very next step.
- **Nothing is saved to disk.** The database exists only in memory.
  `get_state()` and `set_state()` exist for checkpointing, but nothing calls
  them. If the process stops, the population is gone, except whatever
  `LocalLogger` printed to the console.

### 5.9 The simpler option: `GreedyDatabase`

[`GreedyDatabase`](src/components/evolution_database.py#L93) is just the hall of
fame, with no islands and no resets. Each step samples up to 5 distinct record
holders uniformly at random, and a child is kept only if it breaks a record.
With a single metric, that is pure hill climbing on the champion. It's simpler
and greedier, and more likely to get stuck in a local optimum. To switch, set
`database: GreedyDatabase` in the YAML.

---

## 6. How the LLM sits inside the search loop

This is the core of [`step_loop`](src/alpha_evolve.py#L51), with comments added:

```python
while logger.total_steps < total_steps:                       # budget check
    root, *previous = database.sample_programs()              # SELECTION
    prompt = prompt_sampler.sample_prompt(desc, root, previous)  # CONTEXT for the LLM
    try:
        llm_sample = await llm.generate_async(prompt)         # LLM PROPOSES A MUTATION
        code = parsing_utils.apply_diffs(root.text, llm_sample)  # MATERIALISE THE CHILD
        scores, output = await evaluator.run_async(code, root.output)  # FITNESS
    except (CodeExecutionError, LLMInferenceError, DiffParsingError,
            TypeError, ValueError) as e:
        logger.log_error(...); continue                       # failed step, still counts
    child = Program(text=code, scores=scores, output=output,
                    sampled_from=root.sampled_from, island_index=root.island_index)
    database.register_program(child)                          # SURVIVAL
    logger.log_step(child, prompt, llm_sample)
```

The LLM is never asked to solve the maths problem directly. It is asked to
**improve a program** whose score it can see. The research loop closes through
two feedback channels:

1. **Scores in context, which works like in-context learning.** Every program in
   the prompt carries its `Obtained scores`. The LLM sees which variants scored
   higher and is asked to combine their strengths or try something new. Once a
   child is registered, its scores show up in future prompts. The database
   decides **which** history the LLM gets to see. That's the selection pressure.

2. **Warm-starting through `output_artifacts` → `PARENT_OUTPUT`.** In the
   `maths/` problems, the evolved code is a *search function* such as
   `search_for_best_circle_packing(init_centers)`. `evaluate()` starts it from
   `PARENT_OUTPUT`, the best solution the parent found, and returns the improved
   solution as `output_artifacts`. Progress therefore builds up in two places:
   the **algorithm**, which the LLM edits, and the **solution**, which is handed
   from parent to child. Even a small code change can push the solution further,
   because the child starts where the parent stopped.

   The `src/data` circle packing example is the simpler *constructor* style: the
   code builds the packing directly and returns no artifacts.

### Writing your own problem: the contract

An initial program must:

- wrap the code the LLM may edit in `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`,
- define a top-level `evaluate()` **outside** the evolve blocks that returns
  `{"scores_to_maximize": {metric: float, ...}, "output_artifacts": <JSON-serialisable, optional>}`.
  **Higher is better** for every metric,
- optionally read `PARENT_OUTPUT` to warm-start (but see gotcha 1 in section 8),
- import only what is installed in `src/requirements.txt`, because that's what
  the sandbox image contains.

The config's `problem_description` should describe anything the LLM can't see:
the scoring rules, time limits and the expected function signature.

---

## 7. The async design: how and why

### 7.1 What runs where

```
 Main thread ─ asyncio event loop
 │  step_loop #1 ─ sample → prompt ─ await LLM ············ → diff ─ await eval ················· → register → log → ...
 │  step_loop #2 ─ sample → prompt ─ await LLM ····· → diff ─ await eval ·························· → register → ...
 │  step_loop #3 ─ ...
 │  step_loop #4 ─ ...
 │
 ├─ worker threads (one per call):       generate() → blocking HTTPS request to Gemini
 ├─ default thread pool (asyncio.to_thread): run() → docker-py → blocks on container.wait()
 └─ Docker containers (separate processes, 1 CPU each): the candidate program itself
```

- **Coroutines (`async def`)** are `step_loop` and `async_run`. All of their
  logic runs on **one thread**, the event-loop thread. That includes database
  sampling, prompt building, diff application, `register_program` and logging.
- The LLM call is adapted to async by
  [`generate_async`](src/components/llm_client.py#L45), which runs the blocking
  `generate()` with `loop.run_in_executor(...)` on a fresh single-thread
  `ThreadPoolExecutor` and then `await`s it.
- Evaluation is adapted by
  [`run_async`](src/components/evaluator_client.py#L59), which calls
  `asyncio.to_thread(self.run, ...)` and so uses the loop's default thread pool.
  On Python 3.11 that pool's `max_workers` is `min(32, cpu_count + 4)`.
- **Concurrency** comes from `asyncio.TaskGroup` starting `num_coroutines`
  copies of `step_loop` ([alpha_evolve.py:122](src/alpha_evolve.py#L122)).
  Whenever one coroutine hits an `await`, the event loop switches to another.

### 7.2 Why async?

1. **The workload mostly waits.** An LLM call takes seconds to minutes. An
   evaluation can take up to `timeout_seconds` (2000 s). The `maths/` search
   functions deliberately run for 100–1000 s. A sequential loop would leave the
   CPU and the API idle almost all the time. With N coroutines, N candidates are
   in flight at once, giving roughly N× throughput. Different candidates also
   **overlap in their stages**: one loop waits on Gemini while another waits on
   Docker.

2. **Shared state needs no locks.** `database` and `logger` are plain Python
   objects with dicts and lists and no locks. That's safe here because:
   - every method that touches them is called only from the event-loop thread,
   - none of those methods contains an `await`, so each one runs to completion
     before any other coroutine can run.

   The only places where a coroutine can be paused are the two `await`s, and the
   shared state is consistent at both. If you replaced the coroutines with plain
   threads, each running `step_loop`, you would need locks around
   `register_program`, `sample_programs`, island resets and the logger counters.

3. **Evolution becomes asynchronous and steady-state.** There are no generation
   barriers. Each loop samples from the database *as it is right now*, so a
   strong child is picked up by the other loops at once. Nobody waits for the
   slowest evaluation of a generation. The AlphaEvolve paper describes the same
   asynchronous pipeline, built to maximise how many candidates get evaluated.

4. **The control flow stays simple.** Each coroutine reads like an ordinary
   sequential loop. Parallelism is one config number, and stopping is just
   cancelling tasks.

5. **The GIL isn't a bottleneck.** The worker threads spend their time blocked on
   network or Docker-socket I/O, which releases the GIL. The heavy computation
   happens inside containers, which are separate processes.

The thread wrappers exist because `google-genai` (as used here) and `docker-py`
are **synchronous** libraries, and `run_in_executor` / `to_thread` is the
standard way to call blocking code from asyncio without freezing the event loop.
`google-genai` also has a native async API (`client.aio.models.generate_content`)
that could replace the thread for the LLM side. `docker-py` has no async API.

### 7.3 Budget, termination and cancellation

- **Budget:** each loop checks `logger.total_steps < num_llm_calls` at the top
  of every iteration. Failed steps count too.
- **Stopping:** `asyncio.wait(..., return_when=FIRST_COMPLETED)`
  ([alpha_evolve.py:137](src/alpha_evolve.py#L137)) returns as soon as one loop
  sees the budget used up. The others are then `cancel()`ed. The code comment
  gives the reason: some evaluations can hang, and waiting for every loop would
  stall the end of the run.
- **What a cancel does:** it raises `CancelledError` at the coroutine's current
  `await`. Its in-flight step is thrown away and never registered. So the number
  of LLM calls actually made is at most `num_llm_calls + num_coroutines − 1`.
- **Cancelling a coroutine doesn't kill its thread.** Python threads can't be
  forcibly stopped:
  - An evaluation thread keeps running until its container exits or times out.
    Then it removes the container in its `finally`.
  - `asyncio.run` calls `loop.shutdown_default_executor()` on exit. In
    **Python 3.11, which this venv uses, that call waits with no timeout**. So
    the process can stay alive after printing the best program, until the last
    evaluation finishes.
  - For a cancelled LLM call, the `with ThreadPoolExecutor(...)` block in
    `generate_async` exits through `shutdown(wait=True)`, which **blocks the
    event-loop thread** until that HTTP request returns.

  Both effects only matter at the end of a run.
- **Errors that aren't caught:** an exception outside the caught list in a
  `step_loop` makes the `TaskGroup` cancel every loop. The run then ends with an
  `ExceptionGroup`.

### 7.4 Choosing `num_coroutines`

Each concurrent evaluation is a container with 1 CPU and 512 MB of RAM.
Reasonable upper limits are:

- your CPU core count,
- free RAM divided by 512 MB,
- your Gemini rate limit.

Above `cpu_count + 4` (capped at 32), the default thread pool queues the extra
evaluations anyway.

---

## 8. Gotchas worth knowing

1. **The `PARENT_OUTPUT` fallback never fires on the first evaluation.**
   `_build_script` always defines `PARENT_OUTPUT`. When there is no parent output
   it is `json.loads("null")`, which is `None`. As a result, the pattern
   `globals().get('PARENT_OUTPUT', default)` used in the `maths/*/initial_program.py`
   files returns `None`, not the default. I checked this in Python:

   - `globals().get(...)` returns `None` when the name is bound to `None`.
   - `np.ascontiguousarray(None, dtype=float64)` returns `array([nan])`.

   For example, `cap_sets` would then iterate over `None` and raise `TypeError`.
   That would fail the initial evaluation, which isn't wrapped in `try`, so the
   run would abort. I haven't run these configs end to end. The default
   `src/data` example doesn't read `PARENT_OUTPUT` and isn't affected. Two
   possible fixes:
   - only emit the `PARENT_OUTPUT` assignment when `parent_output is not None`,
   - or use `globals().get('PARENT_OUTPUT') or default` in the programs.

2. **Edits with no match fail silently.** `_apply_edits` uses
   `str.replace(old, new, 1)`, which replaces only the first occurrence inside
   each mutable block, even though the prompt says "replace *all* matching
   occurrences". A SEARCH text that matches nothing is skipped without an error.
   The child can then be identical to its parent and still use up an evaluation
   and a budget step.

3. **Edits across evolve-block boundaries never match.** `Program.render()` joins
   all mutable blocks into one listing. The LLM may therefore write a SEARCH
   block that spans two blocks, for example the import block and the search
   function in `maths/circle_packing`. Edits are applied to each block
   separately, so such a SEARCH never matches.

4. **The default `--problem_config` only resolves from the repo root.** Run
   `python src/run.py` from `alphaevolve/`. The value
   `data/problem_config/circle_packing.yaml` would work from any directory,
   thanks to the fallback that resolves paths relative to `src/`.

5. **Requirements:** Docker must be running, and `GEMINI_API_KEY` must be
   exported.

---

## 9. Running it

```bash
cd /home/bingcheng/Downloads/alphaevolve
source .venv/bin/activate
export GEMINI_API_KEY=<your key>

python src/run.py                                                   # default: src/data circle packing
python src/run.py --problem_config=maths/circle_packing/config_square.yaml
```

The sandbox image is built on the first run, which takes a few minutes. Each
step is then printed by `LocalLogger`. When the run finishes, the best program
and its scores are logged.
