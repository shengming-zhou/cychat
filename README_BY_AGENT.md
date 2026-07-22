# CyChat agent handoff

Last updated: 2026-07-22 (Asia/Shanghai)

## Project objective

CyChat is a small Chinese-language model training project inspired by nanochat.
The aim is to build a clear, reproducible training stack and finish the final
training run on 8 H100 GPUs in about two hours or less.

This is not a requirement to reproduce nanochat line for line. GPT-style models
are a baseline rather than a restriction; newer architectures or training
methods are welcome when their benefit can be measured within the project's
time, compute, and implementation-complexity budget.

## Hard constraints

1. Development follows three hardware stages:
   - MacBook Air M4: write code and run tiny functional tests locally.
   - One RTX 5090: validate the CUDA path, GPU memory use, numerical behavior,
     throughput, and training feasibility.
   - 8 H100 GPUs: run the final distributed training job, targeting no more
     than approximately two wall-clock hours.
2. The model and training/evaluation data are Chinese-focused.
3. The user is the primary code author. Agents should normally explain, design,
   review, diagnose, and provide pseudocode. Agents modify project source only
   when the user explicitly requests implementation.
4. `/nanochat/` is a local reference copy only. It must never be committed or
   pushed to the CyChat repository.
5. Establish a measurable baseline before introducing major architectural,
   tokenizer, data, or systems changes. Prefer controlled changes that preserve
   attribution of improvements and regressions.
6. On the M4 Mac, keep all project-controlled environments, installed
   libraries, managed Python installations, and caches inside the repository
   root. This storage-location constraint does not apply to the 5090 or H100
   environments. OS-managed caches may remain outside the repository when the
   operating system or driver does not provide a supported override.

## Repository map

- Repository root: `/Volumes/cutset/cychat`
- GitHub: `https://github.com/shengming-zhou/cychat`
- Default branch: `main`
- `nanochat/`: ignored local reference source; not part of CyChat
- `.venv/`, `.python/`, `.cache/`: ignored Mac-local environment and cache
  locations; never commit their contents
- `.env.mac.sh`: optional ignored Mac-only shell configuration
- `README_BY_AGENT.md`: durable project state shared across conversations
- `AGENTS.md`: instructions that require agents to read and maintain this file

## Current status

- Phase: repository bootstrap and project framing.
- GitHub contains the project shell; no CyChat model/training implementation has
  been established yet.
- The three-stage hardware workflow and final time budget are agreed.
- Dependency management will use uv. The Mac environment/cache storage policy
  is decided; the concrete `pyproject.toml` and Mac-local setup are not created
  yet.
- Model architecture, tokenizer, dataset mixture, parameter count, context
  length, training-token budget, and evaluation suite remain undecided.

## Stage gates

### Stage 1 — M4 functional baseline

The smallest configuration should exercise the tokenizer/data path, model
forward pass, loss, backward pass, optimizer step, checkpoint round trip, and a
short deterministic overfit/smoke test. Tests should remain cheap enough for
frequent local iteration; MPS support is useful but must not distort the CUDA
design.

### Stage 2 — RTX 5090 feasibility

Run the same training path on CUDA and record precision mode, peak allocated
memory, tokens per second, step time, loss behavior, and any compile/fused-kernel
settings. Use these measurements to discover unsupported operations and estimate
the 8-GPU configuration; do not treat single-GPU speedup as perfectly linear.

### Stage 3 — 8 H100 final training

Validate distributed startup and checkpointing with a short run before spending
the full budget. Derive the required aggregate token throughput from the chosen
training-token budget and the two-hour wall-clock limit. Record H100 variant,
interconnect/topology, world size, precision, global batch size, sequence length,
tokens per second, model FLOPs utilization if available, peak memory, total
tokens, final loss/evaluations, and elapsed time.

## Open decisions

- Chinese tokenizer type and vocabulary size
- Data sources, licensing, cleaning, deduplication, and train/eval split
- Architecture family and exact parameter budget
- Context length and global batch/token schedule
- Pretraining-only versus any supervised/post-training stage
- Chinese evaluation tasks and comparison baseline
- H100 model (SXM/PCIe and 80/94 GB), topology, and storage/data bandwidth

These are deliberately open. Resolve them from measurements and the final
compute budget rather than silently assuming nanochat's choices.

## Experiment record convention

For a result that affects decisions, record at least: date, Git commit, hardware,
configuration identifier, precision, parameter count, sequence length, batch or
tokens per step, throughput, peak memory, losses/evaluations, elapsed time, and a
short conclusion. Large logs and checkpoints stay outside Git; this file should
hold only durable conclusions and the current next action.

## Maintenance rule

After a meaningful decision or experiment, update the date, current status,
resolved/open decisions, and next action here. Keep facts separate from proposals
and do not claim that an unrun experiment succeeded.
