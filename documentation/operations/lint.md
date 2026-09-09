# Lint

`lint.yml` runs on every push and pull request, and [`.githooks/pre-commit`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/.githooks/pre-commit) runs the same checks locally so the two cannot disagree.

| Workflow | What it checks |
|---|---|
| [`lint.yml`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/.github/workflows/lint.yml) | `ruff check .` — rules in [`.ruff.toml`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/.ruff.toml) |

The rule set is deliberately **correctness-only** (`E4`, `E7`, `E9`, `F`, `W6`): syntax errors, undefined names, unused imports and variables, bare `except`, invalid escape sequences. No formatting, import-ordering, or type-annotation rules are enabled, so a red build always means a real defect rather than a style preference. Notebooks are excluded — several carry pasted tabular output inside code cells and are not parseable Python.

The `ruff` version is pinned in the workflow so a new upstream release cannot redden an untouched branch; bump it deliberately.

## Running it by hand

```bash
pip install ruff==0.16.1
ruff check .
```

## Enabling the hook

```bash
git config core.hooksPath .githooks
```

The hook checks only **staged** files, and skips with a notice (rather than failing) when the linter is not installed — a fresh clone stays committable, and CI remains the enforcement. Bypass with `git commit --no-verify`.

