---
name: Change
about: A behaviour or feature change, with the tests and docs it drags along
title: ''
labels: ''
assignees: ''
---

## Problem

<!-- What is wrong or missing, and what it costs. Name the file, the run, the
     figure -- concrete beats abstract. -->

## What already exists

<!-- The near-misses: what is already here that looks like this feature and is
     not. A job manifest that carries no digests, a contract hash that is not a
     data hash. Skipping this section is what gets an issue re-litigated in
     review. "Nothing" is a valid answer. -->

## Proposal

<!-- What to build, and the decisions inside it someone could reasonably
     disagree with. -->

## Code changes

| File | Change |
| --- | --- |
|  |  |

## Unit tests

<!-- Which tier, per documentation/operations/testing.md. e2e.yml is
     workflow_dispatch only, so a change covered ONLY by an e2e assertion is not
     covered by CI. -->

## Documentation

<!-- Which mkdocs pages describe the behaviour being changed. "None" is an
     answer; silence is an omission. -->

## Acceptance criteria

<!-- The first line is what proves this change works. The rest is the standing
     definition of done here -- delete one only when it genuinely does not
     apply. -->

- [ ]
- [ ] unit tests in the right tier
- [ ] mkdocs pages updated, `mkdocs build --strict` clean
- [ ] `ruff`, `codespell` and `scripts/check_unicode.py` pass

## Out of scope

<!-- Follow-ups that belong in their own issue. -->

<!-- Branch `feature-<issue>`; commits read `#<issue>: <file>, <what changed>`. -->
