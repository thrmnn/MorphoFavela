# `.claude/agents/` — IVF project subagents

Project-scoped Claude Code subagents that travel with the repo. Each
agent is a single markdown file with YAML frontmatter (`name`,
`description`, `tools`) plus a system prompt body.

These are **validators**: read-only agents that check the working tree
or a git diff against a contract and report findings as a structured
punch list. They do not modify files. Their output is meant to be
read by a human (or piped into CI later).

## Why agents and not scripts?

A script can check a contract; an agent can also *interpret* what the
contract means in context — e.g. mapping a changed file to "which
section of the technical report should mention this?". For the
boundary cases (was that script change pipeline-relevant or just a
typo fix?) the LLM judgment earns its keep. For the rigid cases
(does this JSON have all required keys?) the agent just runs a
Python one-liner and reports — same as a script would, but inside
the same harness as the rest of the pipeline.

## Available agents

| Name | When to invoke | Inputs |
|---|---|---|
| [`data-contract-checker`](data-contract-checker.md) | Before running pipeline scripts on a site, after pulling new data, or before changing `data/README.md` | Site name (e.g. `vidigal`) or `--all` |
| [`sampling-auditor`](sampling-auditor.md) | After `run_campaign_sampling.py`, before submitting patches to CFD execution, or to confirm campaign integrity | Site name or `--all` |
| [`report-sync-auditor`](report-sync-auditor.md) | Before committing pipeline / figure / sampling changes, or to audit a series of recent commits for documentation drift | Git ref range (`HEAD~3..HEAD`), `staged`, or `working` (default) |

## Invocation

Agents are invoked from a Claude Code session via the `Task` /
`Agent` tool, by name. Example phrasings that the orchestrator will
match against the `description` field:

- "Run the data contract check on vidigal"
- "Audit the campaign sampling for all sites"
- "Check whether the report is in sync with HEAD~5..HEAD"

For deterministic invocation, name the agent explicitly:
"Use the `report-sync-auditor` agent on the staged diff."

## Output contract

Every validator returns a markdown report with this shape:

```
# <agent-name> — <scope>

**Status: PASS** | **WARNING** | **FAIL**

## <section per check>
- [PASS|WARN|FAIL] <check> — <details if not PASS>

## Summary
<1-3 lines>

## Next steps
<concrete remediation commands, or "no action needed">
```

Severity rules:

- **FAIL** = a hard contract violation. Block the action that
  triggered the check (commit, run, submission) until fixed.
- **WARN** = borderline or context-dependent. Read and decide.
- **PASS** = check ran and the contract holds.

The overall status is the worst per-check status across the report.

## Design rules (when adding a new agent)

1. **Read-only.** Tools should be `Read, Grep, Glob, Bash` — no `Edit`
   or `Write`. Validators report; humans (or other agents) fix.
2. **Single responsibility.** One agent = one contract. Don't bundle
   data-contract checks with sampling checks; users invoke the right
   agent for the question they're asking.
3. **Self-contained system prompt.** Subagents do not see the parent
   conversation. The system prompt must encode the contract
   explicitly (paths, schemas, thresholds) — not "go read the README".
4. **Cite the contract.** When flagging a finding, reference the file
   and line of the source rule (`data/README.md L37`, `CLAUDE.md
   "Technical report" section`).
5. **Fail loudly on missing inputs.** A scope referencing a site that
   doesn't exist is FAIL with a clear message — not a silent skip.
6. **Deterministic.** Stable ordering, no timestamps, no random
   sampling.
7. **Don't auto-fix.** Even when the fix is obvious, describe it under
   "Next steps" only.

## Adding a new agent

1. Create `.claude/agents/<name>.md` following the template of an
   existing one.
2. Set `tools: Read, Grep, Glob, Bash` unless the agent has a
   genuine need for more.
3. Include a `description` that explains *when* the orchestrator
   should invoke this agent — start with a verb and mention the
   trigger conditions.
4. Add a row to the *Available agents* table above.
5. Test by invoking it on the current repo and confirming it produces
   the expected punch list.

## Hooks (not yet wired)

These agents are currently invoked manually. A future iteration may
auto-run `report-sync-auditor` on `PreCommit` via
`.claude/settings.json`; for now we want to characterise the
false-positive rate before adding it to the gate.
