# Project conventions for Claude

The global rules in `~/.claude/CLAUDE.md` apply. This file lists project-specific procedure on top of those.

## Communication

- Plain language only. No jargon, no metaphors, no domain shorthands that aren't explained. Words to avoid in particular: "wall-clock", "load-bearing", "orthogonal", "downstream consumers", "delve", "nuanced". Spell things out.
- When asked for a summary, lead with the headline conclusion. Don't bury the most important fact in the middle of supporting detail. If the answer is a verdict (safe / unsafe / unverifiable), say it first.
- For multi-part findings, use tables or short bullet lists rather than paragraphs. One number per row, not a sentence describing the number.
- Don't restate the question or the conversation history. Answer the question.

## Investigating data state

- Information about one session comes from a few Alyx REST calls. `one.alyx.rest('sessions', 'read', id=eid)` returns the session dict, including `extended_qc` and `data_dataset_session_related` (every dataset registered for the session, each entry carrying `collection`, `name`, `data_url`, `revision`, `version`). `one.alyx.rest('insertions', 'list', session=eid)` lists the probe insertions; `one.alyx.rest('insertions', 'list', id=pid)[0]` gives one insertion's full record, with histology state under `json`. `one.load_dataset(eid, name, collection)` fetches the actual file contents.
- Alyx supports dataset revisions (e.g. `#2024-12-01#` between dataset name and extension). Today every dataset in this project has revision `''` (the default), but re-sortings or re-extractions will land under dated revisions. Each `data_dataset_session_related` entry carries a `revision` tag and a `default_revision` flag (serialised as the string `'True'` or `'False'`); when more than one revision exists, pick the `default_revision == 'True'` entry rather than assuming a single version exists.
- When auditing extraction status across the dataset, write a script that produces a CSV under `metadata/` and commit the CSV alongside the script. Don't paste large output blocks into chat.
- When the user reports a surprising empirical observation, before theorising, check whether the validation tool actually tests what the user thinks it tests. The events used for neural-response checks come from a separate IBL alf file than the epoch timings; both must be considered together when reasoning about what a check proves.
- When a hypothesis is dead, say it's dead and move on. Don't keep defending it.

## Specs

Non-trivial changes get a spec under `specs/<short_name>.md` before implementation. The spec must be self-contained: an agent without conversation context should be able to implement from it. Include exact dataset names, exact column names, exact code snippets for non-obvious logic, and explicit lists of files to add / modify / delete.

## GitHub issues

When updating an issue with new findings:
- Rewrite the body to put the new headline first, not append a comment.
- Remove sections that are no longer relevant to the current state of the investigation.
- Keep code snippets exact (paste real lines, not paraphrases).

## Skills the user invokes most

- `/discuss` for problem exploration. Ask one clarifying question at a time; do not jump to a plan.
- `/software-eng` for any code change.
- `/reflect` after a task where workflow or project knowledge changed.

## Environment

Never install Python packages or change the virtual environment without explicit approval. Never set or export environment variables inline in `Bash` commands without approval.
