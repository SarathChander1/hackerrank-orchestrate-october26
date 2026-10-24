# Evaluation report

The solution uses only participant-facing files under `dataset/`. It normalizes
JSON, handover text, log, and SQL evidence into policy records, resolves current
and historical identifiers, and emits phase-ordered audit steps.

The public sample set contains 32 cases. Run `python code/main.py` from the
repository root to regenerate the full 518-case output. The validation checks
case coverage, sorted case IDs, valid JSON arrays, consecutive step numbers,
allowed operation/state combinations, and the required output header.

Additional evaluation cases:

1. A request containing only a historical email must resolve to the same
   `user_id` and include evidence found under the current identifier.
2. A retained security audit record must remain present while deletion findings
   are remediated.
3. An active data flow must be disabled before downstream deletion is verified.

Known limitations are recorded by the generated output itself rather than
silently fabricating evidence: non-text image/PDF evidence is not OCR'd, and
only records with an explicit participant-facing subject reference are emitted.
