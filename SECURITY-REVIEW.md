# Security Review

## Scope

The review covered the compliance pipeline in `code/main.py`, focusing on
untrusted evidence ingestion, deletion authorization, retention enforcement,
identity matching, activity indexing, and LLM-generated remediation suggestions.

## Findings and Remediation

| # | Severity | Area | Finding | Remediation |
|---|---|---|---|---|
| 1 | HIGH | SQL evidence loading | Evidence SQL was executed with `sqlite3.executescript`, allowing arbitrary statements in a compromised evidence file. | SQL evidence is now parsed with the restricted `INSERT` parser only; no evidence SQL is executed. |
| 2 | HIGH | Intent fallback | LLM/API failures could default ambiguous requests to `DELETE`. | Classification failures now fail closed to `NOT_APPLICABLE` and emit fallback logging. |
| 3 | HIGH | Retention enforcement | Generated destructive steps could conflict with security, legal-hold, or backup retention requirements. | Retention is resolved per resource key before LLM auditing. Destructive operations are removed and replaced with retention verification or restore blocking. |
| 4 | HIGH | Activity scanning | Flat activity scans caused repeated global work and could miss precise resource associations. | Activity is indexed by `system_id|resource_id`; replay and flow checks use direct lookups. |
| 5 | MEDIUM | Identity matching | Lossy substring/digit matching could associate unrelated identifiers. | Record matching now uses exact normalized set intersection. |
| 6 | MEDIUM | LLM auto-fixer | LLM suggestions could introduce unsupported resources or operations. | Only `MISSING_VERIFICATION` suggestions for existing resources and allowlisted verification operations are accepted. |

## Safety Properties

- LLM output cannot override retention decisions.
- Invalid or unavailable LLM responses cannot authorize deletion.
- Untrusted SQL evidence cannot execute filesystem or database side effects.
- Auto-fixes cannot create new resource keys.
- Output generation remains deterministic when LLM services are unavailable.
- LLM fallback state is persisted in `llm_audit.json` through `llm_failed` and
  `fallback_mode`.

## Validation

- Python compilation passed.
- Focused security regression tests passed for:
  - fail-closed intent classification;
  - parser-only SQL evidence loading;
  - retention override idempotence and destructive-step removal;
  - exact identity matching;
  - indexed activity loading;
  - hallucinated-resource rejection.
- Full pipeline completed successfully.
- Final output contains 518 rows.
- All generated step numbers are consecutive.
- All `NOT_APPLICABLE` rows contain an empty action list.
- LLM audit metadata was generated for all 518 cases.
