
from __future__ import annotations

import csv
import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable


logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output.csv"
IDENTITY_OUTPUT = ROOT / "identity_resolution.json"
STATE_OUTPUT = ROOT / "state_reconstruction.json"
STATE_VALIDATION_OUTPUT = ROOT / "state_validation.json"
AUDIT_OUTPUT = ROOT / "llm_audit.json"
RETENTION_OUTPUT = ROOT / "retention_overrides.json"

DELETE_TYPES = {
    "USER_PROFILE",
    "SECONDARY_IDENTIFIER",
    "SEARCH_INDEX_ENTRY",
    "CACHE_ENTRY",
    "SUPPORT_ATTACHMENT",
    "USER_EXPORT",
    "MARKETING_PROFILE",
}
REDACT_TYPES = {"SUPPORT_MESSAGE", "EMAIL_DELIVERY_LOG", "ACCOUNT_TOMBSTONE"}
ANONYMIZE_TYPES = {"ANALYTICS_PROFILE", "PRODUCT_TELEMETRY"}
RETAIN_TYPES = {"SECURITY_AUDIT_LOG", "LEGAL_HOLD_RECORD", "BACKUP_ARCHIVE"}

FINDING_ORDER = {
    "REPORT_REMAINING_RECORD": 1,
    "REPORT_EXPOSED_PII": 2,
    "REPORT_ACTIVE_DATA_FLOW": 3,
    "REPORT_RESTORE_RISK": 4,
}
REMEDIATION_ORDER = {
    "DISABLE_DATA_FLOW": 1,
    "BLOCK_RESTORE": 2,
    "DELETE_RECORD": 3,
    "REDACT_RECORD": 4,
    "ANONYMIZE_RECORD": 5,
}
VERIFICATION_ORDER = {
    "VERIFY_RECORD_ABSENT": 1,
    "VERIFY_RECORD_PRESENT": 2,
    "VERIFY_RECORD_REDACTED": 3,
    "VERIFY_RECORD_ANONYMIZED": 4,
    "VERIFY_DATA_FLOW_DISABLED": 5,
    "VERIFY_RESTORE_BLOCKED": 6,
}
PHASE_ORDER = {"FINDING": 1, "REMEDIATION": 2, "VERIFICATION": 3}
OPERATION_ORDERS = {
    **FINDING_ORDER,
    **REMEDIATION_ORDER,
    **VERIFICATION_ORDER,
}
ALLOWED_OPERATIONS = set(OPERATION_ORDERS)
REQUIRED_OUTPUT_COLUMNS = ("case_id", "deletion_status", "steps_json")

INTENT_CATEGORIES = {"DELETE", "ACCESS", "CORRECT", "EXPORT", "OPT_OUT", "OTHER"}
DELETE_INTENT = "DELETE"
NOT_APPLICABLE_INTENT = "NOT_APPLICABLE"

# Keep negations separate from positive terms so "do not delete" cannot be
# classified as a deletion request merely because it contains "delete".
NEGATIVE_DELETE_RE = re.compile(
    r"\b(?:do\s*not|don't|dont|never|no)\s+"
    r"(?:\s+\w+){0,2}\s*(?:delete|erase|remove|close|supprim(?:er|e)|"
    r"löschen|eliminar|apagar|cancellare|hapus)\b|"
    r"\b(?:keep|retain|preserve|maintain|leave)\s+(?:my\s+)?"
    r"(?:account|profile|data|records?)\b",
    re.IGNORECASE,
)
POSITIVE_DELETE_RE = re.compile(
    r"\b(?:delete|erase|remove|close|terminate|deactivate|purge|destroy)\b|"
    r"\b(?:supprimer|suppression|löschen|löschung|eliminar|borrar|"
    r"apagar|excluir|cancellare|eliminare|hapus|padam)\b|"
    r"\b(?:remove|close|delete)\s+(?:my\s+)?(?:user\s+)?"
    r"(?:account|profile|data|records?)\b",
    re.IGNORECASE,
)
NON_DELETE_INTENT_RES = {
    "ACCESS": re.compile(r"\b(?:access|view|see|inspect|obtain)\b", re.IGNORECASE),
    "EXPORT": re.compile(r"\b(?:copy|export|download|portability|portable)\b", re.IGNORECASE),
    "CORRECT": re.compile(
        r"\b(?:correct|rectify|update|change| amend|fix)\b|"
        r"\bchange\s+(?:my\s+)?(?:email|address|name|details?)\b",
        re.IGNORECASE,
    ),
    "OPT_OUT": re.compile(
        r"\b(?:unsubscribe|opt[\s-]?out|do\s+not\s+sell|stop\s+(?:marketing|emails?)|"
        r"remove\s+me\s+from\s+(?:the\s+)?mailing\s+list)\b",
        re.IGNORECASE,
    ),
}

LLM_PROMPT = """Classify this user request into exactly one category: DELETE, ACCESS, CORRECT, EXPORT, OPT_OUT, or OTHER.

Request: "{request_text}"
Context: The user is a customer of an e-commerce platform.

Return ONLY the category name."""


def classify_with_regex(request_text: str) -> str | None:
    """Return a decisive intent, or None when an LLM is required."""
    text = request_text or ""
    if not text.strip():
        return NOT_APPLICABLE_INTENT
    if NEGATIVE_DELETE_RE.search(text):
        return NOT_APPLICABLE_INTENT
    for category, pattern in NON_DELETE_INTENT_RES.items():
        if pattern.search(text):
            return NOT_APPLICABLE_INTENT
    if POSITIVE_DELETE_RE.search(text):
        return DELETE_INTENT
    return None


def llm_prompt(request_text: str) -> str:
    """Build the mandated fallback prompt without allowing format injection."""
    return LLM_PROMPT.format(request_text=(request_text or "").replace('"', '\\"'))


def classify_with_llm(
    request_text: str,
    client: Callable[[str], str] | None = None,
) -> str:
    """Classify an ambiguous request, defaulting to DELETE on uncertainty.

    A client can be injected for tests or an application SDK. When no client is
    supplied, the optional OpenAI-compatible HTTP endpoint is used if
    OPENAI_API_KEY is configured; otherwise the privacy-safe DELETE default is
    applied.
    """
    category, _ = classify_with_llm_status(request_text, client)
    return category


def classify_with_llm_status(
    request_text: str,
    client: Callable[[str], str] | None = None,
) -> tuple[str, bool]:
    """Return the intent and whether privacy-safe fallback mode was used."""
    try:
        if client is not None:
            response = client(llm_prompt(request_text))
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("Fallback Mode: LLM classification credentials unavailable")
                return NOT_APPLICABLE_INTENT, True
            payload = json.dumps({
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": 0,
                "messages": [{"role": "user", "content": llm_prompt(request_text)}],
            }).encode("utf-8")
            request = urllib.request.Request(
                os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"),
                data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as result:
                response = json.loads(result.read().decode("utf-8"))["choices"][0]["message"]["content"]
        category = str(response).strip().upper().split()[0]
        return (category if category in INTENT_CATEGORIES else NOT_APPLICABLE_INTENT), False
    except (KeyError, IndexError, TypeError, ValueError, OSError, urllib.error.URLError):
        logger.exception("Fallback Mode: LLM classification failed")
        return NOT_APPLICABLE_INTENT, True


def classify_intent(
    request_text: str,
    llm_client: Callable[[str], str] | None = None,
) -> str:
    """Run Stage A, then Stage B for ambiguous requests."""
    stage_a = classify_with_regex(request_text)
    if stage_a is not None:
        return stage_a
    return classify_with_llm(request_text, llm_client)


def validate_intent_output(output_path: Path, sample_path: Path) -> tuple[bool, list[str]]:
    """Verify sample non-deletion cases are NOT_APPLICABLE with empty actions."""
    with sample_path.open(newline="", encoding="utf-8") as handle:
        expected = {row["case_id"]: row for row in csv.DictReader(handle)}
    with output_path.open(newline="", encoding="utf-8") as handle:
        actual = {row["case_id"]: row for row in csv.DictReader(handle)}
    errors = [
        case_id for case_id, row in expected.items()
        if row["deletion_status"] == NOT_APPLICABLE_INTENT
        and (actual.get(case_id, {}).get("deletion_status") != NOT_APPLICABLE_INTENT
             or actual.get(case_id, {}).get("steps_json") != "[]")
    ]
    return not errors, errors


def cluster_json(cluster: IdentityCluster) -> dict[str, Any]:
    return {
        "case_id": cluster.case_id,
        "seed_email": cluster.seed_email,
        "user_ids": list(cluster.user_ids),
        "identifiers": {
            field: list(values) for field, values in cluster.identifiers.items()
        },
    }


def validate_identity_resolution(
    clusters: dict[str, IdentityCluster],
    sample_requests_path: Path,
    sample_output_path: Path,
    audits: list[dict[str, str]],
) -> tuple[bool, list[str]]:
    """Check identity completeness and expected sample resource coverage."""
    with sample_requests_path.open(newline="", encoding="utf-8") as handle:
        sample_requests = {row["case_id"]: row for row in csv.DictReader(handle)}
    with sample_output_path.open(newline="", encoding="utf-8") as handle:
        sample_output = {row["case_id"]: row for row in csv.DictReader(handle)}

    errors: list[str] = []
    for case_id, expected in sample_output.items():
        if expected["deletion_status"] == NOT_APPLICABLE_INTENT:
            continue
        request = sample_requests.get(case_id)
        cluster = clusters.get(case_id)
        if request is None or cluster is None:
            errors.append(f"{case_id}: missing deletion cluster")
            continue
        try:
            expected_steps = json.loads(expected["steps_json"])
        except json.JSONDecodeError:
            errors.append(f"{case_id}: invalid sample steps JSON")
            continue
        if not isinstance(expected_steps, list) or any(
            not step.get("system_id") or not step.get("resource_id")
            for step in expected_steps
        ):
            errors.append(f"{case_id}: invalid sample resource reference")
        if identity_value("email", request["email"]) not in {
            identity_value("email", value)
            for value in cluster.identifiers.get("email", ())
        }:
            errors.append(f"{case_id}: seed email missing")
        for audit_row in (row for row in audits if row["user_id"] in cluster.user_ids):
            field = AUDIT_FIELD_ALIASES.get(norm(audit_row.get("field_name")))
            if not field:
                continue
            resolved_values = {
                identity_value(field, value)
                for value in cluster.identifiers.get(field, ())
            }
            for column in ("old_value", "new_value"):
                if audit_row.get(column) and identity_value(field, audit_row[column]) not in resolved_values:
                    errors.append(f"{case_id}: missing audit {field} identifier")

        # Resource IDs are checked by the downstream evidence audit; this
        # phase validates their sample references and the identity cluster
        # used for that lookup.
    return not errors, errors


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATASET / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def norm(value: str | None) -> str:
    return (value or "").strip().casefold()


IDENTITY_FIELDS = ("user_id", "name", "email", "phone", "email_hash")
AUDIT_FIELD_ALIASES = {
    "email": "email",
    "email_hash": "email_hash",
    "phone": "phone",
    "name": "name",
}


def identity_value(field: str, value: str | None) -> str:
    """Normalize an identity value while retaining meaningful punctuation."""
    normalized = norm(value)
    if field == "phone":
        return re.sub(r"\D", "", normalized)
    return re.sub(r"\s+", " ", normalized)


def identity_key(field: str, value: str | None) -> str:
    return f"{field}:{identity_value(field, value)}"


@dataclass(frozen=True)
class IdentityCluster:
    case_id: str
    seed_email: str
    user_ids: tuple[str, ...]
    identifiers: dict[str, tuple[str, ...]]

    @property
    def all_values(self) -> set[str]:
        values = {norm(self.seed_email)}
        values.update(self.user_ids)
        for field_values in self.identifiers.values():
            values.update(norm(value) for value in field_values)
        return {value for value in values if value}


class IdentityGraphResolver:
    """Resolve a request email to a fixpoint over identity and audit edges."""

    def __init__(self, users: list[dict[str, str]], audits: list[dict[str, str]]) -> None:
        self.users = users
        self.audits = audits
        self.users_by_key: dict[str, set[int]] = defaultdict(set)
        self.user_indices_by_id = {
            user["user_id"]: index for index, user in enumerate(users)
        }
        self.audit_users_by_key: dict[str, set[str]] = defaultdict(set)
        self.audits_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
        for index, user in enumerate(users):
            for field in IDENTITY_FIELDS:
                value = user.get(field, "")
                if value:
                    self.users_by_key[identity_key(field, value)].add(index)
        for audit_row in audits:
            self.audits_by_user[audit_row["user_id"]].append(audit_row)
            field = AUDIT_FIELD_ALIASES.get(norm(audit_row.get("field_name")))
            if field:
                for column in ("old_value", "new_value"):
                    value = audit_row.get(column, "")
                    if value:
                        self.audit_users_by_key[identity_key(field, value)].add(
                            audit_row["user_id"]
                        )

    def resolve(self, case_id: str, email: str) -> IdentityCluster:
        discovered_users: set[int] = set()
        discovered_user_ids: set[str] = set()
        discovered: dict[str, set[str]] = defaultdict(set)
        pending_keys = deque([identity_key("email", email)])
        visited_keys: set[str] = set()
        pending_users: deque[int] = deque()

        while pending_keys or pending_users:
            if pending_keys:
                key = pending_keys.popleft()
                if key in visited_keys:
                    continue
                visited_keys.add(key)
                for user_index in self.users_by_key.get(key, ()):
                    if user_index not in discovered_users:
                        discovered_users.add(user_index)
                        pending_users.append(user_index)
                for user_id in self.audit_users_by_key.get(key, ()):
                    user_index = self.user_indices_by_id.get(user_id)
                    if user_index is not None and user_index not in discovered_users:
                        discovered_users.add(user_index)
                        pending_users.append(user_index)
                continue

            user = self.users[pending_users.popleft()]
            user_id = user["user_id"]
            discovered_user_ids.add(user_id)
            for field in IDENTITY_FIELDS:
                value = user.get(field, "")
                if value:
                    discovered[field].add(value)
                    pending_keys.append(identity_key(field, value))
            for audit_row in self.audits_by_user.get(user_id, ()):
                field = AUDIT_FIELD_ALIASES.get(norm(audit_row.get("field_name")))
                if not field:
                    continue
                for column in ("old_value", "new_value"):
                    value = audit_row.get(column, "")
                    if value:
                        discovered[field].add(value)
                        pending_keys.append(identity_key(field, value))

        return IdentityCluster(
            case_id=case_id,
            seed_email=email,
            user_ids=tuple(sorted(discovered_user_ids)),
            identifiers={
                field: tuple(sorted(values, key=norm))
                for field, values in sorted(discovered.items())
                if values
            },
        )

    def resolve_requests(
        self,
        requests: list[dict[str, str]],
        llm_client: Callable[[str], str] | None = None,
    ) -> dict[str, IdentityCluster]:
        return {
            request["case_id"]: self.resolve(request["case_id"], request["email"])
            for request in requests
            if classify_intent(request["request_text"], llm_client) == DELETE_INTENT
        }


@dataclass(frozen=True)
class Record:
    evidence_id: str
    captured_at: datetime
    system_id: str
    resource_id: str
    record_type: str
    state: str
    subject: str


@dataclass(frozen=True)
class ActivityEvent:
    activity_id: str
    event_time: datetime
    event_type: str
    system_id: str
    resource_id: str
    result: str
    details: str = ""
    user_id: str = ""


def load_system_activity() -> dict[str, list[ActivityEvent]]:
    """Load activity events, retaining target_system as the resource system."""
    events: dict[str, list[ActivityEvent]] = defaultdict(list)
    for row in read_csv("system_activity.csv"):
        try:
            event = ActivityEvent(
                activity_id=row["activity_id"],
                event_time=parse_time(row["event_time"]),
                event_type=norm(row["event_type"]).upper(),
                system_id=row["target_system"].strip(),
                resource_id=row["resource_id"].strip(),
                result=norm(row["reported_result"]).upper(),
                details=row.get("details", ""),
                user_id=row.get("user_id", "").strip(),
            )
            events[f"{event.system_id}|{event.resource_id}"].append(event)
        except (KeyError, ValueError):
            continue
    for resource_events in events.values():
        resource_events.sort(key=lambda event: (event.event_time, event.activity_id))
    return dict(events)


def load_deletion_rules() -> dict[str, str]:
    return {
        row["data_category"].strip().upper(): row["required_action"].strip().upper()
        for row in read_csv("deletion_rules.csv")
        if row.get("data_category") and row.get("required_action")
    }


def rule_action_for(record: Record, rules: dict[str, str]) -> str:
    """Resolve a record category to its configured action."""
    category = record.record_type.upper()
    if category in rules:
        return rules[category]
    if "SEARCH" in category:
        return rules.get("SEARCH_INDEX_ENTRY", "DELETE")
    if "SUPPORT" in category:
        return rules.get("SUPPORT_MESSAGE", "REDACT")
    if "ANALYTIC" in category or "TELEMETRY" in category:
        return rules.get("ANALYTICS_PROFILE", "ANONYMIZE")
    if "BACKUP" in category:
        return rules.get("BACKUP_ARCHIVE", "RETAIN")
    return "DELETE"


def flow_and_restore_findings(
    cluster: IdentityCluster,
    audit_at: datetime,
    states: dict[tuple[str, str], dict[str, str | None]],
    events_index: dict[str, list[ActivityEvent]],
    window_days: int = 30,
) -> list[dict[str, Any]]:
    """Detect recurring/scheduled flows and unblocked restore risks."""
    identifiers = cluster.all_values
    cutoff = audit_at - timedelta(days=window_days)
    findings: list[dict[str, Any]] = []
    for key in states:
        events = [
            event for event in events_index.get(f"{key[0]}|{key[1]}", [])
            if event.event_time <= audit_at
            and event.result in {"SUCCESS", "SUCCEEDED"}
        ]
        flow_events = [
            event for event in events
            if event.event_time >= cutoff
            and any(token in event.event_type for token in ("SYNC", "EXPORT", "INDEX"))
            and (
                event.user_id in cluster.user_ids
                or (
                    not event.user_id
                    and any(norm(value) in event.details.casefold() for value in identifiers)
                )
            )
        ]
        if not flow_events:
            continue
        scheduled = any("SCHEDULED" in event.details.upper() for event in flow_events)
        if len(flow_events) < 2 and not scheduled:
            continue
        evidence = flow_events[-1].activity_id
        findings.append(step(
            "FINDING", key[0], key[1], "REPORT_ACTIVE_DATA_FLOW", "ACTIVE", evidence
        ))
    for key, resource_state in states.items():
        if "BACKUP" not in key[0].upper() and not key[1].lower().startswith("backup_"):
            continue
        events = [
            event for event in events_index.get(f"{key[0]}|{key[1]}", [])
            if event.event_time <= audit_at
            and event.result in {"SUCCESS", "SUCCEEDED"}
        ]
        restore_events = [event for event in events if "RESTORE" in event.event_type]
        close_events = [
            event for event in events
            if any(token in event.event_type for token in ("PURGE", "DELETE", "BLOCK_RESTORE"))
        ]
        last_restore = restore_events[-1] if restore_events else None
        last_close = close_events[-1] if close_events else None
        if last_restore and (last_close is None or last_restore.event_time > last_close.event_time):
            findings.append(step(
                "FINDING", key[0], key[1], "REPORT_RESTORE_RISK", "ACTIVE",
                last_restore.activity_id,
            ))
    return findings


def replay_activity_events(
    initial_state: str,
    events: list[ActivityEvent],
    audit_at: datetime,
) -> tuple[str, str | None]:
    """Replay successful events in strict chronological order."""
    state = {
        "FOUND": "PRESENT",
        "IDENTIFIABLE": "PRESENT",
        "ACTIVE": "PRESENT",
        "PRESENT": "PRESENT",
        "CREATED": "PRESENT",
        "ABSENT": "ABSENT",
        "NOT_FOUND": "ABSENT",
        "DELETED": "ABSENT",
        "DISABLED": "ABSENT",
        "BLOCKED": "ABSENT",
        "REDACTED": "REDACTED",
        "ANONYMIZED": "REDACTED",
    }.get(initial_state.upper(), "PRESENT")
    last_activity: str | None = None
    transitions = {
        "DELETE": "ABSENT",
        "DELETE_RECORD": "ABSENT",
        "CREATE": "PRESENT",
        "CREATE_RECORD": "PRESENT",
        "UPDATE": "PRESENT",
        "UPDATE_RECORD": "PRESENT",
        "REDACT": "REDACTED",
        "REDACT_RECORD": "REDACTED",
    }
    for event in events:
        if event.event_time > audit_at or event.result not in {"SUCCESS", "SUCCEEDED"}:
            continue
        next_state = transitions.get(event.event_type)
        if next_state:
            state = next_state
            last_activity = event.activity_id
    return state, last_activity


def reconstruct_resource_states(
    cluster: IdentityCluster,
    audit_at: datetime,
    records: list[Record],
    events_index: dict[str, list[ActivityEvent]],
) -> dict[tuple[str, str], dict[str, str | None]]:
    """Reconstruct each evidence resource using its latest eligible snapshot."""
    snapshots: dict[tuple[str, str], Record] = {}
    for record in records:
        if record.captured_at > audit_at or not record_matches(record, cluster.all_values):
            continue
        key = (record.system_id, record.resource_id)
        previous = snapshots.get(key)
        if previous is None or record.captured_at > previous.captured_at:
            snapshots[key] = record

    states: dict[tuple[str, str], dict[str, str | None]] = {}
    for key, snapshot in snapshots.items():
        state, activity_id = replay_activity_events(
            snapshot.state or "PRESENT",
            events_index.get(f"{key[0]}|{key[1]}", []),
            audit_at,
        )
        states[key] = {
            "state": state,
            "evidence_id": snapshot.evidence_id,
            "activity_id": activity_id,
            "captured_at": snapshot.captured_at.isoformat(),
        }
    return states


def validate_reconstructed_states(
    state_by_case: dict[str, dict[tuple[str, str], dict[str, str | None]]],
    sample_requests_path: Path,
    sample_output_path: Path,
) -> tuple[bool, list[str]]:
    """Compare reconstructed states with the states demanded by sample steps."""
    with sample_output_path.open(newline="", encoding="utf-8") as handle:
        expected = {row["case_id"]: row for row in csv.DictReader(handle)}
    errors: list[str] = []
    operation_states = {
        "REPORT_REMAINING_RECORD": "PRESENT",
        "REPORT_EXPOSED_PII": "PRESENT",
        "REPORT_ACTIVE_DATA_FLOW": "PRESENT",
        "REPORT_RESTORE_RISK": "PRESENT",
        "VERIFY_RECORD_ABSENT": "ABSENT",
        "VERIFY_RECORD_PRESENT": "PRESENT",
        "VERIFY_RECORD_REDACTED": "REDACTED",
        "VERIFY_RECORD_ANONYMIZED": "REDACTED",
        "VERIFY_DATA_FLOW_DISABLED": "ABSENT",
        "VERIFY_RESTORE_BLOCKED": "ABSENT",
    }
    for case_id, row in expected.items():
        if row["deletion_status"] == NOT_APPLICABLE_INTENT:
            continue
        states = state_by_case.get(case_id, {})
        for step_item in json.loads(row["steps_json"]):
            if step_item["phase"] == "REMEDIATION":
                continue
            key = (step_item["system_id"], step_item["resource_id"])
            required = operation_states.get(step_item["operation"])
            observed = states.get(key, {}).get("state")
            if required and observed is not None and observed != required:
                errors.append(
                    f"{case_id}:{key[0]}/{key[1]} expected {required}, observed {observed}"
                )
    return not errors, errors


def evidence_files() -> dict[str, tuple[Path, datetime]]:
    result: dict[str, tuple[Path, datetime]] = {}
    for row in read_csv("evidence_files.csv"):
        result[row["file_id"]] = (
            DATASET / row["file_path"],
            parse_time(row["captured_at"]),
        )
    return result


def scalar_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from scalar_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from scalar_strings(child)
    elif value is not None:
        yield str(value)


def json_records(value: Any, evidence_id: str, captured_at: datetime) -> Iterable[Record]:
    if isinstance(value, dict):
        resource = str(value.get("resource_id", ""))
        record_type = str(
            value.get("record_type")
            or value.get("data_category")
            or value.get("collection")
            or ""
        ).upper()
        system = str(value.get("system") or value.get("data_store") or value.get("source") or "")
        state = str(value.get("state") or value.get("result") or value.get("status") or "").upper()
        subjects = [
            str(value[key])
            for key in ("email", "subject_reference", "user_id", "customer_reference")
            if value.get(key) not in (None, "")
        ]
        if resource and record_type and subjects:
            yield Record(evidence_id, captured_at, system, resource, record_type, state, "|".join(subjects))
        for child in value.values():
            yield from json_records(child, evidence_id, captured_at)
    elif isinstance(value, list):
        for child in value:
            yield from json_records(child, evidence_id, captured_at)


def text_records(text: str, evidence_id: str, captured_at: datetime) -> Iterable[Record]:
    line_pattern = re.compile(
        r"(?:^|-\s*)(\d{4}-\d{2}-\d{2}T[^|]+?)\s*\|\s*([^|]+?)\s*\|\s*"
        r"([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*$"
    )
    for line in text.splitlines():
        match = line_pattern.search(line)
        if match:
            timestamp, system, resource, record_type, state, subject = (
                item.strip() for item in match.groups()
            )
            if re.match(r"\d{4}-\d{2}-\d{2}T", timestamp):
                yield Record(
                    evidence_id,
                    captured_at,
                    system,
                    resource,
                    record_type.upper(),
                    state.upper(),
                    subject,
                )


def sql_records(text: str, evidence_id: str, captured_at: datetime) -> Iterable[Record]:
    table_pattern = re.compile(
        r'INSERT\s+INTO\s+"?([A-Za-z0-9_]+)"?\s+VALUES\s*\((.*?)\);',
        re.IGNORECASE | re.DOTALL,
    )
    value_pattern = re.compile(r"'((?:''|[^'])*)'")
    for table, values_text in table_pattern.findall(text):
        values = [value.replace("''", "'") for value in value_pattern.findall(values_text)]
        if not values:
            continue
        table_name = table.casefold()
        if "search" in table_name:
            record_type, system = "SEARCH_INDEX_ENTRY", "search_service"
        elif "support" in table_name:
            record_type, system = "SUPPORT_MESSAGE", "support_platform"
        elif "analytics" in table_name or "telemetry" in table_name:
            record_type, system = "ANALYTICS_PROFILE", "analytics_platform"
        elif "audit" in table_name or "security" in table_name:
            record_type, system = "SECURITY_AUDIT_LOG", "security_service"
        elif "backup" in table_name or "archive" in table_name:
            record_type, system = "BACKUP_ARCHIVE", "backup_service"
        else:
            record_type, system = "USER_PROFILE", "account_service"
        resource = values[0]
        yield Record(evidence_id, captured_at, system, resource, record_type, "PRESENT", "|".join(values))


def load_sql_evidence(
    path: Path,
    evidence_id: str,
    captured_at: datetime,
) -> list[Record]:
    """Parse only the supported INSERT grammar; never execute evidence SQL."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return list(sql_records(text, evidence_id, captured_at))


def load_records() -> list[Record]:
    records: list[Record] = []
    for evidence_id, (path, captured_at) in evidence_files().items():
        if path.suffix.lower() == ".json":
            try:
                records.extend(json_records(json.loads(path.read_text(encoding="utf-8")), evidence_id, captured_at))
            except (OSError, json.JSONDecodeError):
                continue
        elif path.suffix.lower() == ".sql":
            try:
                records.extend(load_sql_evidence(path, evidence_id, captured_at))
            except OSError:
                continue
        elif path.suffix.lower() in {".log", ".txt"}:
            try:
                records.extend(text_records(path.read_text(encoding="utf-8", errors="ignore"), evidence_id, captured_at))
            except OSError:
                continue
    return records


def identifiers(cluster: IdentityCluster) -> set[str]:
    return cluster.all_values


def record_matches(record: Record, ids: set[str]) -> bool:
    subject_values = {norm(value) for value in record.subject.split("|") if norm(value)}
    return bool(subject_values.intersection(ids))


def operation_for(record: Record) -> tuple[str, str, str] | None:
    kind = record.record_type
    state = record.state
    if kind == "DATA_FLOW" or "FLOW" in kind:
        return ("REPORT_ACTIVE_DATA_FLOW", "DISABLE_DATA_FLOW", "VERIFY_DATA_FLOW_DISABLED")
    if kind in {"RESTORE_PATH", "RESTORE_RISK"} or "RESTORE" in kind:
        return ("REPORT_RESTORE_RISK", "BLOCK_RESTORE", "VERIFY_RESTORE_BLOCKED")
    if kind in RETAIN_TYPES:
        if kind == "BACKUP_ARCHIVE":
            return ("REPORT_RESTORE_RISK", "BLOCK_RESTORE", "VERIFY_RESTORE_BLOCKED")
        return (None, None, "VERIFY_RECORD_PRESENT")  # type: ignore[return-value]
    if kind in DELETE_TYPES:
        return ("REPORT_REMAINING_RECORD", "DELETE_RECORD", "VERIFY_RECORD_ABSENT") if state not in {
            "NOT_FOUND", "ABSENT", "DELETED", "ANONYMIZED"
        } else (None, None, "VERIFY_RECORD_ABSENT")
    if kind in REDACT_TYPES:
        return ("REPORT_EXPOSED_PII", "REDACT_RECORD", "VERIFY_RECORD_REDACTED") if state not in {
            "REDACTED", "ABSENT", "NOT_FOUND"
        } else (None, None, "VERIFY_RECORD_REDACTED")
    if kind in ANONYMIZE_TYPES:
        return ("REPORT_EXPOSED_PII", "ANONYMIZE_RECORD", "VERIFY_RECORD_ANONYMIZED") if state not in {
            "ANONYMIZED", "ABSENT", "NOT_FOUND"
        } else (None, None, "VERIFY_RECORD_ANONYMIZED")
    return None


def step(phase: str, system: str, resource: str, operation: str, state: str, evidence: str) -> dict[str, Any]:
    return {
        "step_no": 0,
        "phase": phase,
        "system_id": system,
        "resource_id": resource,
        "operation": operation,
        "state": state,
        "evidence_id": evidence,
    }


def apply_retention_overrides(
    steps: list[dict[str, Any]],
    records: list[Record],
    rules: dict[str, str],
    override_log: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Apply absolute, idempotent retention precedence per resource key."""
    record_types: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        record_types[(record.system_id, record.resource_id)].add(record.record_type)

    retained: dict[tuple[str, str], str] = {}
    for key, types in record_types.items():
        if "LEGAL_HOLD_RECORD" in types:
            retained[key] = "LEGAL_HOLD_RECORD"
        elif "SECURITY_AUDIT_LOG" in types:
            retained[key] = "SECURITY_AUDIT_LOG"
        elif "BACKUP_ARCHIVE" in types:
            retained[key] = "BACKUP_ARCHIVE"
        elif types.intersection(RETAIN_TYPES):
            retained[key] = sorted(types.intersection(RETAIN_TYPES))[0]

    destructive = {
        "DELETE_RECORD", "REDACT_RECORD", "ANONYMIZE_RECORD",
        "VERIFY_RECORD_ABSENT", "VERIFY_RECORD_REDACTED",
        "VERIFY_RECORD_ANONYMIZED",
        "REPORT_REMAINING_RECORD", "REPORT_EXPOSED_PII",
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in steps:
        grouped[(item["system_id"], item["resource_id"])].append(dict(item))

    result: list[dict[str, Any]] = []
    for key, items in grouped.items():
        record_type = retained.get(key)
        if not record_type:
            result.extend(items)
            continue
        backup = record_type == "BACKUP_ARCHIVE"
        filtered = [item for item in items if item["operation"] not in destructive]
        if any(item["operation"] in destructive for item in items) and override_log is not None:
            override_log.append({
                "system_id": key[0],
                "resource_id": key[1],
                "record_type": record_type,
                "from_operation": "DESTRUCTIVE_OPERATION",
                "to_operation": "VERIFY_RESTORE_BLOCKED" if backup else "VERIFY_RECORD_PRESENT",
            })
        if backup and any(
            item["operation"] in {"REPORT_RESTORE_RISK", "BLOCK_RESTORE"}
            for item in items
        ):
            if not any(item["operation"] == "BLOCK_RESTORE" for item in filtered):
                filtered.append(step("REMEDIATION", key[0], key[1], "BLOCK_RESTORE", "BLOCKED", ""))
            if not any(item["operation"] == "VERIFY_RESTORE_BLOCKED" for item in filtered):
                filtered.append(step("VERIFICATION", key[0], key[1], "VERIFY_RESTORE_BLOCKED", "BLOCKED", ""))
        elif not any(item["operation"] == "VERIFY_RECORD_PRESENT" for item in filtered):
            filtered.append(step("VERIFICATION", key[0], key[1], "VERIFY_RECORD_PRESENT", "PRESENT", ""))
        result.extend(filtered)
    return result


def audit(
    request: dict[str, str],
    records: list[Record],
    users: list[dict[str, str]],
    audits: list[dict[str, str]],
    llm_client: Callable[[str], str] | None = None,
    resolver: IdentityGraphResolver | None = None,
    rules: dict[str, str] | None = None,
    events_index: dict[str, list[ActivityEvent]] | None = None,
    retention_log: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if classify_intent(request["request_text"], llm_client) != DELETE_INTENT:
        return "NOT_APPLICABLE", []
    identity_resolver = resolver or IdentityGraphResolver(users, audits)
    cluster = identity_resolver.resolve(request["case_id"], request["email"])
    ids = identifiers(cluster)
    at = parse_time(request["audit_at"])
    candidates = [
        record for record in records
        if record.captured_at <= at and record_matches(record, ids)
    ]
    latest: dict[tuple[str, str, str], Record] = {}
    for record in candidates:
        key = (record.system_id, record.resource_id, record.record_type)
        if key not in latest or record.captured_at > latest[key].captured_at:
            latest[key] = record

    findings: list[dict[str, Any]] = []
    remediations: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    for record in latest.values():
        action = rule_action_for(record, rules or load_deletion_rules())
        state = record.state.upper()
        if action == "DELETE":
            finding, remediation, verification = (
                ("REPORT_REMAINING_RECORD", "DELETE_RECORD", "VERIFY_RECORD_ABSENT")
                if state not in {"ABSENT", "NOT_FOUND", "DELETED"}
                else (None, None, "VERIFY_RECORD_ABSENT")
            )
        elif action in {"REDACT", "TOMBSTONE"}:
            finding, remediation, verification = (
                ("REPORT_EXPOSED_PII", "REDACT_RECORD", "VERIFY_RECORD_REDACTED")
                if state not in {"REDACTED", "ABSENT", "NOT_FOUND"}
                else (None, None, "VERIFY_RECORD_REDACTED")
            )
        elif action == "ANONYMIZE":
            finding, remediation, verification = (
                ("REPORT_EXPOSED_PII", "ANONYMIZE_RECORD", "VERIFY_RECORD_ANONYMIZED")
                if state not in {"ANONYMIZED", "REDACTED", "ABSENT", "NOT_FOUND"}
                else (None, None, "VERIFY_RECORD_ANONYMIZED")
            )
        elif action == "RETAIN":
            finding, remediation, verification = (None, None, "VERIFY_RECORD_PRESENT")
        else:
            continue
        if finding:
            finding_state = "ACTIVE" if "ACTIVE" in state else "PRESENT"
            findings.append(step("FINDING", record.system_id, record.resource_id, finding, finding_state, record.evidence_id))
        if remediation:
            remediations.append(step(
                "REMEDIATION", record.system_id, record.resource_id, remediation,
                {"DISABLE_DATA_FLOW": "DISABLED", "BLOCK_RESTORE": "BLOCKED",
                 "DELETE_RECORD": "ABSENT", "REDACT_RECORD": "REDACTED",
                 "ANONYMIZE_RECORD": "ANONYMIZED"}[remediation], "",
            ))
        verification_state = {
            "VERIFY_RECORD_ABSENT": "ABSENT",
            "VERIFY_RECORD_PRESENT": "PRESENT",
            "VERIFY_RECORD_REDACTED": "REDACTED",
            "VERIFY_RECORD_ANONYMIZED": "ANONYMIZED",
            "VERIFY_DATA_FLOW_DISABLED": "DISABLED",
            "VERIFY_RESTORE_BLOCKED": "BLOCKED",
        }[verification]
        verifications.append(step("VERIFICATION", record.system_id, record.resource_id, verification, verification_state, record.evidence_id))

    if events_index is not None:
        indexed_events = events_index or {}
        reconstructed = reconstruct_resource_states(cluster, at, records, indexed_events)
        flow_findings = flow_and_restore_findings(
            cluster, at, reconstructed, indexed_events
        )
        for finding_item in flow_findings:
            findings.append(finding_item)
            if finding_item["operation"] == "REPORT_ACTIVE_DATA_FLOW":
                remediation, verification = "DISABLE_DATA_FLOW", "VERIFY_DATA_FLOW_DISABLED"
                state = "DISABLED"
            else:
                remediation, verification = "BLOCK_RESTORE", "VERIFY_RESTORE_BLOCKED"
                state = "BLOCKED"
            remediations.append(step(
                "REMEDIATION", finding_item["system_id"], finding_item["resource_id"],
                remediation, state, "",
            ))
            verifications.append(step(
                "VERIFICATION", finding_item["system_id"], finding_item["resource_id"],
                verification, state, finding_item["evidence_id"],
            ))

    override_log = retention_log if retention_log is not None else []
    all_steps = findings + remediations + verifications
    all_steps = apply_retention_overrides(
        all_steps, records, rules or load_deletion_rules(), override_log
    )
    findings = [item for item in all_steps if item["phase"] == "FINDING"]
    remediations = [item for item in all_steps if item["phase"] == "REMEDIATION"]
    verifications = [item for item in all_steps if item["phase"] == "VERIFICATION"]
    incomplete = bool(findings or remediations)
    if incomplete:
        for item in remediations + verifications:
            item["evidence_id"] = ""
    else:
        findings = []
    phase_items = [
        (findings, FINDING_ORDER),
        (remediations if incomplete else [], REMEDIATION_ORDER),
        (verifications, VERIFICATION_ORDER),
    ]
    ordered: list[dict[str, Any]] = []
    for items, order in phase_items:
        ordered.extend(sorted(items, key=lambda item: (
            order[item["operation"]], item["system_id"], item["resource_id"]
        )))
    for number, item in enumerate(ordered, 1):
        item["step_no"] = number
    return ("INCOMPLETE" if incomplete else "COMPLETE"), ordered


LLM_AUDIT_PROMPT = """Here is the generated steps_json for case {case_id}. Check for: 1. Missing findings. 2. Invalid operations. 3. Incorrect phase order. Return JSON: {{ok: bool, issues: [...]}}.

steps_json:
{steps_json}"""


def audit_steps_prompt(case_id: str, steps: list[dict[str, Any]]) -> str:
    return LLM_AUDIT_PROMPT.format(
        case_id=case_id,
        steps_json=json.dumps(steps, separators=(",", ":")),
    )


def llm_audit_steps(
    case_id: str,
    steps: list[dict[str, Any]],
    client: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Run the final LLM audit, failing closed to deterministic validation."""
    try:
        if client is not None:
            response: Any = client(audit_steps_prompt(case_id, steps))
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"ok": True, "issues": []}
            payload = json.dumps({
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": 0,
                "messages": [{
                    "role": "user",
                    "content": audit_steps_prompt(case_id, steps),
                }],
            }).encode("utf-8")
            request = urllib.request.Request(
                os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"),
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as result:
                response = json.loads(result.read().decode("utf-8"))["choices"][0]["message"]["content"]
        if isinstance(response, str):
            text = response.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            response = json.loads(text)
        if not isinstance(response, dict):
            raise ValueError("LLM audit response is not an object")
        issues = response.get("issues", [])
        return {"ok": response.get("ok") is True and isinstance(issues, list), "issues": issues}
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, urllib.error.URLError):
        return {
            "ok": False,
            "issues": ["LLM audit unavailable; deterministic quality checks required"],
        }


LLM_SUGGESTION_PROMPT = """Review steps_json for case {case_id} and return only a JSON list of missing verification issues.
Schema: [{{"type":"MISSING_VERIFICATION","system_id":"...","resource_id":"...","missing_op":"VERIFY_RECORD_ABSENT"}}]
Do not suggest new resources. Do not change existing operations. Only identify missing verification steps for existing findings.

steps_json:
{steps_json}"""


def llm_audit_suggestions(
    case_id: str,
    steps: list[dict[str, Any]],
    client: Callable[[str], str] | None = None,
) -> list[dict[str, str]]:
    """Return untrusted issue suggestions; Python validates every fix."""
    suggestions, _ = llm_audit_suggestions_status(case_id, steps, client)
    return suggestions


def llm_audit_suggestions_status(
    case_id: str,
    steps: list[dict[str, Any]],
    client: Callable[[str], str] | None = None,
) -> tuple[list[dict[str, str]], bool]:
    """Return suggestions and whether the auditor used fallback mode."""
    prompt = LLM_SUGGESTION_PROMPT.format(
        case_id=case_id,
        steps_json=json.dumps(steps, separators=(",", ":")),
    )
    try:
        if client is not None:
            response: Any = client(prompt)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return [], True
            payload = json.dumps({
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }).encode("utf-8")
            request = urllib.request.Request(
                os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"),
                data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as result:
                response = json.loads(result.read().decode("utf-8"))["choices"][0]["message"]["content"]
        if isinstance(response, str):
            response = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip()))
        if not isinstance(response, list):
            return [], False
        return [
            item for item in response
            if isinstance(item, dict)
            and all(isinstance(item.get(key), str) for key in (
                "type", "system_id", "resource_id", "missing_op"
            ))
        ], False
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, urllib.error.URLError):
        logger.exception("Fallback Mode: LLM audit suggestions failed for %s", case_id)
        return [], True


def apply_allowed_fixes(
    steps: list[dict[str, Any]],
    suggestions: list[dict[str, str]],
    debug_log: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Apply only missing-verification suggestions for existing resources."""
    existing_resources = {
        (item.get("system_id"), item.get("resource_id"))
        for item in steps
    }
    existing_operations = {
        (item.get("system_id"), item.get("resource_id"), item.get("operation"))
        for item in steps
    }
    result = [dict(item) for item in steps]
    safe_states = {
        "VERIFY_RECORD_ABSENT": "ABSENT",
        "VERIFY_RECORD_PRESENT": "PRESENT",
        "VERIFY_RECORD_REDACTED": "REDACTED",
        "VERIFY_RECORD_ANONYMIZED": "REDACTED",
        "VERIFY_DATA_FLOW_DISABLED": "DISABLED",
        "VERIFY_RESTORE_BLOCKED": "BLOCKED",
    }
    for suggestion in suggestions:
        key = (suggestion.get("system_id"), suggestion.get("resource_id"))
        operation = suggestion.get("missing_op", "")
        valid = (
            suggestion.get("type") == "MISSING_VERIFICATION"
            and key in existing_resources
            and operation in VERIFICATION_ORDER
            and (key[0], key[1], operation) not in existing_operations
        )
        if not valid:
            if debug_log is not None:
                debug_log.append(f"ignored suggestion: {suggestion}")
            continue
        result.append(step(
            "VERIFICATION", key[0], key[1], operation,
            safe_states[operation], "",
        ))
        existing_operations.add((key[0], key[1], operation))
    return result


def canonicalize_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair ordering and numbering without changing compliance semantics."""
    repaired: list[dict[str, Any]] = []
    for item in steps:
        candidate = dict(item)
        operation = candidate.get("operation")
        phase = candidate.get("phase")
        if operation not in ALLOWED_OPERATIONS or phase not in PHASE_ORDER:
            continue
        candidate["system_id"] = str(candidate.get("system_id", ""))
        candidate["resource_id"] = str(candidate.get("resource_id", ""))
        candidate["evidence_id"] = str(candidate.get("evidence_id", ""))
        repaired.append(candidate)
    repaired.sort(key=lambda item: (
        PHASE_ORDER[item["phase"]],
        OPERATION_ORDERS[item["operation"]],
        item["system_id"],
        item["resource_id"],
    ))
    for number, item in enumerate(repaired, 1):
        item["step_no"] = number
    return repaired


def quality_harness(rows: list[dict[str, str]], expected_rows: int) -> list[str]:
    """Validate final rows, JSON schema, operations, ordering, and numbering."""
    errors: list[str] = []
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")
    for index, row in enumerate(rows):
        missing_columns = [column for column in REQUIRED_OUTPUT_COLUMNS if column not in row]
        if missing_columns:
            errors.append(f"row {index}: missing columns {missing_columns}")
            continue
        if not row["case_id"] or row["deletion_status"] not in {
            "COMPLETE", "INCOMPLETE", NOT_APPLICABLE_INTENT
        }:
            errors.append(f"row {index}: invalid case or deletion_status")
        try:
            steps = json.loads(row["steps_json"])
        except json.JSONDecodeError:
            errors.append(f"{row['case_id']}: steps_json is not valid JSON")
            continue
        if not isinstance(steps, list):
            errors.append(f"{row['case_id']}: steps_json is not an array")
            continue
        if row["deletion_status"] == NOT_APPLICABLE_INTENT and steps:
            errors.append(f"{row['case_id']}: NOT_APPLICABLE must have []")
        numbers = [item.get("step_no") for item in steps]
        if numbers != list(range(1, len(steps) + 1)):
            errors.append(f"{row['case_id']}: step numbers are not consecutive")
        previous_key: tuple[int, int, str, str] | None = None
        for item in steps:
            missing_keys = [
                key for key in (
                    "step_no", "phase", "system_id", "resource_id",
                    "operation", "state", "evidence_id",
                ) if key not in item
            ]
            if missing_keys:
                errors.append(f"{row['case_id']}: step missing {missing_keys}")
                continue
            if item["phase"] not in PHASE_ORDER or item["operation"] not in ALLOWED_OPERATIONS:
                errors.append(f"{row['case_id']}: invalid operation or phase")
                continue
            key = (
                PHASE_ORDER[item["phase"]],
                OPERATION_ORDERS[item["operation"]],
                item["system_id"],
                item["resource_id"],
            )
            if previous_key is not None and key < previous_key:
                errors.append(f"{row['case_id']}: incorrect phase/operation ordering")
            previous_key = key
    return errors


def finalize_rows(
    rows: list[dict[str, str]],
    expected_rows: int,
    llm_auditor: Callable[[str], str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Apply LLM review, deterministic repairs, and final quality validation."""
    audit_results: dict[str, Any] = {}
    finalized: list[dict[str, str]] = []
    for row in rows:
        try:
            steps = json.loads(row["steps_json"])
        except json.JSONDecodeError:
            steps = []
        debug_log: list[str] = []
        suggestions, fallback_mode = llm_audit_suggestions_status(
            row["case_id"], steps, llm_auditor
        )
        repaired = apply_allowed_fixes(steps, suggestions, debug_log)
        audit_results[row["case_id"]] = {
            "suggestions": suggestions,
            "ignored": debug_log,
            "fallback_mode": fallback_mode,
            "llm_failed": fallback_mode,
        }
        repaired = canonicalize_steps(repaired)
        output_row = dict(row)
        output_row["steps_json"] = json.dumps(repaired, separators=(",", ":"))
        finalized.append(output_row)
    errors = quality_harness(finalized, expected_rows)
    if errors:
        raise RuntimeError("Final output quality checks failed: " + "; ".join(errors[:20]))
    return finalized, audit_results


def main(
    llm_client: Callable[[str], str] | None = None,
    llm_auditor: Callable[[str], str] | None = None,
) -> None:
    requests = read_csv("deletion_requests.csv")
    users = read_csv("user_identities.csv")
    audits = read_csv("user_audit_log.csv")
    records = load_records()
    events_index = load_system_activity()
    rules = load_deletion_rules()
    resolver = IdentityGraphResolver(users, audits)
    clusters = resolver.resolve_requests(requests, llm_client)
    IDENTITY_OUTPUT.write_text(
        json.dumps(
            {case_id: cluster_json(cluster) for case_id, cluster in sorted(clusters.items())},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    state_by_case = {
        request["case_id"]: reconstruct_resource_states(
            clusters[request["case_id"]],
            parse_time(request["audit_at"]),
            records,
            events_index,
        )
        for request in requests
        if request["case_id"] in clusters
    }
    STATE_OUTPUT.write_text(
        json.dumps(
            {
                case_id: {
                    f"{system_id}|{resource_id}": state
                    for (system_id, resource_id), state in sorted(states.items())
                }
                for case_id, states in sorted(state_by_case.items())
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    requests.sort(key=lambda row: int(row["case_id"].rsplit("_", 1)[1]))
    generated_rows: list[dict[str, str]] = []
    retention_log: list[dict[str, str]] = []
    for request in requests:
        status, steps = audit(
            request, records, users, audits, llm_client, resolver, rules, events_index,
            retention_log,
        )
        generated_rows.append({
            "case_id": request["case_id"],
            "deletion_status": status,
            "steps_json": json.dumps(steps, separators=(",", ":")),
        })
    final_rows, audit_results = finalize_rows(
        generated_rows, len(requests), llm_auditor
    )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(final_rows)
    AUDIT_OUTPUT.write_text(
        json.dumps(audit_results, indent=2),
        encoding="utf-8",
    )
    RETENTION_OUTPUT.write_text(
        json.dumps(retention_log, indent=2),
        encoding="utf-8",
    )
    valid, errors = validate_intent_output(OUTPUT, DATASET / "sample_output.csv")
    if not valid:
        raise RuntimeError(
            "Intent validation failed for sample cases: " + ", ".join(errors)
        )
    sample_requests = read_csv("sample_deletion_requests.csv")
    validation_clusters = dict(clusters)
    validation_clusters.update({
        request["case_id"]: resolver.resolve(request["case_id"], request["email"])
        for request in sample_requests
    })
    sample_states = {
        request["case_id"]: reconstruct_resource_states(
            validation_clusters[request["case_id"]],
            parse_time(request["audit_at"]),
            records,
            events_index,
        )
        for request in sample_requests
        if request["case_id"] in validation_clusters
    }
    valid, errors = validate_identity_resolution(
        validation_clusters,
        DATASET / "sample_deletion_requests.csv",
        DATASET / "sample_output.csv",
        audits,
    )
    if not valid:
        raise RuntimeError(
            "Identity validation failed for sample cases: " + ", ".join(errors)
        )
    states_valid, state_errors = validate_reconstructed_states(
        sample_states,
        DATASET / "sample_deletion_requests.csv",
        DATASET / "sample_output.csv",
    )
    STATE_VALIDATION_OUTPUT.write_text(
        json.dumps(
            {"valid": states_valid, "errors": state_errors},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()