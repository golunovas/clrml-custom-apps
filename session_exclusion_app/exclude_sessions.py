'''
Session Exclusion app.

One-shot ClearML app: each launch is one exclusion request. It validates the
given session IDs, merges them into the latest version of the
compliance/excluded_sessions dataset and publishes a new (finalized) version
that contains a full snapshot of the blocklist in excluded_sessions.csv.

The dataset is append-only. Training tasks fetch the latest completed version
and filter out every session listed in it.
'''
import argparse
import csv
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from clearml import Dataset, Task, TaskTypes

CSV_FILE_NAME = 'excluded_sessions.csv'
CSV_COLUMNS = ['session_id', 'added_at', 'requested_by', 'ticket', 'app_task_id']
APP_TAG = 'session-exclusion-app'
OBJECT_ID_RE = re.compile(r'^[0-9a-f]{24}$')
SEPARATORS_RE = re.compile(r'[\s,;]+')
MAX_MERGE_ATTEMPTS = 3

Row = Dict[str, str]


# ---------------------------------------------------------------------------
# Pure helpers (no ClearML calls)
# ---------------------------------------------------------------------------

def parse_session_ids(raw: str) -> Tuple[List[str], List[str], List[str]]:
    '''
    Split the raw wizard input into session IDs.

    :param raw: Text with IDs separated by whitespace, commas or semicolons. Quotes are ignored.
    :return: (valid IDs in input order, lowercase and de-duplicated,
              invalid tokens as typed,
              IDs that appeared more than once in the input)
    '''
    valid: List[str] = []
    seen: Set[str] = set()
    invalid: List[str] = []
    duplicates: List[str] = []
    for token in SEPARATORS_RE.split(raw or ''):
        token = token.strip().strip('"\'')
        if not token:
            continue
        session_id = token.lower()
        if not OBJECT_ID_RE.match(session_id):
            invalid.append(token)
        elif session_id in seen:
            if session_id not in duplicates:
                duplicates.append(session_id)
        else:
            seen.add(session_id)
            valid.append(session_id)
    return valid, invalid, duplicates


def read_rows(path: str) -> Dict[str, Row]:
    '''Read a blocklist CSV into {session_id: row}. The first occurrence of an ID wins.'''
    rows: Dict[str, Row] = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            session_id = (row.get('session_id') or '').strip().lower()
            if session_id and session_id not in rows:
                rows[session_id] = {col: row.get(col, '') or '' for col in CSV_COLUMNS}
                rows[session_id]['session_id'] = session_id
    return rows


def write_rows(path: str, rows: Dict[str, Row]) -> None:
    '''Write the blocklist CSV, sorted by added_at then session_id for stable diffs.'''
    ordered = sorted(rows.values(), key=lambda r: (r['added_at'], r['session_id']))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)


def merge_rows(*snapshots: Dict[str, Row]) -> Dict[str, Row]:
    '''Union of several snapshots. For an ID present in more than one, keep the earliest entry.'''
    merged: Dict[str, Row] = {}
    for snapshot in snapshots:
        for session_id, row in snapshot.items():
            current = merged.get(session_id)
            if current is None or row['added_at'] < current['added_at']:
                merged[session_id] = row
    return merged


def new_rows(session_ids: Iterable[str], requested_by: str, ticket: str, task_id: str) -> Dict[str, Row]:
    added_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return {
        session_id: {
            'session_id': session_id,
            'added_at': added_at,
            'requested_by': requested_by,
            'ticket': ticket,
            'app_task_id': task_id,
        }
        for session_id in session_ids
    }


# ---------------------------------------------------------------------------
# ClearML helpers
# ---------------------------------------------------------------------------

def _created(entry: dict) -> datetime:
    value = entry.get('created')
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    return datetime.min.replace(tzinfo=timezone.utc)


def list_completed_versions(project: str, name: str) -> List[dict]:
    '''All completed versions of the dataset (exact name match), oldest first.'''
    versions = Dataset.list_datasets(dataset_project=project, partial_name=name, only_completed=True,
                                     include_archived=False)
    versions = [v for v in versions if v.get('name') == name and v.get('project') in (project, None)]
    return sorted(versions, key=_created)


def version_created(project: str, name: str, dataset_id: str) -> Optional[datetime]:
    '''Creation time of a completed version, as reported by the dataset listing.'''
    for v in list_completed_versions(project, name):
        if v.get('id') == dataset_id:
            return _created(v)
    return None


def get_latest(project: str, name: str) -> Optional[Dataset]:
    '''Latest completed version, resolved exactly the way training tasks resolve it.'''
    if not list_completed_versions(project, name):
        return None
    return Dataset.get(dataset_project=project, dataset_name=name, only_completed=True)


def load_snapshot(dataset: Dataset) -> Dict[str, Row]:
    path = os.path.join(dataset.get_local_copy(), CSV_FILE_NAME)
    return read_rows(path) if os.path.exists(path) else {}


def resolve_requester(task: Task) -> str:
    '''Name of the user who launched this app instance, falling back to the user ID.'''
    user_id = getattr(task.data, 'user', None) or ''
    try:
        from clearml.backend_api.session.client import APIClient
        response = APIClient().users.get_by_id(user=user_id)
        user = getattr(response, 'user', response)
        name = getattr(user, 'name', None)
        if name:
            return name
    except Exception as ex:
        print(f'Could not resolve user name for {user_id}: {ex}')
    return user_id or 'unknown'


def files_server_url() -> Optional[str]:
    try:
        from clearml.backend_api import Session
        return Session.get_files_server_host()
    except Exception as ex:
        print(f'Could not resolve the files server URL, using the SDK default: {ex}')
        return None


def publish_version(
        project: str,
        name: str,
        parents: Sequence[Dataset],
        rows: Dict[str, Row],
        tags: Sequence[str],
) -> Dataset:
    '''Create, upload and finalize a new dataset version holding the full snapshot.'''
    dataset = Dataset.create(
        dataset_name=name,
        dataset_project=project,
        parent_datasets=[p.id for p in parents] or None,
        dataset_tags=list(tags),
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_rows(os.path.join(tmp, CSV_FILE_NAME), rows)
        dataset.add_files(tmp)
        dataset.upload(output_url=files_server_url())
    dataset.finalize()
    print(f'Published {project}/{name} version {dataset.version} ({dataset.id}) with {len(rows)} sessions')
    return dataset


def reconcile_concurrent_versions(project: str, name: str, parent_created: Optional[datetime]) -> Optional[Dataset]:
    '''
    Guard against two app runs that started from the same parent version.

    Every completed version created after our parent is a full snapshot. The latest version
    must contain the union of all of them; if it doesn't, publish a merge version on top.
    Each run performs this check after finalizing, so the last run to finish sees all others.

    :return: The merge version, if one had to be created.
    '''
    merged: Optional[Dataset] = None
    for _ in range(MAX_MERGE_ATTEMPTS):
        versions = list_completed_versions(project, name)
        if parent_created is not None:
            versions = [v for v in versions if _created(v) > parent_created]
        if len(versions) <= 1:
            return merged

        latest = get_latest(project, name)
        latest_rows = load_snapshot(latest)
        others = [Dataset.get(dataset_id=v['id']) for v in versions if v['id'] != latest.id]
        union = merge_rows(latest_rows, *(load_snapshot(d) for d in others))
        if set(union) <= set(latest_rows):
            return merged

        print(f'Concurrent versions detected; merging {len(union) - len(latest_rows)} missing sessions')
        merged = publish_version(project, name, [latest] + others, union, [APP_TAG, 'merge'])
        parent_created = version_created(project, name, merged.id)
        # Loop again: another run may have finished while we were merging.
    raise RuntimeError(f'Could not reconcile concurrent versions after {MAX_MERGE_ATTEMPTS} attempts')


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(task: Task, summary: List[List[str]], added: List[str], rejected: List[List[str]]) -> None:
    logger = task.get_logger()
    logger.report_table('Session exclusion', 'Summary', iteration=0, table_plot=[['Item', 'Value']] + summary)
    if added:
        logger.report_table('Session exclusion', 'Added', iteration=0,
                            table_plot=[['session_id']] + [[s] for s in added])
    if rejected:
        logger.report_table('Session exclusion', 'Rejected', iteration=0,
                            table_plot=[['input', 'reason']] + rejected)
    for item, value in summary:
        print(f'  {item}: {value}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # fmt: off
    parser = argparse.ArgumentParser(description='Add sessions to the ML consent exclusion dataset')
    parser.add_argument('--session_ids', type=str, default='', help='Session IDs separated by whitespace or commas')
    parser.add_argument('--ticket', type=str, default='', help='Ticket ID or reason for the exclusion')
    parser.add_argument('--dataset_project', type=str, default='compliance')
    parser.add_argument('--dataset_name', type=str, default='excluded_sessions')
    args = parser.parse_args()
    # fmt: on

    task = Task.init(project_name='compliance', task_name='Session exclusion', task_type=TaskTypes.data_processing)

    ticket = args.ticket.strip()
    valid, invalid, duplicates = parse_session_ids(args.session_ids)
    rejected = [[token, 'not a 24-character hex ObjectId'] for token in invalid]

    if not ticket:
        report(task, [['Status', 'Rejected: ticket / reason is required']], [], rejected)
        return 1
    if invalid:
        # Reject the whole request so a typo is fixed instead of silently skipped.
        report(task, [['Status', f'Rejected: {len(invalid)} invalid IDs, nothing was added'],
                      ['Valid IDs in input', str(len(valid))]], [], rejected)
        return 1
    if not valid:
        report(task, [['Status', 'Rejected: no session IDs given']], [], rejected)
        return 1

    project, name = args.dataset_project, args.dataset_name
    requester = resolve_requester(task)

    parent = get_latest(project, name)
    existing = load_snapshot(parent) if parent else {}
    to_add = [s for s in valid if s not in existing]
    already = [s for s in valid if s in existing]

    summary = [
        ['Requested by', requester],
        ['Ticket / reason', ticket],
        ['IDs in input', str(len(valid))],
        ['Duplicates in input (ignored)', str(len(duplicates))],
        ['Already excluded', str(len(already))],
        ['Newly added', str(len(to_add))],
    ]

    if not to_add:
        summary.insert(0, ['Status', 'Nothing to add: all sessions were already excluded'])
        summary.append(['Dataset version', f'{parent.version} ({parent.id})'])
        report(task, summary, [], rejected)
        return 0

    rows = merge_rows(existing, new_rows(to_add, requester, ticket, task.id))
    published = publish_version(project, name, [parent] if parent else [], rows, [APP_TAG])

    parent_created = version_created(project, name, parent.id) if parent is not None else None
    merged = reconcile_concurrent_versions(project, name, parent_created)
    final = merged or published

    task.set_user_properties(
        excluded_sessions_dataset=final.id,
        excluded_sessions_version=final.version,
        added_count=len(to_add),
    )
    summary.insert(0, ['Status', 'Done'])
    summary.append(['Dataset version', f'{final.version} ({final.id})'])
    summary.append(['Total excluded sessions', str(len(load_snapshot(final)) if merged else len(rows))])
    report(task, summary, to_add, rejected)
    return 0


if __name__ == '__main__':
    sys.exit(main())
