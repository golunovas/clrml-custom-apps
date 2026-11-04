'''
List Dataset Updater app.

One-shot ClearML app: each launch validates a list of IDs, merges them into the
latest version of a configured ClearML dataset and publishes a new finalized
version holding a full snapshot of the list in ids.csv.

The list is append-only: IDs that are already present are skipped.
'''
import argparse
import csv
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Set, Tuple

from clearml import Dataset, Task, TaskTypes

CSV_FILE_NAME = 'ids.csv'
CSV_COLUMNS = ['id', 'added_at', 'added_by', 'reference', 'app_task_id']
APP_TAG = 'list-dataset-updater'
DEFAULT_ID_PATTERN = r'^[A-Za-z0-9_-]+$'
SEPARATORS_RE = re.compile(r'[\s,;]+')
MAX_MERGE_ATTEMPTS = 3

Row = Dict[str, str]


# ---------------------------------------------------------------------------
# Pure helpers (no ClearML calls)
# ---------------------------------------------------------------------------

def parse_ids(raw: str, id_pattern: Pattern, lowercase: bool) -> Tuple[List[str], List[str], List[str]]:
    '''
    Split the raw wizard input into IDs.

    :param raw: Text with IDs separated by whitespace, commas or semicolons. Quotes are ignored.
    :param id_pattern: Compiled regex every (normalized) ID must fully match.
    :param lowercase: Normalize IDs to lowercase before validating.
    :return: (valid IDs in input order, de-duplicated,
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
        item_id = token.lower() if lowercase else token
        if not id_pattern.fullmatch(item_id):
            invalid.append(token)
        elif item_id in seen:
            if item_id not in duplicates:
                duplicates.append(item_id)
        else:
            seen.add(item_id)
            valid.append(item_id)
    return valid, invalid, duplicates


def read_rows(path: str) -> Dict[str, Row]:
    '''Read a list CSV into {id: row}. The first occurrence of an ID wins.'''
    rows: Dict[str, Row] = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            item_id = (row.get('id') or '').strip()
            if item_id and item_id not in rows:
                rows[item_id] = {col: row.get(col, '') or '' for col in CSV_COLUMNS}
                rows[item_id]['id'] = item_id
    return rows


def write_rows(path: str, rows: Dict[str, Row]) -> None:
    '''Write the list CSV, sorted by added_at then id for stable diffs.'''
    ordered = sorted(rows.values(), key=lambda r: (r['added_at'], r['id']))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)


def merge_rows(*snapshots: Dict[str, Row]) -> Dict[str, Row]:
    '''Union of several snapshots. For an ID present in more than one, keep the earliest entry.'''
    merged: Dict[str, Row] = {}
    for snapshot in snapshots:
        for item_id, row in snapshot.items():
            current = merged.get(item_id)
            if current is None or row['added_at'] < current['added_at']:
                merged[item_id] = row
    return merged


def new_rows(ids: Iterable[str], added_by: str, reference: str, task_id: str) -> Dict[str, Row]:
    added_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return {
        item_id: {
            'id': item_id,
            'added_at': added_at,
            'added_by': added_by,
            'reference': reference,
            'app_task_id': task_id,
        }
        for item_id in ids
    }


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


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
    '''All completed, non-archived versions of the dataset (exact name match), oldest first.'''
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
    '''Latest completed version, resolved the same way Dataset.get resolves it for any consumer.'''
    if not list_completed_versions(project, name):
        return None
    return Dataset.get(dataset_project=project, dataset_name=name, only_completed=True)


def load_snapshot(dataset: Dataset) -> Dict[str, Row]:
    path = os.path.join(dataset.get_local_copy(), CSV_FILE_NAME)
    return read_rows(path) if os.path.exists(path) else {}


def resolve_user(task: Task) -> str:
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
    print(f'Published {project}/{name} version {dataset.version} ({dataset.id}) with {len(rows)} IDs')
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

        print(f'Concurrent versions detected; merging {len(union) - len(latest_rows)} missing IDs')
        merged = publish_version(project, name, [latest] + others, union, [APP_TAG, 'merge'])
        parent_created = version_created(project, name, merged.id)
        # Loop again: another run may have finished while we were merging.
    raise RuntimeError(f'Could not reconcile concurrent versions after {MAX_MERGE_ATTEMPTS} attempts')


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(task: Task, summary: List[List[str]], added: List[str], rejected: List[List[str]]) -> None:
    logger = task.get_logger()
    logger.report_table('List update', 'Summary', iteration=0, table_plot=[['Item', 'Value']] + summary)
    if added:
        logger.report_table('List update', 'Added', iteration=0, table_plot=[['id']] + [[i] for i in added])
    if rejected:
        logger.report_table('List update', 'Rejected', iteration=0, table_plot=[['input', 'reason']] + rejected)
    for item, value in summary:
        print(f'  {item}: {value}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # fmt: off
    parser = argparse.ArgumentParser(description='Append IDs to a versioned ClearML list dataset')
    parser.add_argument('--ids', type=str, default='', help='IDs separated by whitespace, commas or semicolons')
    parser.add_argument('--reference', type=str, default='', help='Reference stored with every added ID')
    parser.add_argument('--dataset_project', type=str, default='', help='Target dataset project')
    parser.add_argument('--dataset_name', type=str, default='', help='Target dataset name')
    parser.add_argument('--id_pattern', type=str, default=DEFAULT_ID_PATTERN, help='Regex every ID must match')
    parser.add_argument('--lowercase', type=str, default='false', help='Normalize IDs to lowercase (true/false)')
    args = parser.parse_args()
    # fmt: on

    project, name = args.dataset_project.strip(), args.dataset_name.strip()
    task = Task.init(project_name=project or 'DevOps', task_name='List dataset update',
                     task_type=TaskTypes.data_processing)

    if not project or not name:
        report(task, [['Status', 'Rejected: dataset_project and dataset_name must be configured']], [], [])
        return 1

    reference = args.reference.strip()
    id_pattern = re.compile(args.id_pattern or DEFAULT_ID_PATTERN)
    valid, invalid, duplicates = parse_ids(args.ids, id_pattern, parse_bool(args.lowercase))
    rejected = [[token, 'does not match the configured ID pattern'] for token in invalid]

    if not reference:
        report(task, [['Status', 'Rejected: reference is required']], [], rejected)
        return 1
    if invalid:
        # Reject the whole request so a typo is fixed instead of silently skipped.
        report(task, [['Status', f'Rejected: {len(invalid)} invalid IDs, nothing was added'],
                      ['Valid IDs in input', str(len(valid))]], [], rejected)
        return 1
    if not valid:
        report(task, [['Status', 'Rejected: no IDs given']], [], rejected)
        return 1

    added_by = resolve_user(task)

    parent = get_latest(project, name)
    existing = load_snapshot(parent) if parent else {}
    to_add = [i for i in valid if i not in existing]
    already = [i for i in valid if i in existing]

    summary = [
        ['Added by', added_by],
        ['Reference', reference],
        ['IDs in input', str(len(valid))],
        ['Duplicates in input (ignored)', str(len(duplicates))],
        ['Already in list', str(len(already))],
        ['Newly added', str(len(to_add))],
    ]

    if not to_add:
        summary.insert(0, ['Status', 'Nothing to add: all IDs are already in the list'])
        summary.append(['Dataset version', f'{parent.version} ({parent.id})'])
        report(task, summary, [], rejected)
        return 0

    rows = merge_rows(existing, new_rows(to_add, added_by, reference, task.id))
    published = publish_version(project, name, [parent] if parent else [], rows, [APP_TAG])

    parent_created = version_created(project, name, parent.id) if parent is not None else None
    merged = reconcile_concurrent_versions(project, name, parent_created)
    final = merged or published

    task.set_user_properties(
        list_dataset_id=final.id,
        list_dataset_version=final.version,
        added_count=len(to_add),
    )
    summary.insert(0, ['Status', 'Done'])
    summary.append(['Dataset version', f'{final.version} ({final.id})'])
    summary.append(['Total IDs in list', str(len(load_snapshot(final)) if merged else len(rows))])
    report(task, summary, to_add, rejected)
    return 0


if __name__ == '__main__':
    sys.exit(main())
