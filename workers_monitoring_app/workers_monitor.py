import argparse
import json
from dataclasses import dataclass
from datetime import datetime, date, timezone
from time import time, sleep
from typing import Any, Dict, List, Optional

from clearml import Task, TaskTypes
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


@dataclass
class InteractiveSession:
    task_id: str
    username: str
    queue: str
    is_idle: bool
    running_time_days: float
    avg_gpu_usage: float


class WorkersMonitor:
    '''
    Monitors the available ClearML agents (workers) and reports their availability,
    interactive sessions, and general usage by user.
    '''

    def __init__(
            self,
            config: Dict[str, Any],
            slack_client: Optional[WebClient] = None,
            slack_channel: Optional[str] = None,
            low_gpu_threshold: float = 20.0,
            slack_notify_hour: int = 12,
            slack_notify_weekday: Optional[int] = 0,
            min_session_days: float = 1.0,
    ) -> None:
        '''
        Initialize the WorkersMonitor with a configuration dictionary.

        :param config: A dictionary that maps worker IDs to their corresponding queue configuration.
        :param slack_client: Optional Slack WebClient for sending low-utilization alerts.
        :param slack_channel: Slack channel ID or name to post alerts to.
        :param low_gpu_threshold: Avg GPU usage % below which a session is considered underutilized.
        :param slack_notify_hour: UTC hour (0-23) at which to send the daily Slack alert.
        :param slack_notify_weekday: If set (0=Mon…6=Sun), only notify on that weekday; otherwise Mon-Fri.
        :param min_session_days: Minimum session age in days before it is eligible for alerting.
        '''
        self._config = config
        self._slack_client = slack_client
        self._slack_channel = slack_channel
        self._low_gpu_threshold = low_gpu_threshold
        self._slack_notify_hour = slack_notify_hour
        self._slack_notify_weekday = slack_notify_weekday
        self._min_session_days = min_session_days
        self._slack_last_notified_date: Optional[date] = None
        self._slack_user_cache: Dict[str, str] = {}
        if self._slack_client:
            self._init_slack_user_cache()

    def monitor(self, pool_period: float = 60.0) -> None:
        '''
        Continuously runs the monitoring step at the specified interval.

        :param pool_period: Time in seconds between each monitoring step.
        '''
        while True:
            try:
                self._monitor_step()
            except Exception as ex:
                print(f'Exception: {ex}')
            sleep(pool_period)

    def _monitor_step(self) -> None:
        '''
        Performs a single step of monitoring by:
          1) Retrieving all workers.
          2) Calculating the global queue availability.
          3) Reporting the availability to ClearML.
          4) Identifying and reporting interactive sessions.
          5) Aggregating and reporting overall usage per user.
        '''
        try:
            workers: List[Dict[str, Any]] = self._get_workers()

            global_queues = self._calculate_queue_availability(workers)
            self._report_queue_availability(global_queues)

            interactive_sessions = self._identify_interactive_sessions(workers)
            self._report_interactive_sessions(interactive_sessions)
            if self._slack_client and self._slack_channel and self._should_notify_slack_now():
                self._notify_slack_low_utilization(interactive_sessions)

            usage_by_user = self._collect_usage_by_user(workers)
            self._report_usage_by_user(usage_by_user)

        except Exception as ex:
            print(f'Exception querying workers: {ex}')
            return

    def _calculate_queue_availability(
            self, workers: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, int]]:
        '''
        Calculate the number of available and total workers for each queue based on GPU usage.

        :param workers: A list of dictionaries containing worker info from ClearML.
        :return: A dictionary structured like:
            {
                queue_name: {
                    'num_available': <int>,
                    'num_total': <int>
                },
                ...
            }
        '''
        global_queues: Dict[str, Dict[str, int]] = {}

        for worker in workers:
            if not worker.get('queues'):
                continue
            worker_id: str = worker['id']
            if worker_id not in self._config:
                continue

            worker_prefix, worker_suffix = self._parse_worker_id(worker_id)
            if not worker_prefix or not worker_suffix:
                # skip if parsing failed
                continue

            gpus = self._parse_gpus(worker_suffix)
            total_gpus: int = len(gpus)

            worker_config = self._config[worker_id]['queues']
            total_gpus_per_queue = dict()
            for queue_name, num_gpus in worker_config.items():
                total_gpus_per_queue[queue_name] = total_gpus // num_gpus

            available_gpus_per_queue = self._deduct_allocated_gpus_per_queue(
                workers, worker_prefix, worker_config, total_gpus_per_queue
            )

            # Update global queue stats
            for queue_name, num_gpus in worker_config.items():
                if queue_name not in global_queues:
                    global_queues[queue_name] = {
                        'num_available': 0,
                        'num_total': 0,
                    }
                global_queues[queue_name]['num_available'] += available_gpus_per_queue.get(queue_name, 0)
                global_queues[queue_name]['num_total'] += total_gpus_per_queue.get(queue_name, 0)

        return global_queues

    def _report_queue_availability(self, global_queues: Dict[str, Dict[str, int]]) -> None:
        '''
        Logs availability to console and ClearML, showing how many workers in each queue are available.

        :param global_queues: A dictionary of queue statistics from calculate_queue_availability().
        '''
        queues_availability_table: List[List[Any]] = [
            ['Queue', 'Available workers', 'Num workers', '% Available']
        ]
        for queue_name, stats in global_queues.items():
            num_available: int = stats['num_available']
            num_total: int = stats['num_total']
            percent_available: float = (
                100.0 * num_available / num_total if num_total else 0.0
            )
            str_percent_available: str = f'{percent_available:.2f}'
            print(queue_name, ':', num_available, ':', num_total, ':', str_percent_available)
            queues_availability_table.append(
                [queue_name, num_available, num_total, str_percent_available]
            )

        print('')
        Task.current_task().get_logger().report_table(
            title='available_workers',
            series='queue',
            iteration=0,
            table_plot=queues_availability_table,
        )

    def _identify_interactive_sessions(self, workers: List[Dict[str, Any]]) -> List[InteractiveSession]:
        '''
        Build a list of interactive sessions by scanning for tasks named 'Interactive Session'.

        :param workers: Worker list from ClearML.
        :return: A list of InteractiveSession objects.
        '''
        sessions: List[InteractiveSession] = []

        for worker in workers:
            if 'task' not in worker or 'user' not in worker:
                continue

            task_name: str = worker['task']['name']
            if task_name != 'Interactive Session':
                continue

            task_id: str = worker['task']['id']
            running_time_days: float = worker['task']['running_time'] / (1000 * 60 * 60 * 24)
            sessions.append(InteractiveSession(
                task_id=task_id,
                username=worker['user']['name'],
                queue=self._get_queue_by_task_id(task_id),
                is_idle=worker['is_idle'],
                running_time_days=running_time_days,
                avg_gpu_usage=self._get_avg_gpu_usage_7d_by_worker_id(worker['id']),
            ))

        return sessions

    def _report_interactive_sessions(self, sessions: List[InteractiveSession]) -> None:
        '''
        Logs the interactive sessions table to ClearML.

        :param sessions: List of InteractiveSession objects from _identify_interactive_sessions.
        '''
        table = [
            ['Task ID', 'Username', 'Queue', 'Idle (GPU<80 & CPU<30)', 'Running Time (Days)', 'Avg GPU usage 7 days']]
        for s in sessions:
            table.append(
                [s.task_id, s.username, s.queue, s.is_idle, f'{s.running_time_days:.2f}', f'{s.avg_gpu_usage:.2f}'])
        Task.current_task().get_logger().report_table(
            title='interactive_sessions',
            series='sessions',
            iteration=0,
            table_plot=table,
        )

    def _should_notify_slack_now(self) -> bool:
        '''
        Returns True if a Slack notification should be sent right now, False otherwise.

        Conditions (all must hold):
          - Current UTC weekday is Monday–Friday (weekday < 5).
          - If slack_notify_weekday is set, the current weekday must match it exactly.
          - Current UTC hour has reached slack_notify_hour.
          - A notification has not already been sent today (tracked via _slack_last_notified_date).

        When all conditions are met, _slack_last_notified_date is updated to today so subsequent
        calls within the same UTC calendar day return False.
        '''
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return False
        if self._slack_notify_weekday is not None and now.weekday() != self._slack_notify_weekday:
            return False
        if now.hour < self._slack_notify_hour:
            return False
        today = now.date()
        if self._slack_last_notified_date == today:
            return False
        self._slack_last_notified_date = today
        return True

    def _init_slack_user_cache(self) -> None:
        try:
            cursor = None
            while True:
                response = self._slack_client.users_list(limit=200, cursor=cursor)
                for member in response['members']:
                    if member.get('deleted') or member.get('is_bot'):
                        continue
                    user_id: str = member['id']
                    real_name: str = member.get('real_name', '')
                    display_name: str = member.get('profile', {}).get('display_name', '')
                    if real_name:
                        self._slack_user_cache[real_name.lower()] = user_id
                    if display_name:
                        self._slack_user_cache[display_name.lower()] = user_id
                cursor = response.get('response_metadata', {}).get('next_cursor')
                if not cursor:
                    break
        except SlackApiError as e:
            print(f'Failed to fetch Slack users: {e}')

    def _get_slack_mention(self, username: str) -> str:
        lower = username.lower()
        user_id = (
            self._slack_user_cache.get(lower)  # "Name Surname" format
            or self._slack_user_cache.get(lower.replace(' ', '.'))  # "name.surname" format
        )
        return f'<@{user_id}>' if user_id else username

    def _notify_slack_low_utilization(self, sessions: List[InteractiveSession]) -> None:
        low_sessions = [
            s for s in sessions
            if s.running_time_days >= self._min_session_days and 0 <= s.avg_gpu_usage < self._low_gpu_threshold
        ]
        if not low_sessions:
            return

        # Sort sessions by avg GPU usage ascending, then by running time descending
        low_sessions.sort(key=lambda s: (s.avg_gpu_usage, -s.running_time_days))

        lines = [
            f'Sessions below {self._low_gpu_threshold:.0f}% avg GPU (7-day avg):'
        ]
        for s in low_sessions:
            mention = self._get_slack_mention(s.username)
            lines.append(
                f'• {mention} — `{s.task_id[:8]}...` on `{s.queue}` — *{s.avg_gpu_usage:.1f}%* GPU — {s.running_time_days:.0f}d running'
            )
        lines.append(
            '\n If you\'re not actively using your session, please consider terminating it to free up GPU resources for others. Thank you! 🙏'
        )

        try:
            print('Posting Slack alert:\n' + '\n'.join(lines))
            self._slack_client.chat_postMessage(channel=self._slack_channel, text='\n'.join(lines))
        except SlackApiError as e:
            print(f'Failed to send Slack message: {e}')

    def _collect_usage_by_user(
            self, workers: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        '''
        Aggregates the usage info by user across all workers.

        :param workers: A list of worker information from ClearML.
        :return: Dictionary mapping each username to usage info, e.g.:
            {
                username: {
                    'num_tasks': <int>,
                    'queues': <set of queue names>
                },
                ...
            }
        '''
        usage_by_user: Dict[str, Dict[str, Any]] = {}

        for worker in workers:
            if 'task' not in worker or 'user' not in worker:
                continue

            username: str = worker['user']['name']
            task_id: str = worker['task']['id']
            queue_name: str = self._get_queue_by_task_id(task_id)

            if username not in usage_by_user:
                usage_by_user[username] = {
                    'num_tasks': 0,
                    'queues': set(),
                }

            usage_by_user[username]['num_tasks'] += 1
            usage_by_user[username]['queues'].add(queue_name)

        return usage_by_user

    def _report_usage_by_user(self, usage_by_user: Dict[str, Dict[str, Any]]) -> None:
        '''
        Logs users' usage info in a ClearML table.

        :param usage_by_user: The usage dictionary from _collect_usage_by_user().
        '''
        print(usage_by_user)
        usage_by_user_table: List[List[Any]] = [['Username', 'Num Tasks', 'Queues']]
        for username, stats in usage_by_user.items():
            num_tasks: int = stats['num_tasks']
            queues_str: str = ','.join(list(stats['queues']))
            usage_by_user_table.append([username, num_tasks, queues_str])

        Task.current_task().get_logger().report_table(
            title='usage_by_user',
            series='usage',
            iteration=0,
            table_plot=usage_by_user_table,
        )

    def _get_workers(self) -> List[Dict[str, Any]]:
        '''
        Retrieves all worker information from ClearML.

        :return: A list of dictionaries, each containing worker info.
        '''
        session = Task.current_task().session
        response = session.send_request(service='workers', action='get_all', method='post')
        return response.json()['data']['workers']

    def _get_queue_by_task_id(self, task_id: str) -> str:
        '''
        Retrieves the queue name associated with a given Task ID.

        :param task_id: The ClearML Task ID.
        :return: Name of the queue to which the Task is assigned.
        '''
        session = Task.current_task().session
        response = session.send_request(
            service='tasks',
            action='get_by_id_ex',
            method='post',
            json={'id': task_id, 'only_fields': ['id', 'execution.queue.name']},
        )
        return response.json()['data']['tasks'][0]['execution']['queue']['name']

    def _get_avg_gpu_usage_7d_by_worker_id(self, worker_id: str) -> float:
        '''
        Calculates a worker's average GPU usage over the last 7 days.

        :param worker_id: The ID of the ClearML worker.
        :return: The average GPU usage percentage. Returns -1 if no data is available.
        '''
        session = Task.current_task().session
        interval: int = 60 * 60 * 24 * 7  # 7 days in seconds
        to_date: int = int(time())
        from_date: int = to_date - interval + 1
        response = session.send_request(
            service='workers',
            action='get_stats',
            method='post',
            json={
                'worker_ids': [worker_id],
                'items': [{'category': 'avg', 'key': 'gpu_usage'}],
                'from_date': from_date,
                'to_date': to_date,
                'interval': interval,
            },
        )
        workers_data = response.json()['data']['workers']
        if not workers_data:
            return -1.0
        values: List[float] = workers_data[0]['metrics'][0]['stats'][0]['values']
        avg_gpu_usage: float = sum(values) / len(values) if values else -1.0
        return avg_gpu_usage

    def _parse_worker_id(self, worker_id: str) -> (str, str):
        '''
        Splits a worker ID into prefix and suffix if possible.

        :param worker_id: e.g. 'clearml-agent-lambda-server-6-8xA100:dgpu0,1,2,3'
        :return: (prefix, suffix) or ('', '') if invalid
        '''
        worker_id_split = worker_id.split(':')
        return (
            (worker_id_split[0], worker_id_split[1])
            if len(worker_id_split) == 2
            else ('', '')
        )

    def _parse_gpus(self, worker_suffix: str) -> List[str]:
        '''
        Converts a GPU suffix string into a list of GPU IDs.

        :param worker_suffix: e.g. 'dgpu0,1,2,3' or 'dgpu0-3'
        :return: A list of GPU identifiers, e.g. ['0', '1', '2', '3'].
        '''
        gpus_str: str = worker_suffix.replace('dgpu', '')
        if '-' in gpus_str:
            start_str, end_str = gpus_str.split('-')
            gpus_range = range(int(start_str), int(end_str) + 1)
            return [str(gpu_id) for gpu_id in gpus_range]
        return gpus_str.split(',')

    def _deduct_allocated_gpus_per_queue(
            self,
            workers: List[Dict[str, Any]],
            worker_prefix: str,
            worker_config: Dict[str, int],
            total_gpus_per_queue: Dict[str, int]
    ) -> Dict[str, float]:
        '''
        Calculate available GPUs per queue after deducting allocated GPUs.
        If a GPU is occupied by any queue (fractional or whole), it affects availability for all queues.

        :param workers: The list of all workers from ClearML.
        :param worker_prefix: The prefix portion of this worker's ID.
        :param worker_config: Queue config for the current worker (like {'onprem.1xA100': 1, ...}).
        :param total_gpus_per_queue: Total workers available per queue (like {'onprem.1xA100': 8, ...}).
        :return: Dictionary mapping queue names to their available GPU counts.
        '''
        available_gpus_per_queue = total_gpus_per_queue.copy()

        for q_worker in workers:
            if 'id' not in q_worker or not q_worker['id'].startswith(worker_prefix) or 'dgpu' in q_worker['id']:
                continue
            worker_gpu_suffix = q_worker['id'].split(':gpu')[1]
            if len(worker_gpu_suffix.split('.')) > 1:
                queue_val = float('.' + worker_gpu_suffix.split('.')[1][:-1])
            else:
                queue_val = len(worker_gpu_suffix.split(','))
            for queue_name, num_gpus in worker_config.items():
                if queue_val > num_gpus:
                    available_gpus_per_queue[queue_name] -= queue_val // num_gpus
                else:
                    available_gpus_per_queue[queue_name] -= 1
                available_gpus_per_queue[queue_name] = max(0, available_gpus_per_queue[queue_name])
        return available_gpus_per_queue


def main() -> None:
    '''
    Main entry point for the script.
    '''

    # fmt: off
    parser = argparse.ArgumentParser(description='ClearML Monitoring Available Workers')
    parser.add_argument('--period', type=int, help='Poll period in seconds', default=60)
    parser.add_argument('--config', type=str, help='Config', required=True)
    parser.add_argument('--slack-token', type=str, help='Slack bot token', default=None)
    parser.add_argument('--slack-channel', type=str, help='Slack channel to post alerts to', default=None)
    parser.add_argument('--slack-notify-hour', type=int, help='UTC hour (0-23) at which to send the Slack alert (default: 12)', default=12)
    parser.add_argument('--slack-notify-weekday', type=int, help='Weekday to notify on (0=Mon…6=Sun); default: 0 (Monday)', default=0)
    parser.add_argument('--gpu-threshold', type=float, help='Avg GPU usage %% below which a session is alerted', default=10.0)
    parser.add_argument('--min-session-days', type=float, default=2.0, help='Minimum session age in days before it is eligible for Slack alerting (default: 2.0)')
    args = parser.parse_args()
    # fmt: on

    cfg = json.loads(json.loads(args.config))

    task = Task.init(project_name='DevOps', task_name='Workers monitor (TMP VUSAL)', task_type=TaskTypes.monitor)
    task.connect(cfg)

    slack_client = WebClient(token=args.slack_token) if args.slack_token else None

    workers_monitor = WorkersMonitor(
        cfg,
        slack_client=slack_client,
        slack_channel=args.slack_channel,
        low_gpu_threshold=args.gpu_threshold,
        slack_notify_hour=args.slack_notify_hour,
        slack_notify_weekday=args.slack_notify_weekday,
        min_session_days=args.min_session_days,
    )
    workers_monitor.monitor(pool_period=float(args.period))


if __name__ == '__main__':
    main()
