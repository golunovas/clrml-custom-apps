import json
import argparse
import itertools
from time import time, sleep
from typing import Any, Dict, List
from clearml import Task, TaskTypes


class WorkersMonitor:
    '''
    Monitors the available ClearML agents (workers) and reports their availability,
    interactive sessions, and general usage by user.
    '''

    def __init__(self, config: Dict[str, Any]) -> None:
        '''
        Initialize the WorkersMonitor with a configuration dictionary.

        :param config: A dictionary that maps worker IDs to their corresponding queue configuration.
        '''
        self._config = config

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

            interactive_sessions_table = self._identify_interactive_sessions(workers)
            self._report_interactive_sessions(interactive_sessions_table)

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

    def _identify_interactive_sessions(self, workers: List[Dict[str, Any]]) -> List[List[Any]]:
        '''
        Build an interactive sessions information table by scanning for tasks named 'Interactive Session'.

        :param workers: Worker list from ClearML.
        :return: A table of interactive session info ready for reporting.
        '''
        interactive_sessions_table: List[List[Any]] = [
            ['Task ID', 'Username', 'Queue', 'Idle (GPU<80 & CPU<30)', 'Running Time (Days)', 'Avg GPU usage 7 days']
        ]

        for worker in workers:
            if 'task' not in worker or 'user' not in worker:
                continue

            username: str = worker['user']['name']
            task_id: str = worker['task']['id']
            task_name: str = worker['task']['name']
            running_time_msec: int = worker['task']['running_time']
            running_time_days: float = running_time_msec / (1000 * 60 * 60 * 24) # 24 hours in msec

            if task_name != 'Interactive Session':
                continue

            queue_name: str = self._get_queue_by_task_id(task_id)
            is_idle: bool = worker['is_idle']
            avg_gpu_usage: float = self._get_avg_gpu_usage_7d_by_worker_id(worker['id'])
            interactive_sessions_table.append(
                [
                    task_id,
                    username,
                    queue_name,
                    is_idle,
                    f'{running_time_days:.2f}',
                    avg_gpu_usage,
                ]
            )

        return interactive_sessions_table

    def _report_interactive_sessions(self, sessions_table: List[List[Any]]) -> None:
        '''
        Logs the interactive sessions table to ClearML.

        :param sessions_table: Data table about running interactive sessions from _identify_interactive_sessions.
        '''
        Task.current_task().get_logger().report_table(
            title='interactive_sessions',
            series='sessions',
            iteration=0,
            table_plot=sessions_table,
        )

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
    parser = argparse.ArgumentParser(description='ClearML Monitoring Available Workers')
    parser.add_argument('--period', type=int, help='Poll period in seconds', default=60)
    parser.add_argument('--config', type=str, help='Config', required=True)
    args = parser.parse_args()
    
    cfg = json.loads(args.config)

    task = Task.init(project_name='DevOps', task_name='Workers monitor', task_type=TaskTypes.monitor)
    task.connect(cfg)

    workers_monitor = WorkersMonitor(cfg)
    workers_monitor.monitor(pool_period=float(args.period))


if __name__ == '__main__':
    main()