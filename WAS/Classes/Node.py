import datetime


class Node:
    """ Class node represents an instance of a task in a workflow """

    def __init__(self, node_id, node_name, successors=None, graph_depth=None, state=None,
                 execution_duration=0, execution_datetime=None, warnings=None,
                 errors=None, predecessors=None, parent_workflow=None):
        if successors is None:
            successors = []
        if predecessors is None:
            predecessors = []
        self.id = node_id
        self.parent_workflow = parent_workflow
        self.graph_depth = graph_depth
        self.name = node_name
        self.state = state
        self.successors = successors
        self.execution_duration = execution_duration
        self.execution_datetime = execution_datetime
        self.warnings = warnings
        self.errors = errors
        self.predecessors = predecessors

    def execution_datetime_to_date_format(self):
        if self.execution_datetime is not None:
            try:
                date_time_obj = datetime.datetime.strptime(self.execution_datetime, '%Y-%m-%d %H:%M:%S +%f')
                date = date_time_obj.date()
                weekday = date_time_obj.weekday()
                hour = date_time_obj.time().hour
                return date.day, date.month, date.year, weekday, hour
            except (ValueError, TypeError):
                print("Could not convert lastExecutionStartTime string to datetime format. Invalid datetime.")
        return None

    def is_failed(self):
        # to have
        is_failed = not self.state == "EXECUTED" and bool(self.errors)
        return int(is_failed)

    def to_ml_ready_dict(self):
        day, month, year, weekday, time = self.execution_datetime_to_date_format()
        ml_ready_dict = {
            # 'parent_workflow': self.parent_workflow,
            # 'node_id': self.id,
            'node_name': self.name,
            'is_failed': self.is_failed(),
            'execution_duration': self.execution_duration,
            'warnings': int(self.warnings),
            'execution_day': day,
            'execution_month': month,
            'execution_year': year,
            'execution_weekday': weekday,
            'execution_time': time,
        }
        # create a custom column for each predecessor
        for predecessor in self.predecessors:
            key = "has_prdc {}".format(predecessor.name)
            ml_ready_dict[key] = 1
        return ml_ready_dict
