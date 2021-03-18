import datetime

class Node:
    """ Class node represents an instance of a class in a workflow"""

    def __init__(self, node_id, parent_workflow, graph_depth,
                 class_name, state, successors, execution_duration,
                 execution_datetime, warnings, errors, predecessors=None):
        if predecessors is None:
            predecessors = []
        self.id = node_id
        self.parent_workflow = parent_workflow
        self.graph_depth = graph_depth
        self.class_name = class_name
        self.state = state
        self.successors = successors
        self.execution_duration = execution_duration
        self.execution_datetime = execution_datetime
        self.warnings = warnings
        self.errors = errors
        self.predecessors = predecessors

    def execution_datetime_to_date_format(self):
        try:
            date_time_obj = datetime.datetime.strptime(self.execution_datetime, '%Y-%m-%d %H:%M:%S +%f')
            date = date_time_obj.date()
            weekday = date_time_obj.weekday()
            return date.day, date.month, date.year, weekday, date_time_obj.time().hour
        except ValueError:
            print("Could not convert lastExecutionStartTime string to datetime format. Invalid datetime.")

    def isExecuted(self):
        if self.state == "EXECUTED":
            return 1
        else:
            return 0

    def to_ml_ready_dict(self):
        day, month, year, weekday, time = self.execution_datetime_to_date_format()
        return {
            'parent_workflow': self.parent_workflow,
            'class_name': self.class_name,
            'isExecuted': self.isExecuted(),
            'execution_duration': self.execution_duration,
            'warnings': int(self.warnings),
            'errors': int(self.errors),
            'execution_day': day,
            'execution_month': month,
            'execution_year' : year,
            'execution_weekday': weekday,
            'execution_time': time,
            'predecessors_class_name:': [predecessor.class_name for predecessor in iter(self.predecessors)],
            # 'predecessors_warnings': [predecessor.warnings for predecessor in iter(self.predecessors)],
            # 'predecessors_errors': [predecessor.errors for predecessor in iter(self.predecessors)]
        }
