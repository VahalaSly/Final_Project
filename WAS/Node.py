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
            hour = date_time_obj.time().hour
            return date.day, date.month, date.year, weekday, hour
        except ValueError:
            print("Could not convert lastExecutionStartTime string to datetime format. Invalid datetime.")

    def is_executed(self):
        if self.state == "EXECUTED":
            return 1
        else:
            return 0

    def infer_predecessor(self, workflow_nodes):
        for successor in self.successors:
            workflow_nodes[successor].predecessors.append(self)

    def to_ml_ready_dict(self):
        day, month, year, weekday, time = self.execution_datetime_to_date_format()
        ml_ready_dict = {
            'parent_workflow': self.parent_workflow,
            'class_name': self.class_name,
            'is_executed': self.is_executed(),
            'execution_duration': self.execution_duration,
            'warnings': int(self.warnings),
            'errors': int(self.errors),
            'execution_day': day,
            'execution_month': month,
            'execution_year': year,
            'execution_weekday': weekday,
            'execution_time': time,
        }
        # create a custom column for each predecessor
        for predecessor in self.predecessors:
            key = "has_prdc_{}".format(predecessor.class_name)
            ml_ready_dict[key] = 1
        return ml_ready_dict
