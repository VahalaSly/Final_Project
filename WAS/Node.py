class Node:
    """ Class node represents an instance of a class in a workflow"""

    def __init__(self, node_id, parent_workflow, graph_depth,
                 class_name, state, successors, execution_duration,
                 warnings, errors, predecessors=None):
        if predecessors is None:
            predecessors = []
        self.id = node_id
        self.parent_workflow = parent_workflow
        self.graph_depth = graph_depth
        self.class_name = class_name
        self.state = state
        self.successors = successors
        self.execution_duration = execution_duration
        self.warnings = warnings
        self.errors = errors
        self.predecessors = predecessors

    def to_dict(self):
        return {
            'parent_workflow': self.parent_workflow,
            'class_name': self.class_name,
            'state': self.state,
            'execution_duration': self.execution_duration,
            'warnings': self.warnings,
            'errors': self.errors,
            'predecessors_class_name:': [predecessor.class_name for predecessor in iter(self.predecessors)],
            'predecessors_warnings': [predecessor.warnings for predecessor in iter(self.predecessors)],
            'predecessors_errors': [predecessor.errors for predecessor in iter(self.predecessors)]
        }


