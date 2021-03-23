class Workflow:
    """ Class workflow represents an instance of a workflow or sub-workflow """

    def __init__(self, workflow_name=None, has_failed=False, user="UnKnown",
                 total_duration=0, children=None, number_of_tasks=0):
        if children is None:
            children = []
        self.name = workflow_name
        self.is_failed = int(has_failed)
        self.user = user
        self.total_duration = total_duration
        self.children = children
        self.number_of_tasks = number_of_tasks

    def sum_subworkflows_values(self):
        for child in self.children:
            self.total_duration += child.total_duration
            self.number_of_tasks += child.number_of_tasks

    def to_ml_ready_dict(self):
        self.sum_subworkflows_values()
        ml_ready_dict = {
            'workflow_name': self.name,
            'has_failed': self.is_failed,
            'user': self.user,
            'number_of_tasks': self.number_of_tasks,
            'total_duration': self.total_duration
        }
        return ml_ready_dict
