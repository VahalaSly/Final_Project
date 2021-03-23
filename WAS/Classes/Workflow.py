class Workflow:
    """ Class workflow represents an instance of a workflow or sub-workflow """

    def __init__(self, workflow_name=None, has_failed=False, user="UnKnown",
                 total_duration=0, children=None):
        if children is None:
            children = []
        self.name = workflow_name
        self.is_failed = int(has_failed)
        self.user = user
        self.total_duration = total_duration
        self.children = children

    def get_total_duration(self):
        for child in self.children:
            self.total_duration += child.total_duration
        return self.total_duration

    def to_ml_ready_dict(self):
        ml_ready_dict = {
            'workflow_name': self.name,
            'has_failed': self.is_failed,
            'user': self.user,
            'total_duration': self.get_total_duration()
        }
        return ml_ready_dict
