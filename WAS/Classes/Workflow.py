class Workflow:
    """ Class workflow represents an instance of a workflow or sub-workflow """

    def __init__(self, workflow_name=None, user="UnKnown", children_workflows=None, nodes=None):
        if nodes is None:
            nodes = []
        if children_workflows is None:
            children_workflows = []
        self.name = workflow_name
        self.user = user
        self.children_workflows = children_workflows
        self.nodes = nodes

    @property
    def has_failed(self):
        for node in self.nodes:
            if node.has_failed is True:
                return True
        for child in self.children_workflows:
            if child.has_failed is True:
                return True
        return False

    @property
    def number_of_nodes(self):
        number_of_nodes = len(self.nodes)
        for child in self.children_workflows:
            number_of_nodes += child.number_of_nodes
        return number_of_nodes

    @property
    def total_duration(self):
        total_duration = 0
        for node in self.nodes:
            total_duration += node.execution_duration
        for child in self.children_workflows:
            total_duration += child.total_duration
        return total_duration

    @property
    def makespan(self):
            return self.total_duration / self.number_of_nodes

    def to_ml_ready_dict(self):
        ml_ready_dict = {
            'name': self.name,
            'user': self.user,
            # 'number of nodes': self.number_of_nodes,
            # 'total duration': self.total_duration,
            'failure': int(self.has_failed),
            'makespan': self.makespan
        }
        return ml_ready_dict

    def to_dict(self):
        return {
            'name': self.name,
            'user': self.user,
            'number of nodes': self.number_of_nodes,
            'total duration': self.total_duration,
            'children workflows': [child.name for child in self.children_workflows],
            'failure': int(self.has_failed),
            'makespan': self.makespan
        }
