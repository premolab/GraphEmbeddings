class RunConfiguration:

    def __init__(self, method: str, graph_name, dimension):
        self.method = method
        self.graph_name = graph_name
        self.dimension = dimension

    def __str__(self):
        return '{}_{}_{}'.format(self.method, self.graph_name, self.dimension)

    @staticmethod
    def from_string(s):
        last = s.rfind('_')
        pre_last = s.rfind('_', 0, last)
        return RunConfiguration(s[:pre_last], s[pre_last+1:last], s[last+1:])
