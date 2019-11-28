class Landmark(object):

    def __init__(self, graph_id):
        """Creates a landmark

        Arguments:
            graph_id : The id the landmark has in the graph (must be unique)
        """
        self.lm_id = graph_id
        self.is_observable_in_tracking = False
        self.last_frm_identifier = -1  # the last frame where it was observed
        self.n_observable = 0  # the number of frames and KFs it is seen in
        self.descriptor = None

    def compute_descriptor(self, graph):
        """ Computes the descriptor from the observations
        - similar to OpenVSlam
        - or simply take the most recent one
        """
        if len(graph[self.lm_id]) == 0:
            self.descriptor = None
            return

        #Compute the descriptor
        for kf in graph[self.lm_id]:
            print("kf: ", kf)
    # def add_observation(self, graph, keyframe):
