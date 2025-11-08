import numpy as np


class Sampler:
    def __init__(self, batch_size, events):
        self.batch_size = batch_size
        self.events = events

    def step(self, edge_index):
        def sample(idx):
            ui = edge_index[idx]
            i = ui[1]

            return ui[0], i, ui[2]

        for batch_start in range(0, self.events, self.batch_size):
            user, item, r = map(np.array, zip(*[sample(i) for i in range(batch_start, min(batch_start + self.batch_size,
                                                                                          self.events))]))
            yield user, item, r.astype('int64')
