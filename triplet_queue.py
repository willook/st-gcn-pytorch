import random
from typing import Dict, List

import numpy as np

class TripletQueue():
    def __init__(self, num_classes = 61):
        self.queue: Dict[List] = {}
        self.values = [[] for _ in range(num_classes)]
        self.size = [0 for _ in range(num_classes)]
        self._next = None

    def put(self, key, value):
        self.values[key].append(value)
        self.size[key] += 1

    def hasnext(self):
        i0 = np.argmax(self.size)
        temp = self.size[i0]
        self.size[i0] = 0
        i1 = np.argmax(self.size)
        self.size[i0] = temp
        self._next = (i0, i1)
        return self.size[i0] > 1 and self.size[i1] > 0
    
    def next(self):
        assert self._next is not None
        i0, i1 = self._next
        self._next = None
        self.size[i0] -= 2
        self.size[i1] -= 1        
        return self.values[i0].pop(), self.values[i0].pop(), self.values[i1].pop()