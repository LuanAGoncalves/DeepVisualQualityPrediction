import numpy as np
from visdom import Visdom


class Plot(object):
    def __init__(self, title, port=8097):
        self.viz = Visdom(port=port)
        self.windows = {}
        self.title = title

    def register_line(self, name, xlabel, ylabel):
        self.opts = dict(title=self.title, markersize=5, xlabel=xlabel, ylabel=ylabel)
        win = self.viz.line(X=np.zeros((1, 2)), Y=np.zeros((1, 2)), opts=self.opts)
        self.windows[name] = win

    def update_line(self, name, x, y):
        self.viz.line(X=x, Y=y, win=self.windows[name], opts=self.opts)
