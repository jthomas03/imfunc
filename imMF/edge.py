# -*- coding: utf-8 -*-
#
# John C. Thomas 2023

class edge(object):
    def __init__(self,src,dest):
        self.src = src
        self.dest = dest

    def getSource(self):
        return self.src

    def getDestination(self):
        return self.dest
    
    def getWeight(self):
        return self.dest.getvalue()

    def __str__(self):
        return (self.src.getPosition(),)+'->'+(self.dest.getPosition(),)