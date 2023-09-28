# -*- coding: utf-8 -*-
#
# John C. Thomas 2023

class node(object):
    def __init__(self,position,value,ntype):
        self.value=value
        self.position=position
        self.ntype = ntype

    def getPosition(self):
        return self.position

    def getvalue(self):
        return self.value
    
    def gettype(self):
        return self.ntype

    def getNodeHash(self):
        return hash(str(self.position)+str(self.value))

    def __str__(self):
        return 'P:'+str(self.position)+' V:'+str(self.value)+' T:'+str(self.ntype)
    
    def updateposition(self,position):
        self.position=position