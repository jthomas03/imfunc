# -*- coding: utf-8 -*-
#
# John C. Thomas 2023
from .edge import edge
from .node import node

class emap(object):
    def __init__(self):
        self.edges = {}
        self.header = ['#include "colors.inc"',
                       '#include "textures.inc"',
                       'camera{orthographic location<0,-400,0> look_at<0,0,0>}',
                       'light_source{<0,-400,0> rgb<1,1,1>}',
                       'global_settings{ambient_light rgb 1}',
                       'background{rgb<1,1,1>}']
        self.subheader = ['#declare bondcolor = texture{pigment {rgb<0.5,0.5,0.5>} finish { phong 0.7 ambient 0.1 diffuse 0.8 } }',
                          '#declare bondratio = 1;']
        self.subheader2 = ['#declare yoursurface = union{']
        self.footer = ['rotate<0,0,0>',
                       '}',
                       '\n',
                       'object{yoursurface rotate<0,0,0> translate<0,0,0> no_shadow}']
        
    def addNode(self,node):
        if node in self.edges:
            raise ValueError('Duplicate node')
        else:
            self.edges[node]=[]

    def addEdge(self,edge):
        src = edge.getSource()
        dest = edge.getDestination()
        if not (src in self.edges and dest in self.edges):
            raise ValueError('Node not in graph')
        self.edges[src].append(dest)

    def getChildrenof(self,node):
        return self.edges[node]

    def hasNode(self,node):
        return node in self.edges
    
    def display(self):
        for i in self.edges:
            print(i)
    
    def getlen(self):
        return len(self.edges)

    def getdist(self):
        k = 0
        dists = []
        for i in self.edges:
            for j in self.edges[i]:
                pos1 = i.getPosition()
                pos2 = j.getPosition()
                dist = np.sqrt((pos2[1]-pos1[1])**2 + (pos2[0]-pos1[0])**2)
                dists.append(dist)
        return dists

    def getpov(self,fil):
        atype = []
        atoms = []
        bonds = []
        visited = []
        for i in self.edges:
            pos = i.getPosition()
            itype = i.gettype()
            if not itype in atype:
                atype.append(itype)
            atoms.append("sphere {<"+str(pos[0])+","+str(pos[1])+","+str(pos[2])+">, 1*"+str(itype)+"ratio texture{"+str(itype)+"} no_shadow}")
            for j in self.edges[i]:
                pos2 = j.getPosition()
                vis = "cylinder{<"+str(pos[0])+","+str(pos[1])+","+str(pos[2])+">,<"+str(pos2[0])+","+str(pos2[1])+","+str(pos2[2])+">, .1*bondratio texture{bondcolor} no_shadow}"
                visr = "cylinder{<"+str(pos2[0])+","+str(pos2[1])+","+str(pos2[2])+">,<"+str(pos[0])+","+str(pos[1])+","+str(pos[2])+">, .1*bondratio texture{bondcolor} no_shadow}"
                if (vis not in visited) and (visr not in visited):
                    bonds.append(vis)
                    visited.append(vis)
                    visited.append(visr)
        otype = []
        for atomtype in atype:
            otype.append('#declare '+str(atomtype)+' = texture{pigment {rgb<1,1,1>} finish { phong 0.7 ambient 0.1 diffuse 0.8 } }')
            otype.append('#declare '+str(atomtype)+'ratio = 1;')
        outfile = self.header+self.subheader+otype+self.subheader2+atoms+bonds+self.footer
        with open(fil, "w") as text_file:
            for i in outfile:
                text_file.write(i + '\n')
                
    def getpdb(self,fil,l1,l2):
        HETATM = []
        CONECT = []
        visited = []
        header = []
        cnt = 0
        for i in self.edges:
            cnt += 1
            pos = i.getPosition()
            itype = i.gettype()
            ival = i.getvalue()
            inum = str(itype)+str(ival)
            HETATM.append("{:6s}{:5d} {:^4s}              {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}".format("HETATM",int(ival),str(inum[:4]),float(pos[0]),float(pos[1]),float(pos[2]),1.00,1.00,itype))
            conects = 'CONECT'+"  "+str(ival)+"  "
            for j in self.edges[i]:
                bval = j.getvalue()
                conects += str(bval)+"  "
            CONECT.append(conects)
        outfile = header+HETATM+CONECT
        with open(fil, "w") as text_file:
            text_file.write(l1)
            text_file.write(l2)
            for i in outfile:
                text_file.write(i + '\n')
            text_file.write('END')