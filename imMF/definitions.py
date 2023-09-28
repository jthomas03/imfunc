# -*- script: utf-8 -*-
#
# John C. Thomas 2023

from .edge import edge
from .node import node
from .graph import emap

#dpath = '/Users/jthomas/Repo/ML/xyz/'
#fil = dpath+"graphene_sheet_59x29_ang_single.pdb"
#spath = '/Users/jthomas/GitRepo/imMF/imfunc_manip/'

#fil = dpath+"WS2_jz.pdb"

def get_pdb(fil,filout,dpath,spath):
    outfile = []
    cnt = 0
    l1 = ''
    l2 = ''
    with open(dpath+fil,'r') as f:
        cnt += len(f.readlines())
    dlist = emap()
    nlist = dict()
    with open(dpath+fil,'r') as f:
        l1 = f.readline()
        l2 = f.readline()
        l2 = 'COMPND   1Created by jthomas'+'\n'
        for i in range(0,(cnt-3)):
            line = f.readline()
            lsplit = line.split(' ')
            sline = [s for s in lsplit if s]
            stripline = [s.replace('\n','') for s in sline]
            if stripline[0] == 'HETATM':
                exx, eyy, ezz = stripline[3], stripline[4], stripline[5]
                mol = node([exx,eyy,ezz],stripline[1],stripline[8])
                dlist.addNode(mol)
                nlist[mol.getvalue()] = mol
            elif stripline[0] == 'CONECT':
                stripline_out = []
                for j in range(2,len(stripline)):
                    if stripline[j] != '':
                        stripline_out.append(stripline[j])
                for k in range(0,len(stripline_out)):
                    src = nlist[str(stripline[1])]
                    des = nlist[str(stripline_out[k])]
                    iedge = edge(src,des)
                    dlist.addEdge(iedge)
    dlist.getpdb(spath+filout,l1,l2)

def get_pov(fil,filout,dpath,spath):
    outfile = []
    cnt = 0
    with open(dpath+fil,'r') as f:
        cnt += len(f.readlines())
    dlist = emap()
    nlist = dict()
    with open(dpath+fil,'r') as f:
        f.readline()
        f.readline()
        for i in range(0,(cnt-3)):
            line = f.readline()
            lsplit = line.split(' ')
            sline = [s for s in lsplit if s]
            stripline = [s.replace('\n','') for s in sline]
            if stripline[0] == 'HETATM':
                mol = node([stripline[3],stripline[4],stripline[5]],stripline[1],stripline[8])
                dlist.addNode(mol)
                nlist[mol.getvalue()] = mol
            elif stripline[0] == 'CONECT':
                stripline_out = []
                for j in range(2,len(stripline)):
                    if stripline[j] != '':
                        stripline_out.append(stripline[j])
                for k in range(0,len(stripline_out)):
                    src = nlist[str(stripline[1])]
                    des = nlist[str(stripline_out[k])]
                    iedge = edge(src,des)
                    dlist.addEdge(iedge)

    dlist.getpov(spath+filout)