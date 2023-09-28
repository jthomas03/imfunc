# -*- script: utf-8 -*-
#
# John C. Thomas 2023

from imMF.definitions import get_pdb, get_pov

dpath = 'path1'
spath = 'path2'

fil = "your.pdb"
#filout = "WS2_testing.pdb"
filout = "youfile_out.pov"

#get_pdb(fil,filout,dpath,spath)
get_pov(fil,filout,dpath,spath)