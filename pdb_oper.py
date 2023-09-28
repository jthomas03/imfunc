# -*- script: utf-8 -*-
#
# John C. Thomas 2023

from imMF.definitions import get_pdb, get_pov

dpath = '/Users/jthomas/Repo/ML/xyz/'
spath = '/Users/jthomas/GitRepo/imMF/imfunc_manip/'

fil = "WS2_jz.pdb"
#filout = "WS2_testing.pdb"
filout = "WS2_testing.pov"

#get_pdb(fil,filout,dpath,spath)
get_pov(fil,filout,dpath,spath)