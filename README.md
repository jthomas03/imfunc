# imfunc
A Nanonis data analysis class.

## Requirements

Python 

After class instantiation, run any of the definitions available

Examples:
```
image = readfile(filein,'Z','forward')

spec = readspec(filein)
```

POV Ray output from pdb
```
from imMF.definitions get_pov

get_pov(fil,filout,dpath,spath)
```

pdb output from pdb (can be used to manipulate atomic coordinates, bonds, etc)
```
from imMF.definitions get_pdb

get_pdb(fil,filout,dpath,spath)
```

## Python Dependencies

The following dependencies are required, and should be available from PyPi.

* ```numpy```   — support for large, multi-dimensional arrays
* ```matplotlib``` — visualization tool
* ```scipy``` — scientific and technical computing library
* ```opencv``` — computer vision library
