#!/bin/bash

# First, do single-machine test
nosetests *.py

# Second, test when there is no mpi
PYTHONPATH_SAV=$PYTHONPATH
PYTHONPATH=$PWD/nompi:$PYTHONPATH
nosetests *.py
PYTHONPATH=$PYTHONPATH_SAV

# third, test when mpi exists.
for i in {1..5}
do
    mpirun -n $i nosetests *.py
done


