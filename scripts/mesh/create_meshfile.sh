#!/usr/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file> <lscale>"
    exit 1
fi

bin/gmsh $1.geo -2 -clscale $2 -format ply2
python3 mesh_converter.py $1.ply2
rm $1.ply2

