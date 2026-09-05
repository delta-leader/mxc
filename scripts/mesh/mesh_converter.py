#!/usr/bin/python3

import sys

if len(sys.argv) < 2:
  print("Please specify an input file")
if (sys.argv[1].split(".")[1] != "ply2"):
  print("Converter has been written for '.ply2' files and might not work correctly for the specified input file")
with open(sys.argv[1], "r") as input:
  data  = input.readlines()
  num_nodes = int(data[0])
  num_elems = int(data[1])
  print("File contains {} nodes and {} elements".format(num_nodes, num_elems))
  f_out = sys.argv[1].split(".")[0] + "_" + data[0] + ".inp"
  with open(f_out, "w") as output:
    output.write("!------ mesh data ------\n")
    output.write("     {}           ! nel\n".format(num_elems))
    output.write("     {}           ! nnode\n".format(num_nodes))
    output.write("## Node Coordinates\n")
    start = 2
    stop = start + num_nodes
    print("Processing nodes")
    for i in range(start, stop):
      output.write(data[i])
    start += num_nodes
    stop = stop + num_elems
    output.write("## Element Component\n")
    print("Processing elements")
    for i in range(start, stop):
      line = data[i].split(" ")
      for idx in line[2:]:
        output.write(" {}".format(int(idx) + 1))
      output.write("\n")
    output.write("\n")
    print("Finished!")
    
