# Script to run SFI experiments
import re, sys, io, os
import subprocess
import csv
import argparse

CNN="ResNet20"
DATATYPE="fp32"
NBIT=32
NLAYERS=20
infile="n_samples_step1.csv"


def main():
    database_samples=readSamples()
    for i in range(NBIT):
        if i==30: # I am running only the 30 bit (0 is the LSB of the FP data)
            for j in range(NLAYERS):
                n_samples=getsamples(database_samples, 31-i , j)
                if n_samples != 0:
                    command=f"python modes_comparison.py -n {CNN} --batch-size 128 -m stuck-at_params --save-compressed --target_layer_index {j} --target_layer_bit {i} --target_layer_n {n_samples}"
                    print(command)
                    os.system(command)


def readSamples():
    database_samples=[]
    with open(infile) as csv_file:
        csv_reader= csv.reader(csv_file, delimiter=";")
        for row in csv_reader:
            database_samples.append(row)
    return database_samples
            

# This function returns the number of faults (n) to inject in a specific bitposition of a specific layer
def getsamples(database_samples, bitposition, layerposition):
    if database_samples[layerposition][bitposition][0].isnumeric():
        return int(database_samples[layerposition][bitposition])
    else:
        return int(database_samples[layerposition][bitposition][1:])


if __name__ == '__main__':
    main()