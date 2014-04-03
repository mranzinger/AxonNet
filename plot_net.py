#!/usr/bin/env python

import numpy as n
import pylab as pl
import math as m
import os
import sys

def getErrorRates(fileName):
    errRates = {'train': [], 'test': []}

    for l in open(fileName, 'r').readlines():
        if 's' not in l: continue
        if 'TEST' not in l:
            errRates['train'] += [1. - float(l.strip().split()[2])]
        else:
            errRates['test'] += [1. - float(l.strip().split()[2])]
    
    return errRates

def showCost(fileName):
    errRates = getErrorRates(fileName)

    numCycles = len(errRates['train'])
    
    testErrors = n.row_stack(errRates['test'])
    testErrors = n.tile(testErrors, (1, 2000))
    testErrors = list(testErrors.flatten())
    testErrors += [testErrors[-1]] * max(0, len(errRates['train']) - len(errRates['test']))
    testErrors = testErrors[:len(errRates['train'])]

    # TODO: Use the actual training set size here, not the MNIST size
    numEpochs = numCycles / (50000 / 128)

    pl.figure(1)
    x = range(0, numCycles)

    print "Plotting Range:", x[0],"to",x[-1]

    pl.plot(x, errRates['train'], 'k-', label='Training')
    pl.plot(x, testErrors, 'r-', label='Held-Out')
    pl.legend()

    tickLocations = (numCycles, (len(errRates['train']) - len(errRates['test'])) % numCycles + 1, numCycles)
    epochGranularity = max(1, int(m.ceil(numEpochs / 20.)))
    epochGranularity = int(m.ceil(float(epochGranularity)/10) * 10)

    tickLabels = map(lambda x: str((x[1] / numCycles)) if x[0] % epochGranularity == epochGranularity - 1 else '', enumerate(tickLocations))

    pl.xticks(tickLocations, tickLabels)
    pl.xlabel('Batches')
    pl.ylabel('Error Rate')

    print len(errRates['train'])
    print len(errRates['test'])

if __name__ == "__main__":
    if (os.path.exists(sys.argv[1])):
        showCost(sys.argv[1])
        pl.show()
    else:
        print 'File Not Found'
