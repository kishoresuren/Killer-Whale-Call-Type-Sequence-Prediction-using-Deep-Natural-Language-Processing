# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:05:10 2019

@author: Kishore
"""
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Predicting an orca sequence')
parser.add_argument('--inputSequence',required=True,
                    help='Sequence of valid sound patterns separated by commas')
parser.add_argument('--thresholdTime', default='100000',required=False,
                    help='Time gap between the start of the next sound pattern and end of the previous sound pattern within the input sequence, in seconds')
args = parser.parse_args()

tapeList=[]
timestampList=[]
yearList=[]
clusterList=[]
infoList=[]

f = open('output_21.txt','r')
infoList = f.readlines()

#Load each column of the cluster file into separate lists
tapeList = [line.strip().split(',')[0].split('/')[-1].split('-')[0] for line in infoList]
timestampList = [str(line.strip().split(',')[0].split('/')[-1].split('-')[1]) +'-'+str(line.strip().split(',')[0].split('/')[-1].split('-')[2].split('.')[0]) for line in infoList] 
yearList = [line.split(',')[1].split('_')[0].strip() for line in infoList]
clusterList = [line.split(',')[2].strip() for line in infoList]

#Copy the input sequence and the optional time key into separate variables
searchKey = args.inputSequence.split(',')
timeKey = int(args.thresholdTime)  


sublistLen=len(searchKey)

#Divide the lists into lists of sublists of length equal to the input sequence length
yearList = [yearList[n:n+sublistLen] for n in range(0,len(yearList))]
clusterList = [clusterList[n:n+sublistLen] for n in range(0,len(clusterList))]
tapeList = [tapeList[n:n+sublistLen] for n in range(0,len(tapeList))]
timestampList = [timestampList[n:n+sublistLen] for n in range(0,len(timestampList))]
indicesFound = [i for i,j in enumerate(clusterList) if j==searchKey]

yearList2 = np.array([yearList[i] for i in indicesFound]).ravel()
tapeList2 = np.array([tapeList[i] for i in indicesFound]).ravel()
timestampList2 = np.array([timestampList[i] for i in indicesFound]).ravel()

#Convert samples to seconds duration
timestampList2 = [str(round(int(ts.split('-')[0])/44100,3))+'-'+str(round(int(ts.split('-')[1])/44100,3)) for ts in timestampList2]

#Start writing into the txt file
txtfile = open('information_'+args.inputSequence.strip()+'.txt', 'w')
txtfile.write('Sequence: '+",".join(searchKey))
txtfile.write('\n\n\n')
for year in list(set(yearList2)):
    yearIndex = [i for i,j in enumerate(yearList2) if j==year]
    tapes = [tapeList2[i] for i in yearIndex]
    tapetsdic={}
    for tape in list(set(tapes)):
        suppress=True
        tapeindex = [i for i,j in enumerate(tapes) if j ==tape]
        if len([timestampList2[i] for i in tapeindex])==sublistLen:
            tsList0 = [float(timestampList2[i].split('-')[0]) for i in tapeindex]
            tsList1 = [float(timestampList2[i].split('-')[1]) for i in tapeindex]
            tsList0.sort()
            tsList1.sort()
            if tsList0[1] - tsList1[0] <=timeKey and tsList0[2]-tsList1[1] <=timeKey:
                suppress = False  #No need to suppress if each sound pattern is within thresholdTime from one another
            tsList=[str(tsList0[i])+'-'+str(tsList1[i]) for i in range(len(tsList0))]
        if not suppress: #Write only the non suppressed sequence details
            if len([timestampList2[i] for i in tapeindex])==sublistLen:
                txtfile.write('Year: '+year)
                txtfile.write('\n\n')
                txtfile.write('\t')
                txtfile.write('Tape: '+tape)
                txtfile.write('\n\n')
                for t in tsList:
                    txtfile.write('\t\t')
                    txtfile.write('Timestamp(s): '+t)
                    txtfile.write('\n\n')
    
    txtfile.write('\n\n')
    
txtfile.close()
            
