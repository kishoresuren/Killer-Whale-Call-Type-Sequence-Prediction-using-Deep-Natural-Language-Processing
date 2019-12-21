# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:04:19 2019

@author: Kishore
"""
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Splitting input corpus into train and test sets')
parser.add_argument('--trainPercentage', default='0.8',required=False,
                    help='A value between 0 and 1 signifying the train set size')
parser.add_argument('--windowWidth', default='5',required=False,
                    help='Enter width of window for the input')

args = parser.parse_args()


sequenceArray=[]
yearArray=[]

    

with open("output_21.txt", "r") as ins:
    for line in ins:
        yearArray.append(line.split(',')[1].split('_')[0].strip().replace('\n',''))
        sequenceArray.append(line.split(',')[2].strip().replace('\n',''))
    
print(len(sequenceArray))
print(len(yearArray))
XList=[]
yList=[]
sequenceList=[]
trainPercent = float(args.trainPercentage)
    
windowWidth = int(args.windowWidth)  # To be parameterized
seqLen = len(sequenceArray)

for k in range(0,seqLen):
    if k>= (seqLen-windowWidth):
        break
    XList.append(sequenceArray[k:k+windowWidth])
    yList.append(sequenceArray[k+windowWidth])
    sequenceList.append(sequenceArray[k:k+windowWidth+1])

XTrainList = XList[0:int(trainPercent*len(XList))]
XTestList = XList[int(trainPercent*len(XList)):len(XList)]
yTrainList = yList[0:int(trainPercent*len(yList))]
yTestList = yList[int(trainPercent*len(yList)):len(yList)]
testSequenceList = sequenceList[int(trainPercent*len(XList)):len(XList)]
np.save('XTrainSet' , XTrainList)
np.save('YTrainSet' , yTrainList)
np.save('XTestSet' , XTestList)
np.save('YTestSet' , yTestList)
    