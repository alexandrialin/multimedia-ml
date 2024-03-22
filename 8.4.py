# Multiple Binary Classifcations Algo8
import numpy as np

def memorize(data, labelArray):
    mecs = [] 
    thresholds = {}  
    
    for label in labelArray:
        thresholds[label] = 0 
        classData = [point for point, lbl in zip(data, label) if lbl == label]
        table = [(sum(point), label) for point in classData]
        sortedTable = sorted(table, key=lambda x: x[0])  # Sort by the sum of features
        
        previousSum = None
        for row in sortedTable:
            # If we encounter a new sum, increment the threshold for this class
            if previousSum is None or row[0] != previousSum:
                thresholds[label] += 1
                previousSum = row[0]
        
        # Compute the minimum number of bits to encode the thresholds
        minThreshs = np.log2(thresholds[label] + 1)
        # Calculate the MEC for the class and add it to the mecs list
        d = len(data[0]) 
        mecs.append((minThreshs * (d + 1)) + (minThreshs + 1))
    
    return mecs



# Regression Algo8

def memorizeRegression(data, labels):
    table = []
    
    # sum of features for each data point
    for i in range(len(data)):
        sum_of_features = sum(data[i])
        table.append((sum_of_features, labels[i]))
    
    sortedTable = sorted(table, key=lambda x: x[0])  # Sort by the sum of features

    sse = 0
    
    # Calculate the sum of squared errors
    for i in range(1, len(data)):
        # The difference between actual and predicted values
        prediction = sortedTable[i-1][0]
        actual = sortedTable[i][1]
        error = actual - prediction
        sse += error ** 2
    
    # Calculate the mean squared error
    mse = sse / len(data)
    
    return mse 