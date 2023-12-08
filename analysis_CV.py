# Python script to automatize analysis of stability and activity measurements

#import of modules
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import sys
import copy
from itertools import zip_longest
import csv


# Extract voltage and current from data file
def get_voltage_current(filepath):                  
    with open(filepath, 'r') as file:
        data = file.readlines()
        i = 0
        for line in data:                             #loop through the lines in the file
            newline = line.rstrip("\n").split("\t")   #split the columns into a list & remove trailing characters (ex blank space) after new line
            data[i] = newline
            i += 1
        
        del data[0:9]                            #information about the measurement, remove from data

        Vraw = []
        Iraw = []
        for i in range(len(data)): 
            Vraw.insert(i, float(data[i][1].replace(',','.')))      #string to float. Use "."as decimal sign
            Iraw.insert(i, float(data[i][2].replace(',','.')))    
    return Vraw, Iraw

# Store the background measurement in a matrix where each column corresponds to one variable
def get_background(path_background):
    with open(path_background, 'r') as file:
        background_data = file.readlines()
        
        i = 0
        for line in background_data: 
            newline = line.rstrip("\n").rstrip("\t\t").split("\t") 
            background_data[i] = newline
            i += 1

        background_matrix = np.zeros((len(background_data),len(background_data[0])))

        for i in range(len(background_data)):        #loop through rows first
            for j in range (len(background_data[i])):
                background_matrix[i][j] = float(background_data[i][j].replace(',','.'))
    return background_matrix

# Store the measurement parameters in a dictionary
def get_parameters(path_parameters):
    with open(path_parameters, 'r') as file:
        parameters_dictionary = json.loads(file.read())
    return parameters_dictionary

# Correction to RHE potential with voltage as a list
def RHE_correction(voltage, background_matrix, parameters_dictionary):
    for e in voltage:
        e = e - parameters_dictionary["reference"]

    for i in range(len(background_matrix[0])):
        background_matrix[0][i] = background_matrix[0][i] - parameters_dictionary["reference"]
    return voltage, background_matrix