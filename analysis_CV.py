# Python script to automatize analysis of stability and activity measurements

# import of modules
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import sys
import copy
from itertools import zip_longest
import csv


# Extract voltage and current from data file. index indicates how many lines of info to be removed
def get_voltage_current(filepath, index):
    with open(filepath, "r") as file:
        data = file.readlines()
        i = 0
        for line in data:  # loop through the lines in the file
            newline = line.rstrip("\n").split(
                "\t"
            )  # split the columns into a list & remove trailing characters (ex blank space) after new line
            data[i] = newline
            i += 1

        del data[0:index]  # information about the measurement, remove from data

        Vraw = []
        Iraw = []
        for i in range(len(data)):
            Vraw.insert(
                i, float(data[i][1].replace(",", "."))
            )  # string to float. Use "."as decimal sign
            Iraw.insert(i, float(data[i][2].replace(",", ".")))
    return Vraw, Iraw


# Store the background measurement in a matrix where each column corresponds to one variable
def get_background(path_background, index):
    with open(path_background, "r") as file:
        background_data = file.readlines()
        i = 0
        for line in background_data:
            newline = line.rstrip("\n").rstrip("\t\t").split("\t")
            background_data[i] = newline
            i += 1

        del background_data[0:index]

        background_V = []
        background_I = []
        for i in range(len(background_data)):
            background_V.insert(i, float(background_data[i][1].replace(",", ".")))
            background_I.insert(i, float(background_data[i][2].replace(",", ".")))

        background_matrix = np.transpose([background_V, background_I])

    return background_matrix


# Store the measurement parameters in a dictionary
def get_parameters(path_parameters):
    with open(path_parameters, "r") as file:
        parameters_dictionary = json.loads(file.read())
    return parameters_dictionary


# Correction to RHE potential with voltage as a list
def RHE_correction(voltage, parameters_dictionary):
    for i in range(len(voltage)):
        voltage[i] = voltage[i] + parameters_dictionary["reference"]
    return voltage


def RHE_correction_background(background_matrix, parameters_dictionary):
    for i in range(len(background_matrix)):
        background_matrix[i][0] = (
            background_matrix[i][0] + parameters_dictionary["reference"]
        )
    return background_matrix


# IR-drop correction
def ir_drop_correction(voltage, background_matrix, parameters_dictionary):
    for i in range(len(voltage)):
        voltage[i] = (
            voltage[i]
            - background_matrix[i][1] * parameters_dictionary["ir_comp"]  # U=R*I
        )
    return voltage


# Subtracting background from current
def background_correction_current(current, background_matrix):
    for i in range(len(current)):
        current[i] = current[i] - background_matrix[i][1]
    return current


# Normalizing current values to geometric surface area and mass
def normalizing(current, parameters_dictionary):
    mass = (
        parameters_dictionary["loading"] * parameters_dictionary["A_geo"]
    )  # mg/cm2 * cm2 = mg

    mass_I = np.array(current) * 1000 / mass
    surface_I = np.array(current) * 1000 / parameters_dictionary["A_geo"]  # mA/cm2

    return mass_I, surface_I
