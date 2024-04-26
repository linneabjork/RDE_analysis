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

        V_raw = []
        I_raw = []
        for i in range(len(data)):
            V_raw.insert(
                i, float(data[i][1].replace(",", "."))
            )  # string to float. Use "."as decimal sign
            I_raw.insert(i, float(data[i][2].replace(",", ".")))

        I_raw = [i * 1000 for i in I_raw]  # convert to mA

    return V_raw, I_raw


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

        background_I = [i * 1000 for i in background_I]

    return background_V, background_I


# Checking the potential of the reference electrode.
def reference_check(voltage, current, x_lower_limit, x_upper_limit):
    zero_current_x = np.linspace(x_lower_limit, x_upper_limit, len(voltage))
    zero_current_y = np.linspace(0, 0, len(current))

    intersection_indices = np.argwhere(
        np.diff(np.sign(current - zero_current_y))
    ).flatten()
    ref_potential_forward = voltage[intersection_indices[0]]
    ref_potential_backward = voltage[intersection_indices[1]]

    print("Shift by", ref_potential_forward, "V.")
    reference = np.abs(ref_potential_forward)

    plt.plot(voltage, current)
    plt.plot(zero_current_x, zero_current_y)
    plt.plot(ref_potential_forward, current[intersection_indices[0]], "x")
    plt.xlabel("E [V]")
    plt.ylabel("I [mA]")
    plt.title("Reference check")

    return reference


# Correction to RHE potential with voltage as a list
def RHE_correction(voltage, reference):
    for i in range(len(voltage)):
        voltage[i] = voltage[i] + reference
    return voltage


def RHE_correction_background(background_matrix, reference):
    for j in range(len(background_matrix[0])):
        if j % 2 == 0:
            for i in range(len(background_matrix)):
                background_matrix[i][j] = background_matrix[i][j] + reference
    return background_matrix


# IR-drop correction
def ir_drop_correction(voltage, background_I, ir_comp):
    for i in range(len(voltage)):
        voltage[i] = voltage[i] - background_I[i] * ir_comp  # U=R*I
    return voltage


# Subtracting background from current
def background_correction_current(current, background_I):
    for i in range(len(current)):
        current[i] = current[i] - background_I[i]
    return current


# Normalizing current values to geometric surface area and mass
def normalizing(current, loading, A_geo):
    mass = loading * A_geo  # ug/cm2 * cm2 = ug

    mass_I = np.array(current) / mass  # mA/ug
    surface_I = np.array(current) / A_geo  # mA/cm2

    return mass_I, surface_I


class ECSA_calculation:
    def __init__(self, V_RHE, I, background_V_RHE, background_I):
        self.V_RHE = V_RHE
        self.I = I
        self.background_V_RHE = background_V_RHE
        self.background_I = background_I

    def calculating_ECSA(
        self,
        lower_voltage,
        upper_voltage,
        DL_start,
        DL_end,
        tolerence,
        scan_rate,
        charge_ecsa,
        Pt_mass,
    ):
        self.V_RHE = list(self.V_RHE)
        V_lower = next(x for x in self.V_RHE if x >= lower_voltage)
        V_lower_index = self.V_RHE.index(V_lower)

        # Find which index corresponds to the switching potential
        V_upper = next(x for x in self.V_RHE if x >= upper_voltage)
        V_upper_index = self.V_RHE.index(V_upper)

        # Defining the linear slope in the double layer region
        x_DL_start = next(x for x in self.V_RHE if x >= DL_start)
        x_DL_end = next(x for x in self.V_RHE if x >= DL_end)
        x_DL = [x_DL_start, x_DL_end]
        y_DL = [
            self.background_I[self.V_RHE.index(x_DL_start)],
            self.background_I[self.V_RHE.index(x_DL_end)],
        ]
        baseline_coefficients_DL = np.polyfit(x_DL, y_DL, 1)
        baseline_DL = np.poly1d(baseline_coefficients_DL)
        self.x_axis = np.linspace(V_lower, V_upper, len(self.background_V_RHE))
        self.y_axis = baseline_DL(self.x_axis)

        self.intersection_index = np.argwhere(
            np.diff(np.sign(self.background_I - self.y_axis))
        ).flatten()

        # Finding lower limit for CO peak
        CO_first_x = next(x for x in self.V_RHE if x >= DL_end)
        CO_first_index = self.V_RHE.index(CO_first_x)

        slope_values = []
        for i in range(CO_first_index, V_upper_index, 2):
            CO_slope_piece_x = [self.V_RHE[i], self.V_RHE[i + 5]]
            CO_slope_piece_y = [self.I[i], self.I[i + 5]]
            line_coeff = np.polyfit(CO_slope_piece_x, CO_slope_piece_y, 1)
            slope_values.append(line_coeff[0])
            if (
                np.isclose(line_coeff[0], baseline_coefficients_DL[0], atol=tolerence)
                == False
            ):
                self.CO_lower_index = i
                break

        # Idea: two integrations, one each for the CO peak and the HUPD region.
        # Create a respective baseline (straight line) to be subtracted from the
        # current to get the respective charges and so the ECSA.

        CO_start_x = self.V_RHE[self.CO_lower_index]
        CO_start_y = self.I[self.CO_lower_index]

        hupd_start_x = self.V_RHE[self.intersection_index[0]]
        hupd_start_y = self.background_I[self.V_RHE.index(hupd_start_x)]
        hupd_end_x, hupd_end_y = x_DL[0], y_DL[0]  # when HUPD ends, double layer region

        self.V_CO, self.I_CO = [], []
        self.V_hupd, self.I_hupd = [], []
        background_V_CO, background_I_CO = [], []
        self.background_V_hupd, self.background_I_hupd = [], []

        for i in range(V_lower_index, V_upper_index):
            if self.I[i] >= 0 and (CO_start_x <= self.V_RHE[i] <= V_upper):
                self.I_CO.append(self.I[i])
                self.V_CO.append(self.V_RHE[i])
                background_I_CO.append(self.background_I[i])
                background_V_CO.append(self.background_V_RHE[i])
            if self.background_I[i] >= hupd_start_y and (
                hupd_start_x <= self.V_RHE[i] <= hupd_end_x
            ):
                self.I_hupd.append(self.I[i])
                self.V_hupd.append(self.V_RHE[i])
                self.background_I_hupd.append(self.background_I[i])
                self.background_V_hupd.append(self.background_V_RHE[i])

        # Create a baseline to be subtracted from CO peak/HUPD
        x_CO = [CO_start_x, V_upper]
        y_CO = [CO_start_y, self.I[V_upper_index]]
        baseline_coefficients_CO = np.polyfit(x_CO, y_CO, 1)
        baseline_CO = np.poly1d(baseline_coefficients_CO)
        self.x_axis_CO = np.linspace(CO_start_x, V_upper, len(self.V_CO))
        self.y_axis_CO = baseline_CO(self.x_axis_CO)

        self.I_difference_CO = []
        for i in range(len(self.I_CO)):
            self.I_difference_CO.append(self.I_CO[i] - self.y_axis_CO[i])

        # Same slope as in the double layer region, but for the HUPD interval
        self.x_axis_hupd = np.linspace(hupd_start_x, hupd_end_x, len(self.V_hupd))
        self.y_axis_hupd = baseline_DL(self.x_axis_hupd)

        self.I_difference_hupd = []
        for i in range(len(self.I_hupd)):
            self.I_difference_hupd.append(
                self.background_I_hupd[i] - self.y_axis_hupd[i]
            )

        # Integrating
        charge1 = 0.001 * np.trapz(
            self.I_difference_CO, self.V_CO
        )  # currents in mA - need SI-units for correct conversion
        charge1 = (
            0.5 * charge1 / scan_rate
        )  # V*A/(V/s)=A*s=C. Two electrons required to reduce CO
        ECSA_CO = np.abs(charge1 / charge_ecsa)  # C/(C/cm2)=cm2
        specific_ECSA_CO = ECSA_CO / Pt_mass * 100  # converison to m2/g(Pt)

        charge2 = 0.001 * np.trapz(self.I_difference_hupd, self.V_hupd)
        charge2 = charge2 / scan_rate  # C
        ECSA_hupd = np.abs(charge2 / charge_ecsa)  # cm2
        specific_ECSA_hupd = ECSA_hupd / Pt_mass * 100  # m2/g(Pt)

        print(
            "Specific ECSA CO str:",
            specific_ECSA_CO,
            "m2/g(Pt) \tECSA CO str:",
            ECSA_CO,
            "cm2 \nSpecific ECSA HUPD:  ",
            specific_ECSA_hupd,
            "m2/g(Pt) \tECSA HUPD:",
            ECSA_hupd,
            "cm2"
            "\n\t-------------------------------------------------------------------",
        )

        specific_ECSAs = [specific_ECSA_CO, specific_ECSA_hupd]
        ECSAs = [ECSA_CO, ECSA_hupd]  # m2/g(Pt)

        return specific_ECSAs, ECSAs

    def plotting(self):
        plt.plot(self.V_RHE, self.I, label="CO stripping")
        plt.plot(
            self.background_V_RHE, self.background_I, label="CV after CO stripping"
        )
        plt.plot(self.x_axis, self.y_axis, label="Baseline")
        plt.plot(
            self.x_axis[self.intersection_index[0]],
            self.background_I[self.intersection_index[0]],
            "x",
            label="Intersection",
        )
        plt.plot(
            self.V_RHE[self.CO_lower_index],
            self.I[self.CO_lower_index],
            "x",
            label="Start of CO peak",
        )
        plt.xlabel("E vs. RHE [V]")
        plt.ylabel("i [mA]")
        plt.legend()
        plt.title("CO stripping & background CV")

        fig1, ax1 = plt.subplots()
        ax1.plot(self.V_CO, self.I_CO, label="CO peak")
        ax1.plot(self.x_axis_CO, self.y_axis_CO, label="Baseline")
        ax1.plot(self.V_CO, self.I_difference_CO, label="Difference")
        ax1.fill_between(self.V_CO, self.I_difference_CO, label="Integrated area")
        ax1.legend()
        ax1.set_xlabel("E vs. RHE [V]")
        ax1.set_ylabel("i [mA]")
        ax1.set_title("CO peak")

        fig2, ax2 = plt.subplots()
        ax2.plot(self.background_V_hupd, self.background_I_hupd, label="CV")
        ax2.plot(self.x_axis_hupd, self.y_axis_hupd, label="Baseline")
        ax2.plot(self.V_hupd, self.I_difference_hupd, label="Difference")
        ax2.fill_between(self.V_hupd, self.I_difference_hupd, label="Integrated area")
        ax2.legend()
        ax2.set_xlabel("E vs. RHE [V]")
        ax2.set_ylabel("i [mA]")
        ax2.set_title("HUPD region")


def obtain_activities(
    diffusion_voltage, kinetic_voltage, corrected_data_RHE, specific_ECSAs
):
    # Extracting I and I_d (in mA)
    forward_sweeps = copy.deepcopy(corrected_data_RHE)
    switch_voltage = np.max(forward_sweeps[:, 4])  # 1600 rpm as reference
    switch_voltage_index = list(forward_sweeps[:, 4]).index(switch_voltage)

    V_forward_sweeps = []
    I_forward_sweeps = []
    for i in range(len(forward_sweeps[0])):
        if i % 2 == 0:
            V_forward_sweeps.append((forward_sweeps[0:switch_voltage_index, i]))
        else:
            I_forward_sweeps.append((forward_sweeps[0:switch_voltage_index, i]))
    V_forward_sweeps = np.transpose(V_forward_sweeps)
    I_forward_sweeps = np.transpose(I_forward_sweeps)

    # V09_index = []
    I_measured = []
    I_d = []
    for i in range(len(V_forward_sweeps[0])):
        V_list = list(V_forward_sweeps[:, i])
        V09_elm = next(x for x in V_list if x >= kinetic_voltage)
        V09_index = V_list.index(V09_elm)
        V04_elm = next(x for x in V_list if x >= diffusion_voltage)
        V04_index = V_list.index(V04_elm)

        I_list = list(I_forward_sweeps[:, i])
        I_measured.append(I_list[V09_index])
        I_d.append(I_list[V04_index])

    I_k = []
    for i in range(len(I_measured)):
        elm = (I_measured[i] * I_d[i]) / (I_d[i] - I_measured[i])
        I_k.append(np.abs(elm))
        I_d[i] = np.abs(I_d[i])

    print(
        "Diffusion limiting current at",
        diffusion_voltage,
        "and 1600 rpm V [mA]:" "\n",
        I_d[2],
    )

    SA_CO = [i / (specific_ECSAs[0] / 100) for i in I_k]  # ECSAs[0]
    print("\nSpecific activity (CO) at 0.9 V and 1600 rpm [mA/cm2]:" "\n", SA_CO[2])

    SA_hupd = [i / (specific_ECSAs[1] / 100) for i in I_k]  # ECSAs[1]
    print("\nSpecific activity (HUPD) at 0.9 V and 1600 rpm [mA/cm2]:" "\n", SA_hupd[2])

    MA = [i * (specific_ECSAs[0] / 100) for i in SA_CO]
    print("\n Mass activity at 0.9 V and 1600 rpm [mA/ug(Pt)]:" "\n", MA[2])

    return I_d, I_k, SA_CO, SA_hupd, MA
