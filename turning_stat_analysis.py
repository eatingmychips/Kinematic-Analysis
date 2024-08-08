import csv 
import pandas as pd
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal
from turning_analysis import *
import statistics as stat
from os import listdir



######## Here we import the files necessary for turning analysis #######

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

zero_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\0degreesTurning\\"+x 
             for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\0degreesTurning")]

forty_five_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\45degreesTurning\\"+x 
                      for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\45degreesTurning")]

ninety_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\90degreesTurning\\"+x 
                  for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\90degreesTurning")]

understanding = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\45degreesTurning\\B3_45degrees_turning_4DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"]

def stat_analysis(files): 
    heading_angles = []
    rotational_speed = []

    turning_angles_comb = []
    turning_speed_comb = []

    turn_distance_comb = []

    for file in files: 
        parts = file_read(file)

        #Gather the list of heading angles
        heading_angles.append(heading_angle(parts))
    
        #Gather the list of rotational speeds
        rotational_speed.append(ang_vel(heading_angle(parts)))

        ##### MAYBE INSERT GETTING BEFORE AND AFTER  
        

        #turn_distance_comb.append(turn_distance(turning_angles))


    for heading, speed in zip(heading_angles, rotational_speed):
            turning_angles, turning_speed = turning(heading, speed)
            turning_angles_comb.append(turning_angles)
            turning_speed_comb.append(np.mean(turning_speed))
         
    for turning_angles, file in zip(turning_angles_comb, files):
         print("For file: ", file, " \n Length of turn is: ", len(turning_angles), " \n And we see the angles are: ", turning_angles)

    return turning_speed_comb

w_0 = stat_analysis(zero_degrees)
w_45 = stat_analysis(forty_five_degrees)
w_90 = stat_analysis(ninety_degrees)



def velocity_plot(fig):
    vels = [w_0, w_45, w_90]
    devs = [stat.stdev(x) for x in vels]
    mean = [stat.mean(x) for x in vels]

    #### t-test (Welchs https://en.wikipedia.org/wiki/Welch%27s_t-test) ####
    t_zero_fotry = (mean[0] - mean[1])/np.sqrt((devs[0]/np.sqrt(len(zero_degrees)))**2 + (devs[1]/np.sqrt(len(forty_five_degrees)))**2)
    t_zero_ninety = (mean[0] - mean[2])/np.sqrt((devs[0]/np.sqrt(len(zero_degrees)))**2 + (devs[2]/np.sqrt(len(ninety_degrees)))**2)
    t_forty_ninety = (mean[1] - mean[2])/np.sqrt((devs[1]/np.sqrt(len(forty_five_degrees)))**2 + (devs[2]/np.sqrt(len(ninety_degrees)))**2)
    print('***** Students T-test for avarage velocity *******')
    print('The Students t-test between 0 and 45 degrees for average velocity is: ', t_zero_fotry)
    print('The Students t-test between 0 and 90 degrees for average velocity is', t_zero_ninety)
    print('The Students t-test between 45 and 90 degrees for average velocity is', t_forty_ninety)

    ax8 = fig.add_subplot(2,2,4)
    #ax8.set_xticklabels(["0 degrees", "45 degrees", "90 degrees"])
    ax8.boxplot(vels)
    ax8.set_title("Average Angular Velocity vs Angle of Inclination")
    ax8.set_ylabel("Avg Velocity (body lengths / second)")
    print('\n')
    print('**** Average Velocities *****')
    zero_mean = np.mean(w_0)
    forty_mean = np.mean(w_45)
    ninety_mean = np.mean(w_90)
    print('Average angular velocity at 0 degrees: ', abs(zero_mean))
    print('Average angular velocity at 45 degrees: ', abs(forty_mean))
    print('Average angular velocity at 90 degrees: ', abs(ninety_mean))


fig1 = plt.figure()
velocity_plot(fig1)
plt.show()