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
    
    cutoff_list = []
    for file in files: 
        parts = file_read(file)
        left1 = moving_avg(parts[0])
        left2 = moving_avg(parts[1])
        left3 = moving_avg(parts[2])
        right1 = moving_avg(parts[3])
        right2 = moving_avg(parts[4])
        right3 = moving_avg(parts[5])
        top = moving_avg(parts[6])
        middle = moving_avg(parts[7])
        bottom = moving_avg(parts[8])
        print(len(middle))
        print(len(left1))
        #Gather the list of heading angles
        heading_angles.append(heading_angle(middle, bottom))
        print(len(heading_angle(middle, bottom)))
        #Gather the list of rotational speeds
        rotational_speed.append(ang_vel(heading_angle(middle,bottom)))

        
        ##### MAYBE INSERT GETTING BEFORE AND AFTER  
        startstop = turning(heading_angle(middle, bottom), rotational_speed)
        cutoff_list.append(startstop)

        leg_velocity = [leg_abs_velocity(left1), leg_abs_velocity(left2), leg_abs_velocity(left3),
                        leg_abs_velocity(right1), leg_abs_velocity(right2), leg_abs_velocity(right3)]
        
        #turn_distance_comb.append(turn_distance(turning_angles))
        #Plot gait pattern as swing stand over turn 
        #Plot leg swing scattern. 

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(2,1,1)
        ax1.plot(heading_angle(middle, bottom))
        thickness = 0.9

        ax2 = fig1.add_subplot(2,1,2)
        for i, leg_velocity in enumerate(leg_velocity):
        # Loop over each element in the binary list
            for j, value in enumerate(leg_velocity):
                if value == 1:
                    # Plot a bar if the value is 1
                    ax2.barh(i, 1, left=j, height=thickness, color='black')
        
        plt.show()

    return cutoff_list, heading_angles

w_0 = stat_analysis(zero_degrees[0:5])
w_45 = stat_analysis(forty_five_degrees[0:5])
w_90 = stat_analysis(ninety_degrees[0:5])



