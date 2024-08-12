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

understanding = [r"C:\Users\lachl\OneDrive\Thesis\Data\TurningData\B4_0degrees_turning_1DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"]

def stat_analysis(files, degrees): 
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
        heading = heading_angle(middle,bottom)
        heading_angles.append(heading)
        
        #Gather the list of rotational speeds
        rotational_speed.append(ang_vel(heading))

        
        ##### MAYBE INSERT GETTING BEFORE AND AFTER  
        startstop = turning(heading, rotational_speed)
        cutoff_list.append(startstop)

        leg_velocity = [leg_abs_velocity(left1), leg_abs_velocity(left2), leg_abs_velocity(left3),
                        leg_abs_velocity(right1), leg_abs_velocity(right2), leg_abs_velocity(right3)]
        
        #turn_distance_comb.append(turn_distance(turning_angles))
        #Plot gait pattern as swing stand over turn 
        #Plot leg swing scattern. 
        direction = turn_direction(heading)

        #### Plotting for heading angle #####
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(2,1,1)
        ax1.plot(heading)
        ax1.set_title(f"Heading Angle for {direction}, {degrees}")
        thickness = 0.9
        #### End Plotting for heading angle 


        #### Plotting for gait diagram ####
        ax2 = fig1.add_subplot(2,1,2)
        ax2.set_title(f"Gait diagram for {direction}, {degrees}")
        row_labels = ['left1', 'left2', 'left3', 'right1', 'right2', 'right3']
        y_positions = range(len(leg_velocity))
        for i, leg_velocity in enumerate(leg_velocity):
        # Loop over each element in the binary list
            for j, value in enumerate(leg_velocity):
                if value == 1:
                    # Plot a bar if the value is 1
                    ax2.barh(i, 1, left=j, height=thickness, color='black')
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(row_labels)
        #### End plotting for gait diagram ####



        plt.show()

    return cutoff_list, heading_angles

w_0 = stat_analysis(understanding, "0 Degrees")
w_45 = stat_analysis(forty_five_degrees, "45 Degrees")
w_90 = stat_analysis(ninety_degrees, "90 Degrees")



