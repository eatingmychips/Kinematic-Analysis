import csv 
import pandas as pd
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal
from analysis import *
import statistics as stat
from os import listdir
from scipy import stats


### Collect Files data ###
def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

zero_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\0 degrees\\"+x 
             for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\0 degrees")]

forty_five_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\45 degrees\\"+x 
                      for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\45 degrees")]

ninety_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\90 degrees\\"+x 
                  for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\90 degrees")]


all_files = zero_degrees + forty_five_degrees + ninety_degrees

### Add coloured dots to indicate 0, 45, 90 degrees ###


def spread_extraction(leg):
    leg_x = [leg[i][0] for i in range(len(leg))]
    leg_y = [leg[i][1] for i in range(len(leg))]
    
    if leg_x[4] < 0: 
        max_x = np.median(leg_x) - 3.5*stat.stdev(leg_x)
    else: 
        max_x = np.median(leg_x) + 3.5*stat.stdev(leg_x)
        
    vert_top = np.median(leg_y) + 3.5*stat.stdev(leg_y)
    vert_bot = np.median(leg_y) - 3.5*stat.stdev(leg_y)
    vert = vert_top - vert_bot

    return max_x, vert


def rel_analysis(files):
    #Declare empty lists for averages of velocity, leg spread and gait timing
    vel_avg = []
    cycle_times = []
    comb_left1_hor = []
    comb_left2_hor = []
    comb_left3_hor = []
    comb_right1_hor = []
    comb_right2_hor = []
    comb_right3_hor = []

    comb_left1_ver = []
    comb_left2_ver = []
    comb_left3_ver = []
    comb_right1_ver = []
    comb_right2_ver = []
    comb_right3_ver = []


    left1_time = []
    left2_time = []
    left3_time = []
    right1_time = []
    right2_time = []
    right3_time = []

    for file in files: 
        ## Read all files in and pass through moving average filter##
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
    
        ## Calculate avg body velocity and append to vel_avg list##
        body_v = body_vel(middle)
        avg_vel = np.mean(body_v)
        vel_avg.append(avg_vel)


        leg_velocity = [leg_abs_velocity(parts[0]), leg_abs_velocity(parts[1]), leg_abs_velocity(parts[2]),
                leg_abs_velocity(parts[3]), leg_abs_velocity(parts[4]), leg_abs_velocity(parts[5])]
        

        
        rot_parts = part_rotation(parts)

        left1 = rot_parts[0]
        left1 = [(a,b) for a,b in left1 if a < -5]
        left2 = rot_parts[1]
        left2 = [(a,b) for a,b in left2 if a < -5]
        left3 = rot_parts[2]
        left3 = [(a,b) for a,b in left3 if a < -5]
        right1 = rot_parts[3]
        right1 = [(a,b) for a,b in right1 if a > 5]
        right2 = rot_parts[4]
        right2 = [(a,b) for a,b in right2 if a > 5]
        right3 = rot_parts[5]
        right3 = [(a,b) for a,b in right3 if a > 5]




        ### Get horizontal spread and vertical spread ###
        hor_spread_left1, ver_spread_left1 = spread_extraction(left1)
        hor_spread_left2, ver_spread_left2 = spread_extraction(left2)
        hor_spread_left3, ver_spread_left3 = spread_extraction(left3)

        hor_spread_right1, ver_spread_right1 = spread_extraction(right1)
        hor_spread_right2, ver_spread_right2 = spread_extraction(right2)
        hor_spread_right3, ver_spread_right3 = spread_extraction(right3)


        ### Append these to list from all files ###
        comb_left1_hor.append(-1*hor_spread_left1)
        comb_left2_hor.append(-1*hor_spread_left2)
        comb_left3_hor.append(-1*hor_spread_left3)

        comb_right1_hor.append(hor_spread_right1)
        comb_right2_hor.append(hor_spread_right2)
        comb_right3_hor.append(hor_spread_right3)

        comb_left1_ver.append(ver_spread_left1)
        comb_left2_ver.append(ver_spread_left2)
        comb_left3_ver.append(ver_spread_left3)

        comb_right1_ver.append(ver_spread_right1)
        comb_right2_ver.append(ver_spread_right2)
        comb_right3_ver.append(ver_spread_right3)



        horizontal_spread = [comb_left1_hor, comb_left2_hor, comb_left3_hor, 
                             comb_right1_hor, comb_right2_hor, comb_right3_hor]
        
        vert_spread = [comb_left1_ver, comb_left2_ver, comb_left3_ver, 
                       comb_right1_ver, comb_right2_ver, comb_right3_ver]
        

        ### Calculate time of gait swing ###
    
        left1_time.append(np.mean(leg_time(leg_velocity[0])[1]))
        left2_time.append(np.mean(leg_time(leg_velocity[1])[1]))
        left3_time.append(np.mean(leg_time(leg_velocity[2])[1]))

        right1_time.append(np.mean(leg_time(leg_velocity[3])[1]))
        right2_time.append(np.mean(leg_time(leg_velocity[4])[1]))
        right3_time.append(np.mean(leg_time(leg_velocity[5])[1]))
        
        cycle_times = [left1_time, left2_time, left3_time, right1_time, right2_time, right3_time]



    return vel_avg, horizontal_spread, cycle_times, vert_spread



#### Here we call the statistical analysis function once per angle and get a spread, 
#### velocity and stand/swing output. 


vel, horizontal_spread, cycle_times, vert_spread = rel_analysis(all_files)

def best_fit(pt, x, y, degree): 
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    x_sorted = np.sort(x)
    best_fit_line = polynomial(x_sorted)
    pt.plot(x_sorted, best_fit_line, color = 'red', label=f'{degree}-degree Polynomial Fit')


def horizontal_spread_plot(fig):
    ax1 = fig.add_subplot(2,3,1)
    ax1.scatter(horizontal_spread[0], vel)
    ax1.set_title("Leg: Left1, Velocity vs Horizontal Spread")
    ax1.set_xlabel("Horizontal Spread")
    ax1.set_ylabel("Velocity")
    best_fit(ax1, horizontal_spread[0], vel, 1)

    ax2 = fig.add_subplot(2,3,2)
    ax2.scatter(horizontal_spread[1], vel)
    ax2.set_title("Leg: Left2, Velocity vs Horizontal Spread")
    ax2.set_xlabel("Horizontal Spread")
    ax2.set_ylabel("Velocity")
    best_fit(ax2, horizontal_spread[1], vel, 1)

    ax3 = fig.add_subplot(2,3,3)
    ax3.scatter(horizontal_spread[2], vel)
    ax3.set_title("Leg: Left3, Velocity vs Horizontal Spread")
    ax3.set_xlabel("Horizontal Spread")
    ax3.set_ylabel("Velocity")
    best_fit(ax3, horizontal_spread[2], vel, 1)

    ax4 = fig.add_subplot(2,3,4)
    ax4.scatter(horizontal_spread[3], vel)
    ax4.set_title("Leg: Right1, Velocity vs Horizontal Spread")
    ax4.set_xlabel("Horizontal Spread")
    ax4.set_ylabel("Velocity")
    best_fit(ax4, horizontal_spread[3], vel, 1)

    ax5 = fig.add_subplot(2,3,5)
    ax5.scatter(horizontal_spread[4], vel)
    ax5.set_title("Leg: Right2, Velocity vs Horizontal Spread")
    ax5.set_xlabel("Horizontal Spread")
    ax5.set_ylabel("Velocity")
    best_fit(ax5, horizontal_spread[4], vel, 1)

    ax6 = fig.add_subplot(2,3,6)
    ax6.scatter(horizontal_spread[5], vel)
    ax6.set_title("Leg: RIght3, Velocity vs Horizontal Spread")
    ax6.set_xlabel("Horizontal Spread")
    ax6.set_ylabel("Velocity")
    best_fit(ax6, horizontal_spread[5], vel, 1)

def time_plot(fig): 
    ax7 = fig.add_subplot(2,3,1)
    ax7.scatter(cycle_times[0], vel)
    ax7.set_title("Leg: Left1, Velocity vs Gait cycle time")
    ax7.set_xlabel("Horizontal Spread")
    ax7.set_ylabel("Velocity")
    best_fit(ax7, cycle_times[0], vel, 1)

    ax8 = fig.add_subplot(2,3,2)
    ax8.scatter(cycle_times[1], vel)
    ax8.set_title("Leg: Left2, Velocity vs Gait cycle time")
    ax8.set_xlabel("Horizontal Spread")
    ax8.set_ylabel("Velocity")
    best_fit(ax8, cycle_times[1], vel, 1)

    ax9 = fig.add_subplot(2,3,3)
    ax9.scatter(cycle_times[2], vel)
    ax9.set_title("Leg: Left3, Velocity vs Gait cycle time")
    ax9.set_xlabel("Horizontal Spread")
    ax9.set_ylabel("Velocity")
    best_fit(ax9, cycle_times[2], vel, 1)

    ax10 = fig.add_subplot(2,3,4)
    ax10.scatter(cycle_times[3], vel)
    ax10.set_title("Leg: Right1, Velocity vs Gait cycle time")
    ax10.set_xlabel("Horizontal Spread")
    ax10.set_ylabel("Velocity")
    best_fit(ax10, cycle_times[3], vel, 1)

    ax11 = fig.add_subplot(2,3,5)
    ax11.scatter(cycle_times[4], vel)
    ax11.set_title("Leg: Right2, Velocity vs Gait cycle time")
    ax11.set_xlabel("Horizontal Spread")
    ax11.set_ylabel("Velocity")
    best_fit(ax11, cycle_times[4], vel, 1)

    ax12 = fig.add_subplot(2,3,6)
    ax12.scatter(cycle_times[5], vel)
    ax12.set_title("Leg: Right3, Velocity vs Gait cycle time")
    ax12.set_xlabel("Horizontal Spread")
    ax12.set_ylabel("Velocity")
    best_fit(ax12, cycle_times[5], vel, 1)

def vert_spread_plot(fig):
    ax13 = fig.add_subplot(2,3,1)
    ax13.scatter(vert_spread[0], vel)
    ax13.set_title("Leg: Left1, Velocity vs Vertical Spread")
    ax13.set_xlabel("Vertical Spread")
    ax13.set_ylabel("Velocity")
    best_fit(ax13, vert_spread[0], vel, 1)

    ax14 = fig.add_subplot(2,3,2)
    ax14.scatter(vert_spread[1], vel)
    ax14.set_title("Leg: Left2, Velocity vs Vertical Spread")
    ax14.set_xlabel("Vertical Spread")
    ax14.set_ylabel("Velocity")
    best_fit(ax14, vert_spread[1], vel, 1)

    ax15 = fig.add_subplot(2,3,3)
    ax15.scatter(vert_spread[2], vel)
    ax15.set_title("Leg: Left3, Velocity vs Vertical Spread")
    ax15.set_xlabel("Vertical Spread")
    ax15.set_ylabel("Velocity")
    best_fit(ax15, vert_spread[2], vel, 1)

    ax16 = fig.add_subplot(2,3,4)
    ax16.scatter(vert_spread[3], vel)
    ax16.set_title("Leg: Right1, Velocity vs Vertical Spread")
    ax16.set_xlabel("Vertical Spread")
    ax16.set_ylabel("Velocity")
    best_fit(ax16, vert_spread[3], vel, 1)

    ax17 = fig.add_subplot(2,3,5)
    ax17.scatter(vert_spread[4], vel)
    ax17.set_title("Leg: Right2, Velocity vs Vertical Spread")
    ax17.set_xlabel("Vertical Spread")
    ax17.set_ylabel("Velocity")
    best_fit(ax17, vert_spread[4], vel, 1)

    ax18 = fig.add_subplot(2,3,6)
    ax18.scatter(vert_spread[5], vel)
    ax18.set_title("Leg: RIght3, Velocity vs Vertical Spread")
    ax18.set_xlabel("Vertical Spread")
    ax18.set_ylabel("Velocity") 
    best_fit(ax18, vert_spread[5], vel, 1)


fig1 = plt.figure()
horizontal_spread_plot(fig1)
plt.show()

fig2 = plt.figure()
time_plot(fig2)
plt.show()

fig3 = plt.figure()
vert_spread_plot(fig3)
plt.show()