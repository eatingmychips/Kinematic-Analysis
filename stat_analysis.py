import csv 
import pandas as pd
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal
from analysis import *

#Zero degrees
file0_1 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\0degrees\movie20240403_B3_0degDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file0_2 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\0degrees\movie20240403_B1_0degDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file0_3 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\0degrees\movie20240403_B2_0degDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file0_4 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\0degrees\movie20240403_B3_0deg (2)DLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file0_5 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\0degrees\movie20240403_B3_0deg (3)DLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"


#45 degrees
file45_1 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\45degrees\movie20240403_B3_45degDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file45_2 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\45degrees\movie20240403_B3_45deg_verticalDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file45_3 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\45degrees\movie20240403_B2_45deg_verticalDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file45_4 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\45degrees\movie20240403_B1_45deg_vertical_turningDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file45_5 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\45degrees\movie20240403_B1_45deg_vertical_downDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"


#90 degrees
file90_1 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\90degrees\movie20240403_B1_90deg_verticalDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file90_2 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\90degrees\movie20240403_B3_90deg_verticalDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file90_3 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\90degrees\movie20240403_B3_90deg_perpendicularDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"
file90_4 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinAnal_Videos\90degrees\movie20240403_B2_90deg_perpendicularDLC_resnet50_KinematicAnalysisApr2shuffle1_100000.csv"





zero_degrees = [file0_1, file0_2, file0_3, file0_4, file0_5]
forty_five_degrees = [file45_1, file45_2, file45_3, file45_4, file45_5]
ninety_degrees = [file90_1, file90_2, file90_3]

results = {}

def stat_analysis(files):
    vel_avg = []
    spread_avg = []
    
    stand_tot = []
    for file in files: 
        ## Feed all parts through moving average filter ##
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
        
        ## Calculate avg body velocity ##
        body_v = body_vel(middle)
        avg_vel = np.mean(body_v)
        vel_avg.append(avg_vel)

        leg_velocity = [leg_abs_velocity(left1), leg_abs_velocity(left2), leg_abs_velocity(left3),
                        leg_abs_velocity(right1), leg_abs_velocity(right2), leg_abs_velocity(right3)]
        left1_abs_vel = leg_abs_velocity(left1)
        left2_abs_vel = leg_abs_velocity(left2)
        left3_abs_vel = leg_abs_velocity(left3)
        right1_abs_vel = leg_abs_velocity(right1)
        right2_abs_vel = leg_abs_velocity(right2)
        right3_abs_vel = leg_abs_velocity(right3)
        
        stand = []
        for foot in leg_velocity:
            for key, iter in it.groupby(foot):
                if key == 1:
                    stand.append(len(list(iter)))

        print(stand)
        avg_stand = np.mean(stand)

        stand_tot.append(avg_stand)

        #Rotate Parts and calculate the average leg spread
        rot_parts = part_rotation(parts)
        leg_spread = avg_leg_spread(rot_parts)
        spread_avg.append(leg_spread)






    return spread_avg, vel_avg, stand_tot


def spread_plot(fig):
    spread_0,_,_ = stat_analysis(zero_degrees)
    spread_45,_,_ = stat_analysis(forty_five_degrees)
    spread_90,_,_ = stat_analysis(ninety_degrees)

    label_sp = ["0 degrees", "45 degrees", "90 degrees"]
    spreads = [spread_0, spread_45, spread_90]
    

    ax1 = fig.add_subplot(2,2,1)
    ax1.boxplot(spreads)
    ax1.set_xticklabels(label_sp)
    ax1.set_title("Leg Spread")

def velocity_plot(fig):
    _, vel_0,_ = stat_analysis(zero_degrees)
    _, vel_45,_ = stat_analysis(forty_five_degrees)
    _, vel_90,_ = stat_analysis(ninety_degrees)
    vels = [vel_0, vel_45, vel_90]
    label_v = ["0 degrees", "45 degrees", "90 degrees"]
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_xticklabels(label_v)
    ax2.boxplot(vels)
    ax2.set_title("Velocity")


def stand_plot(fig): 
    _,_,stand_0 = stat_analysis(zero_degrees)
    _,_,stand_45 = stat_analysis(forty_five_degrees)
    _,_,stand_90 = stat_analysis(ninety_degrees)

    stands = [stand_0, stand_45, stand_90]
    label_st = ["0 degrees", "45 degrees", "90 degrees"]

    ax3 = fig.add_subplot(2,2,3)
    ax3.set_xticklabels(label_st)
    ax3.boxplot(stands)
    ax3.set_title("Avg Stand Time")

fig = plt.figure()
spread_plot(fig)
velocity_plot(fig)
stand_plot(fig)



plt.show()




