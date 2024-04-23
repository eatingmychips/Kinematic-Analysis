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
        
        stand = []
        for foot in leg_velocity:
            for key, iter in it.groupby(foot):
                if key == 1:
                    stand.append(len(list(iter)))

        stand = [i for i in stand if i < 10]
        
        avg_stand = np.mean(stand)

        stand_tot.append(avg_stand)

        #Rotate Parts and calculate the average leg spread
        rot_parts = part_rotation(parts)
        leg_spread = avg_leg_spread(rot_parts)
        spread_avg.append(leg_spread)






    return spread_avg, vel_avg, stand_tot


spread_0, vel_0, stand_0 = stat_analysis(zero_degrees)
spread_45, vel_45, stand_45 = stat_analysis(forty_five_degrees)
spread_90, vel_90, stand_90 = stat_analysis(ninety_degrees)




def spread_plot(fig):

    label_sp = ["0 degrees", "45 degrees", "90 degrees"]
    spreads = [spread_0, spread_45, spread_90]
    

    ax1 = fig.add_subplot(2,2,1)
    ax1.boxplot(spreads)
    ax1.set_xticklabels(label_sp)
    ax1.set_title("Leg Spread")

def velocity_plot(fig):
    vels = [vel_0, vel_45, vel_90]
    label_v = ["0 degrees", "45 degrees", "90 degrees"]
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_xticklabels(label_v)
    ax2.boxplot(vels)
    ax2.set_title("Velocity")


def stand_plot(fig): 
    stands = [stand_0, stand_45, stand_90]
    label_st = ["0 degrees", "45 degrees", "90 degrees"]

    ax3 = fig.add_subplot(2,2,3)
    ax3.set_xticklabels(label_st)
    ax3.boxplot(stands)
    ax3.set_title("Avg Stand Time")

### Declare the first figure and run all plotting functions ###
fig = plt.figure()
spread_plot(fig)
velocity_plot(fig)
stand_plot(fig)

plt.show()



"""Here we plot the gait phase. We will only choose specific files and show them one by one
    This will reduce the need to extract meaningful data from all the phase plotting. The data
    Analysis above will be used to show data averages. This is a bad comment I will fix it up later"""

### Plotting of gait phase ###
fig2 = plt.figure()

### Make gait phase plotting data from specific files ###
left1_0, left2_0, left3_0, right1_0, right2_0, right3_0 = gait_phase_plotting(file0_2)
left1_45, left2_45, left3_45, right1_45, right2_45, right3_45 = gait_phase_plotting(file45_2)
left1_90, left2_90, left3_90, right1_90, right2_90, right3_90 = gait_phase_plotting(file90_1)


ax5 = fig2.add_subplot(2,2,1)
size0 = range(len(left1_0))
ax5.plot(size0, left1_0, '-', color = 'blue')
ax5.plot(size0, left2_0, '-', color = 'red')
ax5.plot(size0, left3_0, '-', color = 'blue')
ax5.plot(size0, right1_0, '-', color = 'red')
ax5.plot(size0, right2_0, '-', color = 'blue')
ax5.plot(size0, right3_0, '-', color = 'red')
ax5.title.set_text('Foot Vertical Displacement vs Time (90 degrees)')

### Plot phase for 45 degrees ###
ax6 = fig2.add_subplot(2,2,2)
size45 = range(len(left1_45))
ax6.plot(size45, left1_45, '-', color = 'blue')
ax6.plot(size45, left2_45, '-', color = 'red')
ax6.plot(size45, left3_45, '-', color = 'blue')
ax6.plot(size45, right1_45, '-', color = 'red')
ax6.plot(size45, right2_45, '-', color = 'blue')
ax6.plot(size45, right3_45, '-', color = 'red')
ax6.title.set_text('Foot Vertical Displacement vs Time (45 degrees)')

### Plot phase for 90 degrees ###
ax5 = fig2.add_subplot(2,2,3)
size90 = range(len(left1_90))
ax5.plot(size90, left1_90, '-', color = 'blue')
ax5.plot(size90, left2_90, '-', color = 'red')
ax5.plot(size90, left3_90, '-', color = 'blue')
ax5.plot(size90, right1_90, '-', color = 'red')
ax5.plot(size90, right2_90, '-', color = 'blue')
ax5.plot(size90, right3_90, '-', color = 'red')
ax5.title.set_text('Foot Vertical Displacement vs Time (90 degrees)')


plt.show()








