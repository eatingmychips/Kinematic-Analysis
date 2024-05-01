import csv 
import pandas as pd
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal
from analysis import *

#Zero degrees files
file0_1 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B1_0degreees_straightDLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file0_2 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B1_0degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file0_3 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B1_0degrees_straightDLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file0_4 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B3_0degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file0_5 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B3_0degrees_straightDLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"


#45 degrees files
file45_1 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B1_45degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file45_2 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B1_45degrees_straightDLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file45_3 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B2_45degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file45_4 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B2_45degrees_straight (3)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file45_5 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B2_45degrees_straightDLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"


#90 degrees files 
file90_1 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B2_90degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file90_2 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B2_90degrees_straightDLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file90_3 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B3_90degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file90_4 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B3_90degrees_straight (3)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
file90_5 = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B3_90degrees_straightDLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"




#Declare a list of the files by walking angle
zero_degrees = [file0_1, file0_2, file0_3, file0_4, file0_5]
forty_five_degrees = [file45_1, file45_2, file45_3, file45_4, file45_5]
ninety_degrees = [file90_1, file90_2, file90_3, file90_4, file90_5]



"""This is the statistical analysis function and will be called ONCE per angle. 
So, all files of a specific angle will be run through in the inner for loop. """

def stat_analysis(files):
    #Declare empty lists for averages of velocity, leg spread and gait timing
    vel_avg = []
    spread_avg = []
    gaits = []

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

        #Create a list of leg velocity's, this will be a list of lists, with velocities binary, refer to 'analysis.py' for 
        #the leg_abs_velocty function specifics
        leg_velocity = [leg_abs_velocity(left1), leg_abs_velocity(left2), leg_abs_velocity(left3),
                        leg_abs_velocity(right1), leg_abs_velocity(right2), leg_abs_velocity(right3)]
        
        #Extract the 'length' of all the swing and stand phases. 
        swing = []
        stand = []
        for foot in leg_velocity:
            for key, iter in it.groupby(foot):
                if key == 1:
                    swing.append(len(list(iter)))
                elif key == 0: 
                    stand.append(len(list(iter)))
        
        #So for each file we have an average stand and an average swing time
        stand = [x for x in stand if 3 < x < 90]
        swing = [x for x in swing if 3 < x ]
        
        avg_stand = np.mean(stand)
        avg_swing = np.mean(swing)

        #Convert this to a percentage and a tuple form. 
        gait = [avg_stand/(avg_stand+avg_swing)*100, avg_swing/(avg_stand + avg_swing)*100]

        #Append this average to the gaits list.
        gaits.append(gait)

        #Rotate Parts and calculate the average leg spread
        rot_parts = part_rotation(parts)
        leg_spread = avg_leg_spread(rot_parts)
        spread_avg.append(leg_spread)


    #For each average in the gaits list, average the swings and stands. Return a [x,y] tuple of total 
    #average swing and stand. 
    stands = [item[0] for item in gaits]
    swings = [item[1] for item in gaits]


    stand_tot  = [np.mean(stands), np.mean(swings)]




    return spread_avg, vel_avg, stand_tot

#Put in seperate legs like a 'percentage' of its entire swing. 


spread_0, vel_0, stand_0 = stat_analysis(zero_degrees)
spread_45, vel_45, stand_45 = stat_analysis(forty_five_degrees)
spread_90, vel_90, stand_90 = stat_analysis(ninety_degrees)


print(stand_0, stand_45, stand_90)

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
    gaits = [stand_0, stand_45, stand_90] 
    stands = [item[0] for item in gaits]
    swings = [item[1] for item in gaits]


    label_st = ["0 degrees", "45 degrees", "90 degrees"]
    ax3 = fig.add_subplot(2,2,3)
    a1 = ax3.barh(label_st, stands, color = 'white', edgecolor = 'black', hatch = 'x')
    a2 = ax3.barh(label_st, swings, left = stands, color = 'white', edgecolor = 'black', hatch = 'o')
    ax3.legend([a1,a2], ["Stand Phase", "Swing Phase"], title = "Phase of gait cycle", loc = "upper right")
    ax3.set_title("Percentage of gait cycle spent in stand or swing")
    ax3.set_xlim(-10,110)


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
left1_0, left2_0, left3_0, right1_0, right2_0, right3_0 = gait_phase_plotting(file0_4)
left1_45, left2_45, left3_45, right1_45, right2_45, right3_45 = gait_phase_plotting(file45_5)
left1_90, left2_90, left3_90, right1_90, right2_90, right3_90 = gait_phase_plotting(file90_3)


ax5 = fig2.add_subplot(2,2,1)
size0 = range(len(left1_0))
l1, = ax5.plot(size0, left1_0, '-', color = 'purple')
l2, = ax5.plot(size0, left2_0, '-', color = 'darkblue')
l3, = ax5.plot(size0, left3_0, '-', color = 'cornflowerblue')
r1, = ax5.plot(size0, [x + 18 for x in right1_0], '-', color = 'cyan')
r2, = ax5.plot(size0, [x + 25 for x in right2_0], '-', color = 'aquamarine')
r3, = ax5.plot(size0, [x + 25 for x in right3_0], '-', color = 'greenyellow')
ax5.legend([r3, l3, r2, l2, r1, l1], ["Right 3", "Left 3", "Right 2", "Left 2", "Right 1", "Left 1"], title = "Foot Label", loc = "lower right", ncol = 1, fancybox = True, shadow = True)
ax5.title.set_text('Foot Vertical Displacement vs Time (0 degrees)')
ax5.set_facecolor('grey')

### Plot phase for 45 degrees ###
ax6 = fig2.add_subplot(2,2,2)
size45 = range(len(left1_45))
ax6.plot(size45, left1_45, '-', color = 'purple')
ax6.plot(size45, left2_45, '-', color = 'darkblue')
ax6.plot(size45, [x + 30 for x in left3_45], '-', color = 'cornflowerblue')
ax6.plot(size45, [x + 25 for x in right1_45], '-', color = 'cyan')
ax6.plot(size45, [x + 50 for x in right2_45], '-', color = 'aquamarine')
ax6.plot(size45, [x + 80 for x in right3_45], '-', color = 'greenyellow')
ax6.legend([r3, l3, r2, l2, r1, l1], ["Right 3", "Left 3", "Right 2", "Left 2", "Right 1", "Left 1"], title = "Foot Label", loc = "lower right", ncol = 1, fancybox = True, shadow = True)
ax6.title.set_text('Foot Vertical Displacement vs Time (45 degrees)')
ax6.set_facecolor('grey')


### Plot phase for 90 degrees ###
ax7 = fig2.add_subplot(2,2,3)
size90 = range(len(left1_90))
ax7.plot(size90, left1_90, '-', color = 'purple')
ax7.plot(size90, left2_90, '-', color = 'darkblue')
ax7.plot(size90, [x + 40 for x in left3_90], '-', color = 'cornflowerblue')
ax7.plot(size90, [x + 30 for x in right1_90], '-', color = 'cyan')
ax7.plot(size90, [x + 50 for x in right2_90], '-', color = 'aquamarine')
ax7.plot(size90, [x + 80 for x in right3_90], '-', color = 'greenyellow')
ax7.legend([r3, l3, r2, l2, r1, l1], ["Right 3", "Left 3", "Right 2", "Left 2", "Right 1", "Left 1"], title = "Foot Label", loc = "lower right", ncol = 1, fancybox = True, shadow = True)
ax7.title.set_text('Foot Vertical Displacement vs Time (90 degrees)')
ax7.set_facecolor('grey')

plt.show()








