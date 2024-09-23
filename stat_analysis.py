import csv 
import pandas as pd
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal
from analysis import *
import statistics as stat
import matplotlib.patches as mpatches
from os import listdir
import matplotlib.gridspec as gridspec

######## Here we import the files necessary for analysis, we also import the representative files for gait plotting ########

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

zero_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\0 degrees\\"+x 
             for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\0 degrees")]

forty_five_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\45 degrees\\"+x 
                      for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\45 degrees")]

ninety_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\90 degrees\\"+x 
                  for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\90 degrees")]

########### End of file collection ###########

#For gait phase plotting we include these representative sample
file0_gait = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B3_0degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"

file45_gait = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B2_45degrees_straightDLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"

file90_gait = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B3_90degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"

#file90_gait = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\90 degrees\movie20240530_B2_90d (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"



"""This is the statistical analysis function and will be called ONCE per angle. 
So, all files of a specific angle will be run through in the inner for loop. """

def stat_analysis(files):
    #Declare empty lists for averages of velocity, leg spread and gait timing
    vel_avg = []
    gaits = []
    spread_parts = []
    left1_gait = []
    left2_gait = []
    left3_gait = []
    right1_gait = []
    right2_gait = []
    right3_gait = []

    left1_time_l = []
    left2_time_l = []
    left3_time_l = []
    right1_time_l = []
    right2_time_l = []
    right3_time_l = []

    left1_time_s = []
    left2_time_s = []
    left3_time_s = []
    right1_time_s = []
    right2_time_s = []
    right3_time_s = []

    gait_cycle_time = []
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
        left1_gait.append(leg_time(leg_velocity[0])[0])
        left2_gait.append(leg_time(leg_velocity[1])[0])
        left3_gait.append(leg_time(leg_velocity[2])[0])
        right1_gait.append(leg_time(leg_velocity[3])[0])
        right2_gait.append(leg_time(leg_velocity[4])[0])
        right3_gait.append(leg_time(leg_velocity[5])[0])

        #Extract the time of the stand phase
        left1_time_l.append(leg_time(leg_velocity[0])[1])
        left2_time_l.append(leg_time(leg_velocity[1])[1])
        left3_time_l.append(leg_time(leg_velocity[2])[1])
        right1_time_l.append(leg_time(leg_velocity[3])[1])
        right2_time_l.append(leg_time(leg_velocity[4])[1])
        right3_time_l.append(leg_time(leg_velocity[5])[1])

        #Extract the time of the swing phase
        left1_time_s.append(leg_time(leg_velocity[0])[2])
        left2_time_s.append(leg_time(leg_velocity[1])[2])
        left3_time_s.append(leg_time(leg_velocity[2])[2])
        right1_time_s.append(leg_time(leg_velocity[3])[2])
        right2_time_s.append(leg_time(leg_velocity[4])[2])
        right3_time_s.append(leg_time(leg_velocity[5])[2])

        for i in range(6): 
            gait_cycle_time.append(leg_time(leg_velocity[i])[3])

        #Rotate Parts and calculate the average leg spread
        rot_parts = part_rotation(parts)
        spread_parts.append(rot_parts)

    print(vel_avg)
    left1_tot = [np.mean([item[1] for item in left1_gait]), np.mean([item[0] for item in left1_gait])]
    left2_tot = [np.mean([item[0] for item in left2_gait]), np.mean([item[1] for item in left2_gait])]
    left3_tot = [np.mean([item[1] for item in left3_gait]), np.mean([item[0] for item in left3_gait])]
    right1_tot = [np.mean([item[0] for item in right1_gait]), np.mean([item[1] for item in right1_gait])]
    right2_tot = [np.mean([item[1] for item in right2_gait]), np.mean([item[0] for item in right2_gait])]
    right3_tot = [np.mean([item[0] for item in right3_gait]), np.mean([item[1] for item in right3_gait])]

    stand_tot = [left1_tot, left2_tot, left3_tot, right1_tot, right2_tot, right3_tot]

    leg_stand_times = [left1_time_l, [x for x in left2_time_l if x < 0.35], left3_time_l]

    leg_swing_times = [left1_time_s, left2_time_s, left3_time_s, right1_time_s, right2_time_s, right3_time_s]

    time_list = gait_cycle_time
    avg_time = np.mean(time_list)
    

    return spread_parts, vel_avg, stand_tot, time_list, leg_stand_times, leg_swing_times



#### Here we call the statistical analysis function once per angle and get a spread, 
#### velocity and stand/swing output. 
spread_0, vel_0, stand_0, time_0, times_0, swing_0 = stat_analysis(zero_degrees)
spread_45, vel_45, stand_45, time_45, times_45, swing_45 = stat_analysis(forty_five_degrees)
spread_90, vel_90, stand_90, time_90, times_90, swing_90 = stat_analysis(ninety_degrees)

print("For 0 degrees we see avg stand time is: ", np.median([item for sublist in times_0 for item in sublist]), "Average swing time is: ", np.median([item for sublist in swing_0 for item in sublist]))
print("For 45 degrees we see avg stand time is: ", np.median([item for sublist in times_45 for item in sublist]), "Average swing time is: ", np.median([item for sublist in swing_45 for item in sublist]))
print("For 90 degrees we see avg stand time is: ", np.median([item for sublist in times_90 for item in sublist]), "Average swing time is: ", np.median([item for sublist in swing_90 for item in sublist]))
##### Now we have ended the analysis and head into the plotting #####

#Plotting the spread function 
def spread_plot(fig):
    ax1 = fig.add_subplot(2,2,2)

    for file in spread_0:
        temp = [(a,b) for a,b in file[0] if a < 0]
        x,y = zip(*temp)
        ax1.scatter(x,y, marker = ".", color = 'blue')

        temp = [(a,b) for a,b in file[1] if a < 0]
        x,y = zip(*temp)
        ax1.scatter(x,y, marker = ".", color = 'red')

        temp = [(a,b) for a,b in file[2] if a < 0]
        x,y = zip(*temp)
        ax1.scatter(x,y, marker = ".", color = 'purple')

        temp = [(a,b) for a,b in file[3] if a > 0]
        x,y = zip(*temp)
        ax1.scatter(x,y, marker = ".", color = 'orange')

        temp = [(a,b) for a,b in file[4] if a > 0]
        x,y = zip(*temp)
        ax1.scatter(x,y, marker = ".", color = 'darkblue')

        temp = [(a,b) for a,b in file[5] if a > 0]
        x,y = zip(*temp)
        ax1.scatter(x,y, marker = ".", color = 'aquamarine')

        x,y = zip(*file[6])
        ax1.scatter(x,y, marker = ".", color = 'greenyellow')
        x,y = zip(*file[7])
        ax1.scatter(x,y, marker = ".", color = 'forestgreen')
        x,y = zip(*file[8])
        ax1.scatter(x,y, marker = ".", color = 'mediumorchid')
        ax1.arrow(0,0,0,100, width = 1, color = 'black')
        ax1.annotate('Head', xy = (-3,105), xytext=(-3,105), color = 'black')
        ax1.annotate('Bottom', xy = (0,0), xytext=(-5,-5), color = 'black')
        ax1.set_xlim(-80,80)
        ax1.set_ylim(-35,140)
        ax1.set_title("Leg Spread 0 degrees", fontsize = 15)
        ax1.text(-0.1, 1.1, '(B)', transform=ax1.transAxes, size=16, weight='bold')

    ax2 = fig.add_subplot(2,2,3)
    for file in spread_45:
        temp = [(a,b) for a,b in file[0] if a < 0]
        x,y = zip(*temp)
        ax2.scatter(x,y, marker = ".", color = 'blue')

        temp = [(a,b) for a,b in file[1] if a < 0]
        x,y = zip(*temp)
        ax2.scatter(x,y, marker = ".", color = 'red')

        temp = [(a,b) for a,b in file[2] if a < 0]
        x,y = zip(*temp)
        ax2.scatter(x,y, marker = ".", color = 'purple')

        temp = [(a,b) for a,b in file[3] if a > 0]
        x,y = zip(*temp)
        ax2.scatter(x,y, marker = ".", color = 'orange')

        temp = [(a,b) for a,b in file[4] if a > 0]
        x,y = zip(*temp)
        ax2.scatter(x,y, marker = ".", color = 'darkblue')

        temp = [(a,b) for a,b in file[5] if a > 0]
        x,y = zip(*temp)
        ax2.scatter(x,y, marker = ".", color = 'aquamarine')

        x,y = zip(*file[6])
        ax2.scatter(x,y, marker = ".", color = 'greenyellow')
        x,y = zip(*file[7])
        ax2.scatter(x,y, marker = ".", color = 'forestgreen')
        x,y = zip(*file[8])
        ax2.scatter(x,y,marker = ".",  color = 'mediumorchid')
        ax2.arrow(0,0,0,100, width = 1, color = 'black')
        ax2.annotate('Head', xy = (-3,105), xytext=(-3,105), color = 'black')
        ax2.annotate('Bottom', xy = (0,0), xytext=(-5,-5), color = 'black')
        ax2.set_xlim(-80,80)
        ax2.set_ylim(-35,140)
    ax2.set_title("Leg Spread 45 degrees", fontsize = 15)
    ax2.text(-0.1, 1.1, '(C)', transform=ax2.transAxes, size=16, weight='bold')

    ax3 = fig.add_subplot(2,2,4)
    for file in spread_90:
        temp = [(a,b) for a,b in file[0] if a < 0]
        x,y = zip(*temp)
        ax3.scatter(x,y, marker = ".", color = 'blue')

        temp = [(a,b) for a,b in file[1] if a < 0]
        x,y = zip(*temp)
        ax3.scatter(x,y, marker = ".", color = 'red')

        temp = [(a,b) for a,b in file[2] if a < 0]
        x,y = zip(*temp)
        ax3.scatter(x,y, marker = ".", color = 'purple')

        temp = [(a,b) for a,b in file[3] if a > 0]
        x,y = zip(*temp)
        ax3.scatter(x,y, marker = ".", color = 'orange')

        temp = [(a,b) for a,b in file[4] if a > 0]
        x,y = zip(*temp)
        ax3.scatter(x,y, marker = ".", color = 'darkblue')

        temp = [(a,b) for a,b in file[5] if a > 0]
        x,y = zip(*temp)
        ax3.scatter(x,y, marker = ".", color = 'aquamarine')
        x,y = zip(*file[6])
        ax3.scatter(x,y,marker = ".", color = 'greenyellow')
        x,y = zip(*file[7])
        ax3.scatter(x,y,marker = ".", color = 'forestgreen')
        x,y = zip(*file[8])
        ax3.scatter(x,y,marker = ".", color = 'mediumorchid')
        ax3.arrow(0,0,0,100, width = 1, color = 'black')
        ax3.annotate('Head', xy = (-3,105), xytext=(-3,105), color = 'black')
        ax3.annotate('Bottom', xy = (0,0), xytext=(-5,-5), color = 'black')
        ax3.set_xlim(-80,80)
        ax3.set_ylim(-35,140)
    ax3.set_title("Leg Spread 90 degrees", fontsize = 15)
    ax3.text(-0.1, 1.1, '(D)', transform=ax3.transAxes, size=16, weight='bold')



###### Box and whisker plot for velocity  ######
def velocity_plot(fig):
    vels = [vel_0, vel_45, vel_90]
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

    ax8 = fig.add_subplot(2,2,1)
    #ax8.set_xticklabels(["0 degrees", "45 degrees", "90 degrees"])
    ax8.boxplot(vels)
    ax8.set_title("Average Velocity vs Angle of Inclination", fontsize = 15 )
    ax8.set_ylabel("Avg Velocity (body lengths / second)", fontsize = 15)
    label = ["0 degrees", "45 degrees", "90 degrees"]
    ax8.set_xticklabels(label, fontsize = 15)
    print('\n')
    print('**** Average Velocities *****')
    zero_mean = np.mean(vel_0)
    forty_mean = np.mean(vel_45)
    ninety_mean = np.mean(vel_90)
    print('Average velocity at 0 degrees: ', zero_mean)
    print('Average velocity at 45 degrees: ', forty_mean)
    print('Average velocity at 90 degrees: ', ninety_mean)

    ax8.text(-0.1, 1.1, '(A)', transform=ax8.transAxes, size=16, weight='bold')


### Plotting of times of leg swings ###
def time_plot(fig):
    label = ["0", "45", "90"]
    times = [time_0, time_45, time_90]
    ax9 = fig.add_subplot(2,2,4)
    ax9.set_xticklabels(label, fontsize = 13)
    ax9.boxplot(times)
    ax9.set_ylabel("Gait Cycle Time (s)", fontsize = 15)
    ax9.set_xlabel("Angle of Inclination (degrees)", fontsize = 15)
    ax9.set_title("Gait Cycle Time (s) vs Angle of inclination", fontsize = 18)
    
    ax9.text(-0.1, 1.1, '(D)', transform=ax9.transAxes, size=16, weight='bold')
    print("Average gait cycle time for 0 degrees: ", np.mean(time_0), "\n", "Average gait cycle time for 45 degrees: ", np.mean(time_45), 
          "\n", "Average gait cycle time for 90 degrees: ", np.mean(time_90))


### Time Plot with individual legs ###
def leg_time_plot(fig):
    feet_label = ["Hind", "Middle", "Front"]
    ax9 = fig.add_subplot(1,3,1)
    ax10 = fig.add_subplot(1,3,2)
    ax11 = fig.add_subplot(1,3,3)

# Customize the boxplot appearance
    boxprops = dict(color='black', linewidth=2)  # Grey box, thicker lines
    medianprops = dict(color='cornflowerblue', linewidth=2)  # Bold median line
    whiskerprops = dict(color='black', linewidth=2)  # Bold whiskers
    capprops = dict(color='black', linewidth=2)  # Bold caps

    ax9.set_xticklabels(feet_label, fontsize = 15)
    ax9.boxplot(times_0, patch_artist = True,
                 boxprops=dict(facecolor='lightgrey', color='black', linewidth=2), medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    ax9.set_title("Gait Stand time at 0 degrees", fontsize = 20)
    ax9.set_ylim(0,0.7)
    ax9.set_ylabel("Gait Stand Time (seconds)", fontsize = 17)
    ax9.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    ax9.text(-0.1, 1.05, '(A)', transform=ax9.transAxes, size=16, weight='bold')
    ax9.tick_params(axis = 'y', labelsize = 15)
    
    ax10.set_xticklabels(feet_label, fontsize = 15)
    ax10.boxplot(times_45, patch_artist = True,
                 boxprops=dict(facecolor='lightgrey', color='black', linewidth=2), medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    ax10.set_title("Gait Stand time at 45 degrees", fontsize = 20)
    ax10.set_ylim(0,0.7)
    ax10.set_ylabel("Gait Stand Time (seconds)", fontsize = 17)
    ax10.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    ax10.text(-0.1, 1.05, '(B)', transform=ax10.transAxes, size=16, weight='bold')
    ax10.tick_params(axis = 'y', labelsize = 15)
    
    ax11.set_xticklabels(feet_label, fontsize = 15)
    ax11.boxplot(times_90, patch_artist = True,
                 boxprops=dict(facecolor='lightgrey', color='black', linewidth=2), medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    ax11.set_title("Gait Stand time at 90 degrees", fontsize = 20)
    ax11.set_ylim(0,0.7)
    ax11.set_ylabel("Gait Stand Time (seconds)", fontsize = 17)
    ax11.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    ax11.text(-0.1, 1.05, '(C)', transform=ax11.transAxes, size=16, weight='bold')
    ax11.tick_params(axis = 'y', labelsize = 15)


###### Stand plot ######
def stand_plot(fig):  
    zero_stands = [item[0] for item in stand_0]
    zero_swings = [item[1] for item in stand_0]

    forty_stands = [item[0] for item in stand_45]
    forty_swings = [item[1] for item in stand_45]

    ninety_stands = [item[0] for item in stand_90]
    ninety_swings = [item[1] for item in stand_90]



    ax3 = fig.add_subplot(2,2,1)
    ax3.barh("Left 1", zero_swings[0], color = 'black', edgecolor = 'black')
    ax3.barh("Left 1", zero_stands[0], left = zero_swings[0], color = 'white', edgecolor = 'black')
    ax3.barh("Left 2", zero_swings[1], color = 'white', edgecolor = 'black')
    ax3.barh("Left 2", zero_stands[1], left = zero_swings[1], color = 'black', edgecolor = 'black')
    ax3.barh("Left 3", zero_swings[2], color = 'black', edgecolor = 'black')
    ax3.barh("Left 3", zero_stands[2], left = zero_swings[2], color = 'white', edgecolor = 'black')
    ax3.barh("Right 1", zero_swings[3], color = 'white', edgecolor = 'black')
    ax3.barh("Right 1", zero_stands[3], left = zero_swings[3], color = 'black', edgecolor = 'black')
    ax3.barh("Right 2", zero_swings[4], color = 'black', edgecolor = 'black')
    ax3.barh("Right 2", zero_stands[4], left = zero_swings[4], color = 'white', edgecolor = 'black')
    ax3.barh("Right 3", zero_swings[5], color = 'white', edgecolor = 'black')
    ax3.barh("Right 3", zero_stands[5], left = zero_swings[5], color = 'black', edgecolor = 'black')
    black_patch = mpatches.Patch(color='black', label='Swing')
    white_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Stance')
    # Add the legend to the plot
    ax3.legend(handles=[black_patch, white_patch], loc='lower right')
    
    ax3.set_title("Gait Cycle 0 degrees", fontsize = 18)
    ax3.set_xlabel('Percentage of Total Gait Cycle (%)', fontsize = 13)
    ax3.set_xlim(-10,110)
    ax3.tick_params(axis='y', labelsize=13)
    ax3.tick_params(axis = 'x', labelsize = 12)

    ax4 = fig.add_subplot(2,2,2)
    ax4.barh("Left 1", forty_stands[0], color = 'black', edgecolor = 'black')
    ax4.barh("Left 1", forty_swings[0], left = forty_stands[0], color = 'white', edgecolor = 'black')
    ax4.barh("Left 2", forty_stands[1], color = 'white', edgecolor = 'black')
    ax4.barh("Left 2", forty_swings[1], left = forty_stands[1], color = 'black', edgecolor = 'black')
    ax4.barh("Left 3", forty_stands[2], color = 'black', edgecolor = 'black')
    ax4.barh("Left 3", forty_swings[2], left = forty_stands[2], color = 'white', edgecolor = 'black')
    ax4.barh("Right 1", forty_stands[3], color = 'white', edgecolor = 'black')
    ax4.barh("Right 1", forty_swings[3], left = forty_stands[3], color = 'black', edgecolor = 'black')
    ax4.barh("Right 2", forty_stands[4], color = 'black', edgecolor = 'black')
    ax4.barh("Right 2", forty_swings[4], left = forty_stands[4], color = 'white', edgecolor = 'black')
    ax4.barh("Right 3", forty_stands[5], color = 'white', edgecolor = 'black')
    ax4.barh("Right 3", forty_swings[5], left = forty_stands[5], color = 'black', edgecolor = 'black')
    ax4.set_title("Gait cycle 45 degrees", fontsize = 18)
    ax4.set_xlabel('Percentage of Total Gait Cycle (%)', fontsize = 13)
    #ax4.legend([a3,a4], ["Swing Phase", "Stand Phase"], title = "Phase of gait cycle", loc = "upper right")
    ax4.set_xlim(-10,110)
    black_patch = mpatches.Patch(color='black', label='Swing')
    white_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Stance')
    # Add the legend to the plot
    ax4.legend(handles=[black_patch, white_patch], loc='lower right')
    ax4.tick_params(axis='y', labelsize=13)    
    ax4.tick_params(axis = 'x', labelsize = 12)

    ax5 = fig.add_subplot(2,2,3)
    a5 = ax5.barh("Left 1", ninety_stands[0], color = 'black', edgecolor = 'black')
    a6 = ax5.barh("Left 1", ninety_swings[0], left = ninety_stands[0], color = 'white', edgecolor = 'black')
    ax5.barh("Left 2", ninety_stands[1], color = 'white', edgecolor = 'black')
    ax5.barh("Left 2", ninety_swings[1], left = ninety_stands[1], color = 'black', edgecolor = 'black')
    ax5.barh("Left 3",ninety_stands[2], color = 'black', edgecolor = 'black')
    ax5.barh("Left 3", ninety_swings[2], left = ninety_stands[2], color = 'white', edgecolor = 'black')
    ax5.barh("Right 1", ninety_stands[3], color = 'white', edgecolor = 'black')
    ax5.barh("Right 1", ninety_swings[3], left = ninety_stands[3], color = 'black', edgecolor = 'black')
    ax5.barh("Right 2", ninety_stands[4], color = 'black', edgecolor = 'black')
    ax5.barh("Right 2", ninety_swings[4], left = ninety_stands[4], color = 'white', edgecolor = 'black')
    ax5.barh("Right 3", ninety_stands[5], color = 'white', edgecolor = 'black')
    ax5.barh("Right 3", ninety_swings[5], left = ninety_stands[5], color = 'black', edgecolor = 'black')
    ax5.set_title("Gait cycle 90 degrees", fontsize = 18)
    ax5.set_xlabel('Percentage of Total Gait Cycle (%)', fontsize = 13)
    #ax5.legend([a5,a6], ["Swing Phase", "Stand Phase"], title = "Phase of gait cycle", loc = "upper right")
    ax5.set_xlim(-10,110)
    black_patch = mpatches.Patch(color='black', label='Swing')
    white_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Stance')
    # Add the legend to the plot
    ax5.legend(handles=[black_patch, white_patch], loc='lower right')
    ax5.tick_params(axis='y', labelsize=13)
    ax5.tick_params(axis = 'x', labelsize = 12)

        # Add text labels for each subplot
    ax3.text(-0.1, 1.1, '(A)', transform=ax3.transAxes, size=16, weight='bold')
    ax4.text(-0.1, 1.1, '(B)', transform=ax4.transAxes, size=16, weight='bold')
    ax5.text(-0.1, 1.1, '(C)', transform=ax5.transAxes, size=16, weight='bold')



### Declare the first figure and run all plotting functions ###
fig1 = plt.figure()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25, hspace=0.4)
spread_plot(fig1)
velocity_plot(fig1)
plt.show()

fig3 = plt.figure()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25, hspace=0.4)
stand_plot(fig3)
time_plot(fig3)
plt.show()

fig4 = plt.figure()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25, hspace=0.4)
leg_time_plot(fig4)
plt.show()

"""Here we plot the gait phase. We will only choose representative samples for the gait phase plotting"""

### Plotting of gait phase ###
fig2 = plt.figure()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25, hspace=0.4)
### Make gait phase plotting data from specific files ###
left1_0, left2_0, left3_0, right1_0, right2_0, right3_0 = gait_phase_plotting(file0_gait)
left1_45, left2_45, left3_45, right1_45, right2_45, right3_45 = gait_phase_plotting(file45_gait)
left1_90, left2_90, left3_90, right1_90, right2_90, right3_90 = gait_phase_plotting(file90_gait)


ax5 = fig2.add_subplot(2,2,1)
size0 = range(len(left1_0))
ax5.plot(size0, left1_0, '-', color = 'cyan')
ax5.annotate('left 1', xy = (size0[0], left1_0[0]), xytext=(-14 + size0[0], left1_0[0]), color = 'cyan', fontsize = 16)
ax5.plot(size0, left2_0, '-', color = 'darkorange')
ax5.annotate('left 2', xy = (size0[0], left2_0[0]), xytext=(-14+size0[0], left2_0[0]), color = 'darkorange', fontsize = 16)
ax5.plot(size0, left3_0, '-', color = 'cornflowerblue')
ax5.annotate('left 3', xy = (size0[0], left3_0[0]), xytext=(-14+size0[0], left3_0[0]), color = 'cornflowerblue', fontsize = 16)
ax5.plot(size0, [x + 18 for x in right1_0], '-', color = 'purple')
ax5.annotate('right 1', xy = (size0[0], right1_0[0]), xytext=(-14+size0[0], 18+ right1_0[0]), color = 'purple', fontsize = 16)
ax5.plot(size0, [x + 25 for x in right2_0], '-', color = 'aquamarine')
ax5.annotate('right 2', xy = (size0[0], right2_0[0]), xytext=(-14+size0[0], 25+ right2_0[0]), color = 'aquamarine', fontsize = 16)
ax5.plot(size0, [x + 25 for x in right3_0], '-', color = 'greenyellow')
ax5.annotate('right 3', xy = (size0[0], right3_0[0]), xytext=(-14+size0[0], 25+ right3_0[0]), color = 'greenyellow', fontsize = 16)
ax5.set_title('Foot Vertical Displacement vs Time (0 degrees)', fontsize = 23)
ax5.set_xlim(-15,165)
ax5.set_xlabel('Time (ms)', fontsize = 18)
ax5.set_ylabel('Position', fontsize = 18)
ax5.set_facecolor('dimgray')
ax5.text(-0.1, 1.1, '(A)', transform=ax5.transAxes, size=16, weight='bold')

### Plot phase for 45 degrees ###
ax6 = fig2.add_subplot(2,2,2)
size45 = range(len(left1_45))
ax6.plot(size45, left1_45, '-', color = 'cyan')
ax6.annotate('left 1', xy = (size45[0], left1_45[0]), xytext=(-14 + size45[0], -5+left1_45[0]), color = 'cyan', fontsize = 16)
ax6.plot(size45, left2_45, '-', color = 'darkorange')
ax6.annotate('left 2', xy = (size45[0], left2_45[0]), xytext=(-14 + size45[0],  left2_45[0]), color = 'darkorange', fontsize = 16)
ax6.plot(size45, [x + 30 for x in left3_45], '-', color = 'cornflowerblue')
ax6.annotate('left 3', xy = (size45[0], left3_45[0]), xytext=(-14 + size45[0], 30+left3_45[0]), color = 'cornflowerblue', fontsize = 16)
ax6.plot(size45, [x + 25 for x in right1_45], '-', color = 'purple')
ax6.annotate('right 1', xy = (size45[0], right1_45[0]), xytext=(-14 + size45[0], 30+right1_45[0]), color = 'purple', fontsize = 16)
ax6.plot(size45, [x + 50 for x in right2_45], '-', color = 'aquamarine')
ax6.annotate('right 2', xy = (size45[0], right2_45[0]), xytext=(-14 +size45[0], 42+right2_45[0]), color = 'aquamarine', fontsize = 16)
ax6.plot(size45, [x + 80 for x in right3_45], '-', color = 'greenyellow')
ax6.annotate('right 3', xy = (size45[0], right3_45[0]), xytext=(-14 + size45[0], 80+right3_45[0]), color = 'greenyellow', fontsize = 16)
ax6.set_title('Foot Vertical Displacement vs Time (45 degrees)', fontsize = 23)
ax6.set_xlabel('Time (ms)', fontsize = 18)
ax6.set_ylabel('Position', fontsize = 18)
ax6.set_xlim(-15,165)
ax6.set_facecolor('dimgray')
ax6.text(-0.1, 1.1, '(B)', transform=ax6.transAxes, size=16, weight='bold')

### Plot phase for 90 degrees ###
ax7 = fig2.add_subplot(2,2,3)
size90 = range(len(left1_90))
ax7.plot(size90, left1_90, '-', color = 'cyan')
ax7.annotate('left 1', xy = (size90[0], left1_90[0]), xytext=(-14 + size90[0], -5+left1_90[0]), color = 'cyan', fontsize = 16)
ax7.plot(size90, left2_90, '-', color = 'darkorange')
ax7.annotate('left 2', xy = (size90[0], left2_90[0]), xytext=(-14 + size90[0], left2_90[0]), color = 'darkorange', fontsize = 16)
ax7.plot(size90, [x + 40 for x in left3_90], '-', color = 'cornflowerblue')
ax7.annotate('left 3', xy = (size90[0], left3_90[0]), xytext=(-14 + size90[0],  44+left3_90[0]), color = 'cornflowerblue', fontsize = 16)
ax7.plot(size90, [x + 30 for x in right1_90], '-', color = 'purple')
ax7.annotate('right 1', xy = (size90[0], right1_90[0]), xytext=(-14 + size90[0], 25+right1_90[0]), color = 'purple', fontsize = 16)
ax7.plot(size90, [x + 50 for x in right2_90], '-', color = 'aquamarine')
ax7.annotate('right 2', xy = (size90[0], right2_90[0]), xytext=(-14 + size90[0], 40+right2_90[0]), color = 'aquamarine', fontsize = 16)
ax7.plot(size90, [x + 80 for x in right3_90], '-', color = 'greenyellow')
ax7.annotate('right 3', xy = (size90[0], right3_90[0]), xytext=(-14 + size90[0], 82+right3_90[0]), color = 'greenyellow', fontsize = 16)
ax7.set_title('Foot Vertical Displacement vs Time (90 degrees)', fontsize = 23)
ax7.set_xlabel('Time (ms)', fontsize = 18)
ax7.set_ylabel('Position', fontsize = 18)
ax7.set_xlim(-15,165)
ax7.set_facecolor('dimgray')
ax7.text(-0.1, 1.1, '(C)', transform=ax7.transAxes, size=16, weight='bold')

plt.show()

fig5 = plt.figure()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25, hspace=0.4)
gait_diagram_0 = gait_diagram(file0_gait)
gait_diagram_45 = gait_diagram(file45_gait)
gait_diagram_90 = gait_diagram(file90_gait)
row_labels = ['left1', 'left2', 'left3', 'right1', 'right2', 'right3']
y_positions = range(len(gait_diagram_0))

thickness = 0.9
ax1 = fig5.add_subplot(3,1,1)
ax1.set_title('0 degrees gait diagram', fontsize = 24)
for i, leg_velocity in enumerate(gait_diagram_0):
# Loop over each element in the binary list
    for j, value in enumerate(leg_velocity):
        if value == 1:
            # Plot a bar if the value is 1
            ax1.barh(i, 1, left=j, height=thickness, color='black')

ax1.set_xlim(5,100)
ax1.set_yticks(y_positions)
ax1.set_yticklabels(row_labels, fontsize = 16)
black_patch = mpatches.Patch(color='black', label='Swing')
white_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Stance')
# Add the legend to the plot
ax1.legend(handles=[black_patch, white_patch], loc='lower right')
ax1.text(-0.06, 1.1, '(A)', transform=ax1.transAxes, size=16, weight='bold')

ax2 = fig5.add_subplot(3,1,2)
ax2.set_title('45 Degrees Gait Diagram', fontsize = 24)
for i, leg_velocity in enumerate(gait_diagram_45):
# Loop over each element in the binary list
    for j, value in enumerate(leg_velocity):
        if value == 1:
            # Plot a bar if the value is 1
            ax2.barh(i, 1, left=j, height=thickness, color='black')

ax2.set_xlim(5,100)
ax2.set_yticks(y_positions)
ax2.set_yticklabels(row_labels, fontsize = 16)
ax2.legend(handles=[black_patch, white_patch], loc='lower right')
ax2.text(-0.06, 1.1, '(B)', transform=ax2.transAxes, size=16, weight='bold')

ax3 = fig5.add_subplot(3,1,3)
ax3.set_title('90 Degrees Gait Diagram', fontsize = 24)
for i, leg_velocity in enumerate(gait_diagram_90):
# Loop over each element in the binary list
    for j, value in enumerate(leg_velocity):
        if value == 1:
            # Plot a bar if the value is 1
            ax3.barh(i, 1, left=j, height=thickness, color='black')

ax3.set_xlim(5,100)
ax3.set_yticks(y_positions)
ax3.set_yticklabels(row_labels, fontsize = 16)
ax3.legend(handles=[black_patch, white_patch], loc='lower right')
ax3.text(-0.06, 1.1, '(C)', transform=ax3.transAxes, size=16, weight='bold')
ax1.tick_params(axis='x', labelsize=16)  # Adjust font size for x-axis numbers in the first subplot
ax2.tick_params(axis='x', labelsize=16)  # Adjust font size for x-axis numbers in the second subplot
ax3.tick_params(axis='x', labelsize=16)  # Adjust font size for x-axis numbers in the third subplot

plt.show()



