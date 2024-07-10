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


#For gait phase plotting we include these representative sample
file0_gait = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B3_0degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"

file45_gait = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B2_45degrees_straightDLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"

file90_gait = r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B3_90degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"


########### End of file collection ###########


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

        #Extract the time of the gait phase
        left1_time_l.append(leg_time(leg_velocity[0])[1])
        left2_time_l.append(leg_time(leg_velocity[1])[1])
        left3_time_l.append(leg_time(leg_velocity[2])[1])
        right1_time_l.append(leg_time(leg_velocity[3])[1])
        right2_time_l.append(leg_time(leg_velocity[4])[1])
        right3_time_l.append(leg_time(leg_velocity[5])[1])


        #Rotate Parts and calculate the average leg spread
        rot_parts = part_rotation(parts)
        spread_parts.append(rot_parts)


    left1_tot = [np.mean([item[1] for item in left1_gait]), np.mean([item[0] for item in left1_gait])]
    left2_tot = [np.mean([item[0] for item in left2_gait]), np.mean([item[1] for item in left2_gait])]
    left3_tot = [np.mean([item[1] for item in left3_gait]), np.mean([item[0] for item in left3_gait])]
    right1_tot = [np.mean([item[0] for item in right1_gait]), np.mean([item[1] for item in right1_gait])]
    right2_tot = [np.mean([item[1] for item in right2_gait]), np.mean([item[0] for item in right2_gait])]
    right3_tot = [np.mean([item[0] for item in right3_gait]), np.mean([item[1] for item in right3_gait])]

    stand_tot = [left1_tot, left2_tot, left3_tot, right1_tot, right2_tot, right3_tot]

    leg_times = [left1_time_l, left2_time_l, left3_time_l, 
                 right1_time_l, right2_time_l, right3_time_l]

    time_list = left1_time_l + left2_time_l + left3_time_l + right1_time_l + right2_time_l + right3_time_l
    avg_time = np.mean(time_list)
    

    return spread_parts, vel_avg, stand_tot, time_list



#### Here we call the statistical analysis function once per angle and get a spread, 
#### velocity and stand/swing output. 
spread_0, vel_0, stand_0, time_0 = stat_analysis(zero_degrees)
spread_45, vel_45, stand_45, time_45 = stat_analysis(forty_five_degrees)
spread_90, vel_90, stand_90, time_90 = stat_analysis(ninety_degrees)


##### Now we have ended the analysis and head into the plotting #####

#Plotting the spread function 
def spread_plot(fig):
    ax1 = fig.add_subplot(2,2,1)

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
    ax1.set_title("Leg Spread 0 degrees")


    ax2 = fig.add_subplot(2,2,2)
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
    ax2.set_title("Leg Spread 45 degrees")


    ax3 = fig.add_subplot(2,2,3)
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
    ax3.set_title("Leg Spread 90 degrees")




###### Box and whisker plot for velocity  ######
def velocity_plot(fig):
    vels = [vel_0, vel_45, vel_90]
    devs = [stat.stdev(x) for x in vels]
    mean = [stat.mean(x) for x in vels]

    #### t-test (Welchs https://en.wikipedia.org/wiki/Welch%27s_t-test) ####
    t_zero_fotry = (mean[0] - mean[1])/np.sqrt((devs[0]/np.sqrt(len(zero_degrees)))**2 + (devs[1]/np.sqrt(len(forty_five_degrees)))**2)
    t_zero_ninety = (mean[0] - mean[2])/np.sqrt((devs[0]/np.sqrt(len(zero_degrees)))**2 + (devs[2]/np.sqrt(len(ninety_degrees)))**2)
    t_forty_ninety = (mean[1] - mean[2])/np.sqrt((devs[1]/np.sqrt(len(forty_five_degrees)))**2 + (devs[2]/np.sqrt(len(ninety_degrees)))**2)
    print('The Students t-test between 0 and 45 degrees is', t_zero_fotry)
    print('The Students t-test between 0 and 90 degrees is', t_zero_ninety)
    print('The Students t-test between 45 and 90 degrees is', t_forty_ninety)


    label_v = ["0 degrees", "45 degrees", "90 degrees"]
    ax8 = fig.add_subplot(2,2,4)
    ax8.set_xticklabels(label_v)
    ax8.boxplot(vels)
    ax8.set_title("Velocity vs Angle of Inclination")
    ax8.set_ylabel("Velocity")


### Plotting of times of leg swings ###
def time_plot(fig):
    label = ["0 degrees", "45 degrees", "90 degrees"]
    times = [time_0, time_45, time_90]
    ax9 = fig.add_subplot(2,2,4)
    ax9.set_xticklabels(label)
    ax9.boxplot(times)
    ax9.set_ylabel("Gait Cycle Time (s)")
    ax9.set_title("Gait Cycle Time (s) vs Angle of inclination")
    
    print("Average gait cycle time for 0 degrees: ", np.mean(time_0), "\n", "Average gait cycle time for 45 degrees: ", np.mean(time_45), 
          "\n", "Average gait cycle time for 90 degrees: ", np.mean(time_90))


###### Stand plot ######
def stand_plot(fig):  
    zero_stands = [item[0] for item in stand_0]
    zero_swings = [item[1] for item in stand_0]

    forty_stands = [item[0] for item in stand_45]
    forty_swings = [item[1] for item in stand_45]

    ninety_stands = [item[0] for item in stand_90]
    ninety_swings = [item[1] for item in stand_90]



    ax3 = fig.add_subplot(2,2,1)
    a1 = ax3.barh("Left 1", zero_stands[0], color = 'white', edgecolor = 'black')
    a2 = ax3.barh("Left 1", zero_swings[0], left = zero_stands[0], color = 'black', edgecolor = 'black')
    ax3.barh("Left 2", zero_stands[1], color = 'black', edgecolor = 'black')
    ax3.barh("Left 2", zero_swings[1], left = zero_stands[1], color = 'white', edgecolor = 'black')
    ax3.barh("Left 3", zero_stands[2], color = 'white', edgecolor = 'black')
    ax3.barh("Left 3", zero_swings[2], left = zero_stands[2], color = 'black', edgecolor = 'black')
    ax3.barh("Right 1", zero_stands[3], color = 'black', edgecolor = 'black')
    ax3.barh("Right 1", zero_swings[3], left = zero_stands[3], color = 'white', edgecolor = 'black')
    ax3.barh("Right 2", zero_stands[4], color = 'white', edgecolor = 'black')
    ax3.barh("Right 2", zero_swings[4], left = zero_stands[4], color = 'black', edgecolor = 'black')
    ax3.barh("Right 3", zero_stands[5], color = 'black', edgecolor = 'black')
    ax3.barh("Right 3", zero_swings[5], left = zero_stands[5], color = 'white', edgecolor = 'black')
    #ax3.legend([a1,a2], ["Swing Phase", "Stand Phase"], title = "Phase of gait cycle", loc = "upper right")
    ax3.set_title("Gait Cycle 0 degrees")
    ax3.set_xlabel('Percentage of Total Gait Cycle (%)')
    ax3.set_xlim(-10,110)
    
    ax4 = fig.add_subplot(2,2,2)
    a3 = ax4.barh("Left 1", forty_stands[0], color = 'white', edgecolor = 'black')
    a4 = ax4.barh("Left 1", forty_swings[0], left = forty_stands[0], color = 'black', edgecolor = 'black')
    ax4.barh("Left 2", forty_stands[1], color = 'black', edgecolor = 'black')
    ax4.barh("Left 2", forty_swings[1], left = forty_stands[1], color = 'white', edgecolor = 'black')
    ax4.barh("Left 3", forty_stands[2], color = 'white', edgecolor = 'black')
    ax4.barh("Left 3", forty_swings[2], left = forty_stands[2], color = 'black', edgecolor = 'black')
    ax4.barh("Right 1", forty_stands[3], color = 'black', edgecolor = 'black')
    ax4.barh("Right 1", forty_swings[3], left = forty_stands[3], color = 'white', edgecolor = 'black')
    ax4.barh("Right 2", forty_stands[4], color = 'white', edgecolor = 'black')
    ax4.barh("Right 2", forty_swings[4], left = forty_stands[4], color = 'black', edgecolor = 'black')
    ax4.barh("Right 3", forty_stands[5], color = 'black', edgecolor = 'black')
    ax4.barh("Right 3", forty_swings[5], left = forty_stands[5], color = 'white', edgecolor = 'black')
    ax4.set_title("Gait cycle 45 degrees")
    ax4.set_xlabel('Percentage of Total Gait Cycle (%)')
    #ax4.legend([a3,a4], ["Swing Phase", "Stand Phase"], title = "Phase of gait cycle", loc = "upper right")
    ax4.set_xlim(-10,110)

    ax5 = fig.add_subplot(2,2,3)
    a5 = ax5.barh("Left 1", ninety_stands[0], color = 'white', edgecolor = 'black')
    a6 = ax5.barh("Left 1", ninety_swings[0], left = ninety_stands[0], color = 'black', edgecolor = 'black')
    ax5.barh("Left 2", ninety_stands[1], color = 'black', edgecolor = 'black')
    ax5.barh("Left 2", ninety_swings[1], left = ninety_stands[1], color = 'white', edgecolor = 'black')
    ax5.barh("Left 3",ninety_stands[2], color = 'white', edgecolor = 'black')
    ax5.barh("Left 3", ninety_swings[2], left = ninety_stands[2], color = 'black', edgecolor = 'black')
    ax5.barh("Right 1", ninety_stands[3], color = 'black', edgecolor = 'black')
    ax5.barh("Right 1", ninety_swings[3], left = ninety_stands[3], color = 'white', edgecolor = 'black')
    ax5.barh("Right 2", ninety_stands[4], color = 'white', edgecolor = 'black')
    ax5.barh("Right 2", ninety_swings[4], left = ninety_stands[4], color = 'black', edgecolor = 'black')
    ax5.barh("Right 3", ninety_stands[5], color = 'black', edgecolor = 'black')
    ax5.barh("Right 3", ninety_swings[5], left = ninety_stands[5], color = 'white', edgecolor = 'black')
    ax5.set_title("Gait cycle 90 degrees")
    ax5.set_xlabel('Percentage of Total Gait Cycle (%)')
    #ax5.legend([a5,a6], ["Swing Phase", "Stand Phase"], title = "Phase of gait cycle", loc = "upper right")
    ax5.set_xlim(-10,110)


    


### Declare the first figure and run all plotting functions ###
fig1 = plt.figure()
spread_plot(fig1)
velocity_plot(fig1)
plt.show()

fig3 = plt.figure()
stand_plot(fig3)
time_plot(fig3)
plt.show()



"""Here we plot the gait phase. We will only choose representative samples for the gait phase plotting"""

### Plotting of gait phase ###
fig2 = plt.figure()

### Make gait phase plotting data from specific files ###
left1_0, left2_0, left3_0, right1_0, right2_0, right3_0 = gait_phase_plotting(file0_gait)
left1_45, left2_45, left3_45, right1_45, right2_45, right3_45 = gait_phase_plotting(file45_gait)
left1_90, left2_90, left3_90, right1_90, right2_90, right3_90 = gait_phase_plotting(file90_gait)


ax5 = fig2.add_subplot(2,2,1)
size0 = range(len(left1_0))
ax5.plot(size0, left1_0, '-', color = 'cyan')
ax5.annotate('left 1', xy = (size0[0], left1_0[0]), xytext=(-12 + size0[0], left1_0[0]), color = 'cyan')
ax5.plot(size0, left2_0, '-', color = 'darkorange')
ax5.annotate('left 2', xy = (size0[0], left2_0[0]), xytext=(-12+size0[0], left2_0[0]), color = 'darkorange')
ax5.plot(size0, left3_0, '-', color = 'cornflowerblue')
ax5.annotate('left 3', xy = (size0[0], left3_0[0]), xytext=(-12+size0[0], left3_0[0]), color = 'cornflowerblue')
ax5.plot(size0, [x + 18 for x in right1_0], '-', color = 'purple')
ax5.annotate('right 1', xy = (size0[0], right1_0[0]), xytext=(-12+size0[0], 18+ right1_0[0]), color = 'purple')
ax5.plot(size0, [x + 25 for x in right2_0], '-', color = 'aquamarine')
ax5.annotate('right 2', xy = (size0[0], right2_0[0]), xytext=(-14+size0[0], 25+ right2_0[0]), color = 'aquamarine')
ax5.plot(size0, [x + 25 for x in right3_0], '-', color = 'greenyellow')
ax5.annotate('right 3', xy = (size0[0], right3_0[0]), xytext=(-12+size0[0], 25+ right3_0[0]), color = 'greenyellow')
ax5.title.set_text('Foot Vertical Displacement vs Time (0 degrees)')
ax5.set_xlim(-15,165)
ax5.set_xlabel('Time (frames)')
ax5.set_ylabel('Position)')
ax5.set_facecolor('black')

### Plot phase for 45 degrees ###
ax6 = fig2.add_subplot(2,2,2)
size45 = range(len(left1_45))
ax6.plot(size45, left1_45, '-', color = 'cyan')
ax6.annotate('left 1', xy = (size45[0], left1_45[0]), xytext=(-12 + size45[0], -5+left1_45[0]), color = 'cyan')
ax6.plot(size45, left2_45, '-', color = 'darkorange')
ax6.annotate('left 2', xy = (size45[0], left2_45[0]), xytext=(-12 + size45[0],  left2_45[0]), color = 'darkorange')
ax6.plot(size45, [x + 30 for x in left3_45], '-', color = 'cornflowerblue')
ax6.annotate('left 3', xy = (size45[0], left3_45[0]), xytext=(-12 + size45[0], 30+left3_45[0]), color = 'cornflowerblue')
ax6.plot(size45, [x + 25 for x in right1_45], '-', color = 'purple')
ax6.annotate('right 1', xy = (size45[0], right1_45[0]), xytext=(-12 + size45[0], 30+right1_45[0]), color = 'purple')
ax6.plot(size45, [x + 50 for x in right2_45], '-', color = 'aquamarine')
ax6.annotate('right 2', xy = (size45[0], right2_45[0]), xytext=(-12 +size45[0], 42+right2_45[0]), color = 'aquamarine')
ax6.plot(size45, [x + 80 for x in right3_45], '-', color = 'greenyellow')
ax6.annotate('right 3', xy = (size45[0], right3_45[0]), xytext=(-12 + size45[0], 80+right3_45[0]), color = 'greenyellow')
ax6.title.set_text('Foot Vertical Displacement vs Time (45 degrees)')
ax6.set_xlabel('Time (frames)')
ax6.set_ylabel('Position')
ax6.set_xlim(-15,165)
ax6.set_facecolor('black')


### Plot phase for 90 degrees ###
ax7 = fig2.add_subplot(2,2,3)
size90 = range(len(left1_90))
ax7.plot(size90, left1_90, '-', color = 'cyan')
ax7.annotate('left 1', xy = (size90[0], left1_90[0]), xytext=(-12 + size90[0], -5+left1_90[0]), color = 'cyan')
ax7.plot(size90, left2_90, '-', color = 'darkorange')
ax7.annotate('left 2', xy = (size90[0], left2_90[0]), xytext=(-12 + size90[0], left2_90[0]), color = 'darkorange')
ax7.plot(size90, [x + 40 for x in left3_90], '-', color = 'cornflowerblue')
ax7.annotate('left 3', xy = (size90[0], left3_90[0]), xytext=(-12 + size90[0],  44+left3_90[0]), color = 'cornflowerblue')
ax7.plot(size90, [x + 30 for x in right1_90], '-', color = 'purple')
ax7.annotate('right 1', xy = (size90[0], right1_90[0]), xytext=(-12 + size90[0], 25+right1_90[0]), color = 'purple')
ax7.plot(size90, [x + 50 for x in right2_90], '-', color = 'aquamarine')
ax7.annotate('right 2', xy = (size90[0], right2_90[0]), xytext=(-12 + size90[0], 40+right2_90[0]), color = 'aquamarine')
ax7.plot(size90, [x + 80 for x in right3_90], '-', color = 'greenyellow')
ax7.annotate('right 3', xy = (size90[0], right3_90[0]), xytext=(-12 + size90[0], 82+right3_90[0]), color = 'greenyellow')
ax7.title.set_text('Foot Vertical Displacement vs Time (90 degrees)')
ax7.set_xlabel('Time (frames)')
ax7.set_ylabel('Position')
ax7.set_xlim(-15,165)
ax7.set_facecolor('black')

plt.show()








