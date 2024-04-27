import csv 
import pandas as pd
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal


#Path to csv file to read in 


filename1 = r"C:\Users\lachl\OneDrive\Thesis\DeepLabCut\KinematicAnalysis\DLC_Attempt2_Sorted\0degrees\movie20240315_0degrees (2)DLC_resnet50_Attempt2DLCMar15shuffle1_100000.csv"

filename = r"C:\Users\lachl\OneDrive\Thesis\DeepLabCut\KinematicAnalysis\DLC_Attempt2_Sorted\0degrees\movie20240315_0degreesDLC_resnet50_Attempt2DLCMar15shuffle1_100000.csv"

file=  r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysis\movie20240425_B2_45degrees_straight (2)DLC_resnet50_KinematicAnalysisDLCApr24shuffle1_100000.csv"
files = [file]
#Read the csv file \


for file in files:
    df = pd.read_csv(file, skiprows=0)

    size = range(len(df.get('left1x').to_numpy()))

    #Read in left1
    left1x = df.get('left1x').to_numpy()
    left1y = df.get('left1y').to_numpy()
    left1 = []
    for i in size: 
        left1.append([left1x[i],left1y[i]])

    #Read in left2
    left2x = df.get('left2x').to_numpy()
    left2y = df.get('left2y').to_numpy()
    left2 = []
    for i in size: 
        left2.append([left2x[i], left2y[i]])

    #Read in left3
    left3x = df.get('left3x').to_numpy()
    left3y = df.get('left3y').to_numpy()
    left3 = []
    for i in size:
        left3.append([left3x[i], left3y[i]])

    #Read in left4
    try:
        left4x = df.get('left4x').to_numpy()
        left4y = df.get('left4y').to_numpy()
        left4 = []
        for i in size: 
            left4.append([left4x[i],left4y[i]])
    except AttributeError: 
        pass

    #Read in left5
    try:
        left5x = df.get('left5x').to_numpy()
        left5y = df.get('left5y').to_numpy()
        left5 = []
        for i in size: 
            left5.append([left5x[i],left5y[i]])
    except AttributeError:
        pass

    #Read in left6
    try:
        left6x = df.get('left6x').to_numpy()
        left6y = df.get('left5y').to_numpy()
        left6 = []
        for i in size: 
            left6.append([left6x[i],left6y[i]])  
    except AttributeError: 
        pass

    #Read in right4
    try:
        right4x = df.get('right4x').to_numpy()
        right4y = df.get('right4y').to_numpy()
        right4 =[]
        for i in size: 
            right4.append([right4x[i], right4y[i]])
    except AttributeError:
        pass

    #Read in right5
    try: 
        right5x = df.get('right5x').to_numpy()
        right5y = df.get('right5y').to_numpy()
        right5 =[]
        for i in size: 
            right5.append([right5x[i], right5y[i]])
    except AttributeError:
        pass


    #Read in right6
    try: 
        right6x = df.get('right6x').to_numpy()
        right6y = df.get('right6y').to_numpy()
        right6 =[]
        for i in size: 
            right6.append([right6x[i], right6y[i]])
    except AttributeError: 
        pass

    #Read in right1
    right1x = df.get('right1x').to_numpy()
    right1y = df.get('right1y').to_numpy()
    right1 =[]
    for i in size: 
        right1.append([right1x[i], right1y[i]])

    #Read in right2
    right2x = df.get('right2x').to_numpy()
    right2y = df.get('right2y').to_numpy()
    right2 =[]
    for i in size: 
        right2.append([right2x[i], right2y[i]])

    #Read in right3
    right3x = df.get('right3x').to_numpy()
    right3y = df.get('right3y').to_numpy()
    right3 =[]
    for i in size: 
        right3.append([right3x[i], right3y[i]])

    #Read in top
    topx = df.get('topx').to_numpy()
    topy = df.get('topy').to_numpy()
    top =[]
    for i in size: 
        top.append([topx[i], topy[i]])

    #Read in middle 
    middlex = df.get('middlex').to_numpy()
    middley = df.get('middley').to_numpy()
    middle = []
    for i in size: 
        middle.append([middlex[i], middley[i]])

    #Read in bottom 
    bottomx = df.get('bottomx').to_numpy()
    bottomy = df.get('bottomy').to_numpy()
    bottom = []
    for i in size: 
        bottom.append([bottomx[i], bottomy[i]])


    #Declare list of body parts for use in later code
    

    def moving_avg(part):
    
        partx = []
        party = []
        for i in part: 
            partx.append(i[0])
            party.append(i[1])
        
        partx = pd.Series(partx)
        party = pd.Series(party)
        
        partx = round(partx.ewm(alpha=0.2, adjust= False).mean(), 100)
        partx = partx.tolist()

        party = round(party.ewm(alpha=0.2, adjust= False).mean(), 100)
        party = party.tolist()


        smooth_data = []
        for i in range(len(partx)):
            smooth_data.append([partx[i], party[i]])

        return smooth_data

    
    left1 = moving_avg(left1)
    left2 = moving_avg(left2)
    left3 = moving_avg(left3)
    right1 = moving_avg(right1)
    right2 = moving_avg(right2)
    right3 = moving_avg(right3)
    top = moving_avg(top)
    middle = moving_avg(middle)
    bottom = moving_avg(bottom)

    parts = [left1, left2, left3, 
            right1, right2, right3, top, middle, bottom]
    
    ##### ABSOLUTE VELOCITY CALCULATIONS 

    def body_vel(middle):
        body_v = []
        for i in size: 
            if i > 1: 
                delta = np.subtract(middle[i], middle[i-1])
                norm = np.linalg.norm(delta)
                body_v.append(norm)

        body_v = pd.Series(body_v)
        body_v = round(body_v.ewm(alpha = 0.005, adjust= False).mean(), 5)
        body_v = body_v.tolist()

        return body_v

    body_velocity = body_vel(bottom)


    def abs_vel(part): 
        vel =[]
        rawvel = []
        for i in size:
            if i > 1: 
                delta = np.subtract(part[i], part[i-2])
                norm = np.linalg.norm(delta)
                rawvel.append(norm)

        """ sos = signal.butter(10 , 12, 'lp', fs = 100, output = 'sos')
        rawvel = signal.sosfilt(sos, rawvel) """
        for i in rawvel:
            if i > 5: 
                vel.append(1)
            else: 
                vel.append(0)

        return vel



    vel_left1p = abs_vel(left1)
    length = len(vel_left1p)
    vel_left2p = np.subtract(abs_vel(left2), [0.05]*length)
    vel_left3p = np.subtract(abs_vel(left3), [0.1]*length)
    vel_right1p = np.subtract(abs_vel(right1), [0.15]*length)
    vel_right2p = np.subtract(abs_vel(right2), [0.2]*length)
    vel_right3p = np.subtract(abs_vel(right3), [0.25]*length)

    ##### END VELOCITY CALCULATIONS 


    ##### HERE WE CALCULATE ANGLE OFFSET AND ROTATE THE BODY #####

    #Calculate the angle offset using top and middle 
    diff = []
    offset = []
    for i in size: 
        diff.append(np.subtract(middle[i], bottom[i])) 
        offset.append(np.arctan2(diff[i][1],diff[i][0]) + np.pi)


    #Define rotation function to rotate point about an angle
    def rotation(angle, point):
        rotation = [[np.cos(-np.pi/2 - angle), -np.sin(-np.pi/2 - angle)],
                    [np.sin(-np.pi/2 - angle), np.cos(-np.pi/2 - angle)]]

        newpoint = np.matmul(rotation,point)

        return newpoint

    #Here we iterate through each part in the list of body parts. 
    for part in parts: 
        for i in range(len(bottom)): 
            part[i] = np.subtract(part[i], bottom[i])
            part[i] = rotation(offset[i],part[i])
            part[i][1] = -1*part[i][1]


    def avg_leg_spread(parts):    
        avg_spread = []
        j = 0
        for part in parts:
            spreads = []
            if j == 0 or 3:
                for i in size:
                    spreads.append(np.linalg.norm(np.subtract(part[i], bottom[i])))
            if j == 1 or 4 : 
                for i in size:
                    spreads.append(np.linalg.norm(np.subtract(part[i], middle[i])))
            if j == 2 or 5 : 
                for i in size: 
                    spreads.append(np.linalg.norm(np.subtract(part[i], top[i])))

            j += 1
            sum = 0
            for spread in spreads:
                sum += spread
            
            avg_spread.append(sum/len(spreads))

        return(avg_spread)

    def leg_spread_plot():
        legs = ['left 1', 'left2', 'left3', 'right1', 'right2', 'right3']
        spreads = avg_leg_spread([left1, left2, left3, 
                                  right1, right2, right3])
        plt.plot(legs, spreads, 'o')
        plt.show()



    def angle_plot(offset, fig): 
        for i in size: 
            offset[i] = offset[i]*180/(np.pi)
        ax3 = fig.add_subplot(2,2,3)
        ax3.plot(size, offset, '-')
        #ax3.set_xlim(250,750)
        ax3.title.set_text('Body Angle vs Time')


    def line_plotting_example(fig):
        left1_1 = []
        left2_1 = []
        left3_1 = []
        right1_1 = []
        right2_1 = []
        right3_1 = []
        for i in size: 
            left1_1.append(left1[i][1])
            left2_1.append(left2[i][1])
            left3_1.append(left3[i][1])
            right1_1.append(right1[i][1])
            right2_1.append(right2[i][1])
            right3_1.append(right3[i][1])

        ax2 = fig.add_subplot(2,2,1)
        ax2.plot(size, left1_1, '-', color = 'blue')
        ax2.plot(size, left2_1, '-', color = 'red')
        ax2.plot(size, left3_1, '-', color = 'blue')

        ax2.plot(size, right1_1, '-', color = 'red')
        ax2.plot(size, right2_1, '-', color = 'blue')
        ax2.plot(size, right3_1, '-', color = 'red')
        #ax2.set_xlim(0,200)
        #ax2.set_ylim(-100,100)
        ax2.title.set_text('Foot Vertical Displacement vs Time')




    def gait_vel_plotting(fig):
        ax1 = fig.add_subplot(2,2,2)
        ax1.hlines(vel_left1p, range(length), range(1, length+1), color = 'red')
        ax1.hlines(vel_left2p, range(length), range(1, length+1), color = 'blue')
        ax1.hlines(vel_left3p, range(length), range(1, length+1), color = 'red')
        ax1.hlines(vel_right1p, range(length), range(1, length+1), color = 'blue')
        ax1.hlines(vel_right2p, range(length), range(1, length+1), color = 'red')
        ax1.hlines(vel_right3p, range(length), range(1, length+1), color = 'blue')

        #Set Gridlines
        spacing = 5 # This can be your user specified spacing. 
        minorLocator = MultipleLocator(spacing)
        # Set minor tick locations.
        ax1.xaxis.set_minor_locator(minorLocator)
        # Set grid to use minor tick locations. 
        ax1.grid(which = 'minor')
        ax1.set_ylim(0.7,1.1)
        #ax1.set_xlim(0,200)
        ax1.title.set_text('Foot Abs Velocty Vs Time')

    def body_vel_plotting(fig, body_velocity): 
        ax4 = fig.add_subplot(2,2,4)
        ax4.plot(range(len(body_velocity)), body_velocity)
        ax4.title.set_text('Body Velocity')
        #ax4.set_xlim(0,100)   
        #ax4.set_ylim(-1,5)   


    leg_spread_plot()
    
    fig = plt.figure()
    line_plotting_example(fig)
    gait_vel_plotting(fig)
    angle_plot(offset, fig)
    body_vel_plotting(fig, body_velocity)
    plt.show()
    print(np.mean(body_vel(middle)))



