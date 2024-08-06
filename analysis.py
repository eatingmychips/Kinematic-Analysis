import csv 
import pandas as pd
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal
import random

def file_read(file):
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
    
    parts = [left1, left2, left3, right1, right2, right3, top, middle, bottom]
    
    return parts



def body_vel(middle):
    body_v = []
    size = range(len(middle))
    for i in size: 
        if i > 1: 
            delta = np.subtract(middle[i], middle[i-1])
            norm = np.linalg.norm(delta)
            body_v.append(norm)
        
    body_v = pd.Series(body_v)
    body_v = round(body_v.ewm(alpha = 0.005, adjust= False).mean(), 5)
    body_v = body_v.tolist()

    return body_v


def gait_spread(spread_parts): 
    sampled_parts = []
    for file in sampled_parts:
        for part in file:
            part = random.sample(part, len(part)/5)
            sampled_parts.append(part)

    return sampled_parts



def leg_abs_velocity(part):
    vel =[]
    rawvel = []
    size = range(len(part))
    for i in size:
        if i > 0: 
            delta = np.subtract(part[i], part[i-2])
            norm = np.linalg.norm(delta)
            rawvel.append(norm)

    for i in rawvel:
        if i > 4.5: 
            vel.append(1)
        else: 
            vel.append(0)

    return vel

def leg_time(vel):
    swing = []
    stand = []
    for key, iter in it.groupby(vel):
        if key == 1:
            swing.append(len(list(iter)))
        elif key == 0:
            stand.append(len(list(iter)))

    stand = [x for x in stand if 3 < x < 90]
    swing = [x for x in swing if 3 < x ]
    gait_time = (len(stand) + len(swing))*1/100
    avg_swing = np.mean(swing)
    avg_stand = np.mean(stand)
    gait_time = (avg_swing + avg_stand)*1/100
    
    gait = [avg_stand/(avg_stand+avg_swing)*100, avg_swing/(avg_stand + avg_swing)*100]
    
    return [gait, gait_time]




def part_rotation(parts):
    diff = []
    offset = []
    middle = parts[7]
    bottom = parts[8]
    size = range(len(middle))
    for i in size: 
        diff.append(np.subtract(middle[i], bottom[i]))
        offset.append(np.arctan2(diff[i][1],diff[i][0]))


    #Define rotation function to rotate point about an angle
    def rotation(angle, point):
        rotation = [[np.cos(-np.pi/2 - angle), -np.sin(-np.pi/2 - angle)],
                    [np.sin(-np.pi/2 - angle), np.cos(-np.pi/2 - angle)]]

        newpoint = np.matmul(rotation,point)

        return newpoint

    #Here we iterate through each part in the list of body parts. 
    for part in parts: 
        for i in size: 
            part[i] = np.subtract(part[i], bottom[i])
            part[i] = rotation(offset[i],part[i])
            part[i][1] = -1*part[i][1]

    return parts

def avg_leg_spread(parts):    
    avg_spread = []
    j = 0
    top = parts[6]
    middle = parts[7]
    bottom = parts[8]

    for part in parts[0:6]:
        size = range(len(part))
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
        spread = np.mean(spreads)
        
        avg_spread.append(spread)

    avg_spread = np.mean(avg_spread)
    return(avg_spread)

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


def gait_phase_plotting(file): 

    parts = file_read(file)
    parts = part_rotation(parts)
    left1 = moving_avg(parts[0])
    left2 = moving_avg(parts[1])
    left3 = moving_avg(parts[2])
    right1 = moving_avg(parts[3])
    right2 = moving_avg(parts[4])
    right3 = moving_avg(parts[5])
    left1_1 = []
    left2_1 = []
    left3_1 = []
    right1_1 = []
    right2_1 = []
    right3_1 = []
    size = range(len(left1))

    for i in size: 
        left1_1.append(left1[i][1])
        left2_1.append(left2[i][1])
        left3_1.append(left3[i][1])
        right1_1.append(right1[i][1])
        right2_1.append(right2[i][1])
        right3_1.append(right3[i][1])
    

    return left1_1, left2_1, left3_1, right1_1, right2_1, right3_1

