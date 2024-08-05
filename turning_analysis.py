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


######## Here we import the files necessary for turning analysis #######

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

zero_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\0degreesTurning"+x 
             for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\0degreesTurning")]

forty_five_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\45degreesTurning"+x 
                      for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\45degreesTurning")]

ninety_degrees = [r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\90degreesTurning"+x 
                  for x in find_csv_filenames(r"C:\Users\lachl\OneDrive\Thesis\Data\KinematicAnalysisFinalData\90degreesTurning")]






def file_read(file):
    df = pd.read_csv(file, skiprows=2)
    
    new_column_names = ['coords', 'left1x', 'left1y', 'likelihood', 'left2x', 'left2y', 'likelihood', 'left3x', 'left3y', 'likelihood', 
                    'right1x', 'right1y', 'likelihood', 'right2x', 'right2y', 'likelihood', 'right3x', 'right3y', 'likelihood',
                     'topx', 'topy', 'likelihood', 'middlex', 'middley', 'likelihood', 'bottomx', 'bottomy', 'likelihood' ]
    df.columns = new_column_names

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