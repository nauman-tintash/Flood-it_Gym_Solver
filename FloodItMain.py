import gym_flood
import gym
from random import choice
from collections import defaultdict
import numpy as np

import cv2
from matplotlib import pyplot as plt

def convertColorValuesToID(color_grid, grid_row, grid_col) :
    colorID_grid = np.zeros((grid_row, grid_col))

    for index_colors_row in range(grid_row):
        for index_colors_col in range(grid_col):
   
            requested_colour = color_grid[index_colors_row, index_colors_col]
            
            #For red
            if requested_colour[0] >= 0 and requested_colour[0] <= 15 and requested_colour[1] > 150 :
                colorID_grid[index_colors_row][index_colors_col] = 4
            #For green
            elif requested_colour[0] >= 36 and requested_colour[0] <= 86 :
                colorID_grid[index_colors_row][index_colors_col] = 2
            #For yellow
            elif requested_colour[0] >= 15 and requested_colour[0] <= 36 :
                colorID_grid[index_colors_row][index_colors_col] = 3
            elif requested_colour[0] >= 90 and requested_colour[0] <= 130 :
                #For blue
                if requested_colour[2] >= 150 :
                    colorID_grid[index_colors_row][index_colors_col] = 0
                else :
                    #For cyan
                    colorID_grid[index_colors_row][index_colors_col] = 1
   
            #For white
            elif requested_colour[1] < 100 :
                colorID_grid[index_colors_row][index_colors_col] = 5
    print(colorID_grid)
    return colorID_grid
    
def color_possibleTiles(colorID_grid, grid_row, grid_col):
    previous_row = colorID_grid
    current_row = colorID_grid
    next_row = colorID_grid
    previous_col = colorID_grid
    current_col = colorID_grid
    next_col = colorID_grid
    color_grid = {}
    count = 0
    numOfSameTitles = 0
    row_count = 0
    column_count = 0

    # for row in range(grid_row):
    #     for column in range (grid_col):
    #         previous = current = colorID_grid[row][column]
    #         if(column+1 < grid_col):
    #             next = colorID_grid[row][column+1]
    #             if(current != next):
    #                 color_grid[count] = current,next
    #                 count = count + 1
    #                 current = next
    #                 break
    #             elif (current == next):
    #                 numOfSameTitles = numOfSameTitles + 1
                    #break

    neigbours = {}
    for row in range(grid_row):
        for column in range(grid_col):
            if(row+1 < grid_row and column+1 < grid_col):
                neigbours[count] = colorID_grid[row][column], colorID_grid[row][column+1], colorID_grid[row+1][column+1]
                count = count + 1

    # for key in color_grid:
    #     if(key+1 < grid_col):
    #         if(color_grid[key][0] == color_grid[key+1][0]):


                
    print("Same color", neigbours)

def get_binary_sobelxy(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #Along x-axis
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))

    #Along y-axis
    sobely = np.abs(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3))

    ret, binary_sobelx = cv2.threshold(sobelx, 127, 255, cv2.THRESH_BINARY)
    ret, binary_sobely = cv2.threshold(sobely, 127, 255, cv2.THRESH_BINARY)
    
    #Drawing grid by sobel x
    for x in range(len(binary_sobelx)):
        for y in range(len(binary_sobelx[x])):
            if (binary_sobelx[x,y] == 255):
                binary_sobelx[:,y] = 255

    #Drawing grid by sobel y
    for x in range(len(binary_sobely)):
        for y in range(len(binary_sobely[x])):
            if (binary_sobely[x,y] == 255):
                binary_sobely[x,:] = 255

    binary_sobelxy = 1.0 * binary_sobelx + 1.0 * binary_sobely
    
    return binary_sobelxy

def getGridFromImage(img, binary_sobelxy) :
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    index_col = 0
    index_row = 0

    isOnBorder = False

    color_grid = {}
    grid_row = 0
    grid_col = 0

    #print("Binary: ",binary_sobelxy)
    for index_row in range(len(binary_sobelxy)):
        if (binary_sobelxy[index_row, index_col] != 0) :
            isOnBorder = True
        else:
            if (isOnBorder or index_row == 0):
                isOnBorder = False
                grid_col = 0
                for index_col in range(len(binary_sobelxy[0])):
                    if (binary_sobelxy[index_row, index_col] != 0) :
                        isOnBorder = True
                    else:
                        if (isOnBorder or index_col == 0):
                            color_grid[grid_row, grid_col] = hsv_img[index_row, index_col]
                            isOnBorder = False
                            grid_col += 1
                grid_row += 1
    
    colorIDGrid = convertColorValuesToID(color_grid, grid_row, grid_col)
    color_possibleTiles(colorIDGrid, grid_row, grid_col)
    #print(colorIDGrid)

    return colorIDGrid

#main function where the execusion should start
def main():
    print ('Hello, let\'s Flood It!!!')
    env = gym.make("Flood-v0")
    observation, possible = env.reset()

    binary_sobelxy = get_binary_sobelxy(observation)    
    processedObservation = getGridFromImage(observation, binary_sobelxy)
            
    env.render()
    bestmoves = 22
    steps = 0
    current = previous = processedObservation
    done = False
    
    while done != True:
        for row in range(len(processedObservation)):
            for column in range(len(processedObservation[row])):
                #print(processedObservation)
                current = processedObservation[row][column]
                previous = processedObservation[row][column-1]
                if (current != previous):
                    observation, reward, done, info = env.step(int(current))
                    processedObservation = getGridFromImage(observation, binary_sobelxy)
                    #print(processedObservation)
                    env.render()
                        # if (len(info["moves"]) == bestmoves):
                        #     print ("Failed to complete the game")
                        #     observation, possible = env.reset()
                        #     return
            

    plt.show()
main()