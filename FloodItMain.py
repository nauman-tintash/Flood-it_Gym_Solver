import gym_flood
import gym
from random import choice
from collections import defaultdict
import numpy as np

import cv2
from matplotlib import pyplot as plt
import webcolors

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

#main function where the execusion should start
def main():
    print ('Hello, let\'s Flood It!!!')
    env = gym.make("Flood-v0")
    observation, possible = env.reset()
    
    #Reading and converting into grayScale
    img = cv2.imread("img2.png") 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Blurring the Image
    blurred_img = cv2.medianBlur(gray, 11)

    s_mask = 17
    #Along x-axis
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=s_mask))

    #Along y-axis
    sobely = np.abs(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=s_mask))

    #Merging the xy axis into one plot
    sobel_xy = 1.0 * sobelx + 1.0 * sobely

    ret,thresh_img = cv2.threshold(sobel_xy,127,255,cv2.THRESH_BINARY)
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

    
    index_col = 0
    index_row = 0
    colums = 0

    isOnBorder = False

    color_grid = {}
    grid_row = 0
    grid_col = 0

    for index_row in range(len(binary_sobelxy)):
        if (binary_sobelxy[index_row, index_col] != 0) :
            isOnBorder = True
        else:
            if (isOnBorder):
                isOnBorder = False
                grid_col = 0
                for index_col in range(len(binary_sobelxy[x])):
                    if (binary_sobelxy[index_row, index_col] != 0) :
                        isOnBorder = True
                    else:
                        if (isOnBorder):
                            color_grid[grid_row, grid_col] = img[index_row, index_col]
                            isOnBorder = False
                            grid_col += 1
                grid_row += 1
    for index_colors_row in range(grid_row - 1):
        for index_colors_col in range(grid_col - 1):
            print(color_grid[index_colors_row, index_colors_col], end="\t", flush=True)
            requested_colour = color_grid[index_colors_row, index_colors_col]
            actual_name, closest_name = get_colour_name(requested_colour)

            print ("Actual colour name:", actual_name, ", closest colour name:", closest_name)

        print("")
    #Giving keys according to my color values
    c = {"db": 0,
     "bl": 1,
     "gr": 2,
     "ye": 3,
     "re": 4,
     "pi": 5,
}
    
    

    # is_border = False
     #color_grid = []

    # for x in range(len(binary_sobelxy)):
    #     for y in range(len(binary_sobelxy[x])):
    #         if (sobel_xy[x,y] != 0):
    #                 is_border = True
    #         else:
    #             is_border = False
    #             [b, g, r]  = img[x,y]
    #             for col_x in range(len(binary_sobelxy)):

    #             np.array(color_grid)[:,:]
    #             color_grid = img[x,y]
    #             color_grid = np.array(color_grid, dtype='float')
    

    #     # r /= 255.0
    #     # g /= 255.0
    #     # b /= 255.0
    #     print(color_grid)
    #     #print(img)


    #plt.figure(figsize=(10,14))
    plt.subplot(2,2,1),plt.imshow(binary_sobelx, cmap = 'gray')
    plt.title('SobelX'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(binary_sobely, cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(binary_sobelxy,cmap = 'gray')
    plt.title('Grid'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(color_grid)
    plt.title('Original img Binary'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    
    #Colored grid
    

    #env.render()
    #bestmoves = 22
    
    #current = observation
    #previous = observation
    #done = False

    #stepAction = 0
    
    
    #env.render()
    
   # while done != True:
    #    for row in range(0,12):
    #        for column in range(len(observation[row])):
    #            current = observation[row][column]
    #            previous = observation[row][column-1]

#        stepAction = (stepAction + 1) % 6
 #       observation, reward, done, info = env.step(stepAction)
        #print (observation)
  #      env.render()
                # if (current != previous):
                #     observation, reward, done, info = env.step(current)
                #     env.render()
                #     print (len(info["moves"]))
                # if (len(info["moves"]) == bestmoves):
                #     
                # print ("Failed to complete the game")
                #     stop = True
                #     observation, possible = env.reset()
                #     return
            

main()