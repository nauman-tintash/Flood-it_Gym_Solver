from copy import deepcopy
import sys

from six import StringIO

from matplotlib import pyplot as plt

from numpy import zeros
import numpy as np

import gym
import gym.spaces
from gym.utils import colorize

c = {"db": 0,
     "bl": 1,
     "gr": 2,
     "ye": 3,
     "re": 4,
     "pi": 5,
}

colValues = [[0, 0, 120],[0, 0, 250],[0, 250, 0],[250, 250, 0],[250, 0, 0],[250, 250, 250]]

# mapping for colorize
c2 = {"db": "blue",
      "bl": "cyan",
      "gr": "green",
      "ye": "yellow",
      "re": "red",
      "pi": "white"
}


MAP1 = [
    [c["gr"], c["db"], c["gr"], c["gr"], c["bl"], c["gr"], c["bl"], c["gr"], c["db"], c["bl"], c["bl"], c["db"]],
    [c["gr"], c["gr"], c["db"], c["bl"], c["db"], c["gr"], c["db"], c["gr"], c["db"], c["db"], c["bl"], c["gr"]],
    [c["gr"], c["db"], c["db"], c["gr"], c["db"], c["gr"], c["db"], c["db"], c["db"], c["gr"], c["db"], c["bl"]],
    [c["db"], c["bl"], c["db"], c["bl"], c["db"], c["gr"], c["bl"], c["db"], c["bl"], c["gr"], c["db"], c["bl"]],
    [c["bl"], c["db"], c["gr"], c["bl"], c["db"], c["db"], c["db"], c["db"], c["bl"], c["gr"], c["db"], c["gr"]],
    [c["gr"], c["bl"], c["gr"], c["db"], c["bl"], c["db"], c["bl"], c["bl"], c["gr"], c["gr"], c["db"], c["db"]],
    [c["bl"], c["bl"], c["bl"], c["db"], c["bl"], c["bl"], c["db"], c["gr"], c["gr"], c["db"], c["bl"], c["db"]],
    [c["db"], c["db"], c["db"], c["gr"], c["db"], c["bl"], c["gr"], c["db"], c["gr"], c["gr"], c["db"], c["bl"]],
    [c["gr"], c["bl"], c["db"], c["bl"], c["db"], c["bl"], c["db"], c["gr"], c["db"], c["bl"], c["gr"], c["gr"]],
    [c["db"], c["db"], c["gr"], c["gr"], c["db"], c["bl"], c["db"], c["gr"], c["gr"], c["bl"], c["gr"], c["gr"]],
    [c["db"], c["bl"], c["gr"], c["db"], c["db"], c["gr"], c["gr"], c["db"], c["bl"], c["bl"], c["gr"], c["gr"]],
    [c["db"], c["gr"], c["gr"], c["db"], c["db"], c["bl"], c["db"], c["bl"], c["db"], c["bl"], c["bl"], c["db"]]
]

MAP2 = [
    [c["db"], c["db"], c["gr"], c["ye"], c["gr"], c["gr"], c["ye"], c["db"], c["gr"], c["ye"], c["bl"], c["gr"]],
    [c["db"], c["db"], c["ye"], c["db"], c["bl"], c["pi"], c["ye"], c["pi"], c["gr"], c["bl"], c["ye"], c["gr"]],
    [c["gr"], c["bl"], c["ye"], c["db"], c["re"], c["pi"], c["gr"], c["pi"], c["bl"], c["pi"], c["db"], c["gr"]],
    [c["pi"], c["bl"], c["bl"], c["pi"], c["ye"], c["db"], c["ye"], c["db"], c["pi"], c["db"], c["db"], c["db"]],
    [c["pi"], c["bl"], c["ye"], c["gr"], c["re"], c["db"], c["pi"], c["bl"], c["re"], c["re"], c["db"], c["gr"]],
    [c["ye"], c["db"], c["gr"], c["re"], c["bl"], c["pi"], c["gr"], c["ye"], c["pi"], c["db"], c["re"], c["bl"]],
    [c["re"], c["pi"], c["ye"], c["db"], c["pi"], c["ye"], c["gr"], c["re"], c["gr"], c["ye"], c["re"], c["db"]],
    [c["pi"], c["bl"], c["re"], c["re"], c["re"], c["bl"], c["ye"], c["bl"], c["db"], c["gr"], c["bl"], c["db"]],
    [c["ye"], c["db"], c["gr"], c["bl"], c["re"], c["pi"], c["ye"], c["ye"], c["gr"], c["re"], c["re"], c["ye"]],
    [c["re"], c["gr"], c["ye"], c["ye"], c["ye"], c["ye"], c["re"], c["bl"], c["pi"], c["pi"], c["gr"], c["gr"]],
    [c["db"], c["re"], c["pi"], c["db"], c["pi"], c["re"], c["db"], c["db"], c["db"], c["bl"], c["re"], c["db"]],
    [c["db"], c["ye"], c["bl"], c["db"], c["re"], c["db"], c["re"], c["gr"], c["db"], c["re"], c["pi"], c["pi"]]
]

class FloodEnv(gym.Env):
    metadata = {"render.modes": ["graphics", "ansi"]}

    renderMode = "ansi"

    imga=zeros([120,120,3], dtype=np.uint8)
    imga.fill(150)

    plt.imshow(imga)

    def __init__(self):
        self.action_space = gym.spaces.Discrete(6)
        self.maxmoves = 32
        self.rc = {}
        for color, num in c.items():
            self.rc[num] = color
        self.reset()

    def step(self, action):
        self.flood_board(action)
        self.moves.append(action)

        if self.renderMode == "graphics":
            for row in range(len(self.board)):
                for col in range(len(self.board)):
                    color = self.board[row][col]
                    
                    for pixRow in range(row * 10, row*10 + 10):
                        for pixCol in range(col * 10, col*10 + 10):
                            self.imga[pixRow][pixCol] = colValues[color]
            observation = self.imga
        else:
            observation = self.board

        done = self.solved()
        if len(self.moves) > self.maxmoves:
            done = True
            reward = 0.0
        else:
            reward = len(self.connected_tiles()) / (1.0 * len(self.board) * len(self.board[0]))
        info = {"maxmoves": self.maxmoves,
                "moves": [self.rc[num] for num in self.moves],
                "possible": self.possible_colors()}

        return observation, reward, done, info

    def reset(self, renderMode = "graphics"):
        self.board = deepcopy(MAP2)
        self.start = (0, 0)
        self.moves = []
        self.renderMode = renderMode

        if renderMode == "graphics":
            for row in range(len(self.board)):
                for col in range(len(self.board)):
                    color = self.board[row][col]
                    for pixRow in range(row * 10, row*10 + 10):
                        for pixCol in range(col * 10, col*10 + 10):
                            self.imga[pixRow][pixCol] = colValues[color]
            observation = self.imga
        else:
            observation = self.board

        return observation, self.possible_colors()[0]

    def render(self, close=False):
        if self.renderMode == "graphics":
            plt.imshow(self.imga)
            plt.draw()
        
        else:
            outfile = sys.stdout
            
            for row in range(len(self.board)):
                outrow = [colorize(self.rc[num][0].upper(), c2[self.rc[num]]) for num in self.board[row]]
                outfile.write("".join(outrow) + "\n")
            outfile.write("moves: %d %s\n" %
                        (len(self.moves),
                        " ".join([self.rc[num] for num in self.moves]) + "\n"))
            return outfile

    def solved(self):
        first = self.board[0][0]
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] != first:
                    return False

        return True

    def possible_tiles(self, row, col, same=True):
        rows = len(self.board)
        cols = len(self.board[0])
        initial = self.board[row][col]
        tiles = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        rettiles = []

        for tile in tiles:
            if tile[0] < 0 or tile[1] < 0 or tile[0] >= rows or tile[1] >= cols:
                continue
            if same:
                if self.board[tile[0]][tile[1]] != initial:
                    continue
            else:
                if self.board[tile[0]][tile[1]] == initial:
                    continue
            rettiles.append(tile)

        return rettiles

    def connected_tiles(self, row=0, col=0):
        visited = [(row, col)]
        to_visit = self.possible_tiles(row, col)
        tile = None

        while to_visit:
            tile = to_visit.pop(0)
            visited.append(tile)
            maybe = [tile for tile in self.possible_tiles(tile[0], tile[1])
                     if tile not in visited and tile not in to_visit]
            if maybe or to_visit:
                to_visit.extend(maybe)

        visited.reverse()

        return visited

    def flood_board(self, color):
        for tile in self.connected_tiles():
            self.board[tile[0]][tile[1]] = color

    def possible_colors(self):
        visited = []
        colors = []
        connected = self.connected_tiles()
        for tile in connected:
            for ptile in self.possible_tiles(tile[0], tile[1], False):
                if ptile not in connected and ptile not in visited:
                    visited.append(ptile)
                    colors.append(self.board[ptile[0]][ptile[1]])
        return colors, visited


# eof
