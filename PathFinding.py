# Imported libaries
import matplotlib.pyplot as plt
import pygame
from queue import PriorityQueue
import random
import time

# Creating the pygame window
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Path Finding Algorithms")

# Initialising font
pygame.init()
font=pygame.font.SysFont('dejavusans', 25)

# Setting global variables
Distance = 0        # Current Distance
DistanceArray = []  # Stored Distances
Time = '0'          # Current Time
TimeArray = []      # Stored Times
Operations = 0      # Current Operations
change = False      # Changes occured while the program is running
graph = False       # Graph open

# Colours
GREEN = (0, 255, 0)
LIGHTGREEN = (200, 255, 200)
BLUE = (193, 225, 246)
DARKBLUE = (70, 90, 246)
RED = (255, 26, 26)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)

# Creating a class for a single grid cell
class Cell:
    # Initialise class attributes
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.colour = WHITE
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows

    # Get Cell conditions
    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.colour == BLUE

    def is_open(self):
        return self.colour == LIGHTGREEN

    def is_barrier(self):
        return self.colour == BLACK

    def is_start(self):
        return self.colour == GREEN

    def is_end(self):
        return self.colour == RED
    
    def is_path(self):
        return self.colour == YELLOW

    # Set Cell condition
    def reset(self):
        self.colour = WHITE

    def make_closed(self):
        self.colour = BLUE

    def make_open(self):
        self.colour = LIGHTGREEN

    def make_barrier(self):
        self.colour = BLACK
    
    def make_start(self):
        self.colour = GREEN

    def make_end(self):
        self.colour = RED

    def make_path(self):
        self.colour = YELLOW

    # Draw cell
    def draw(self, win):
        pygame.draw.rect(win, self.colour, (self.x, self.y, self.width, self.width))

    # Check neighbours states around cell arent barriers or edges 
    def update_neighbours(self, grid):
        self.neighbours = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # down
            self.neighbours.append(grid[self.row + 1][self.col])# add neighbour

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # up
            self.neighbours.append(grid[self.row - 1][self.col])# add neighbour

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # right
            self.neighbours.append(grid[self.row][self.col + 1])# add neighbour

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # left
            self.neighbours.append(grid[self.row][self.col - 1])# add neighbour

# Manhattan Heuristic
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # Returns the estimated distance of the cell to the endpoint
    return abs(x1 - x2) + abs(y1 - y2)


# Retracing the end to start path
def reconstruct_path(came_from, current, draw, start):
    num = 0 # Distance starts at 0
    # while loop of path coordiantes
    while current in came_from:
        num +=1
        current = came_from[current]
        current.make_path()
        draw()
    # Setting the distance score
    print("Distance: " +str(num)+" Blocks")
    global Distance
    Distance = num
    if graph: # If graph is true store results
        global DistanceArray
        DistanceArray.append(Distance)
    # re-drawing the start and endpoint
    current.make_end()
    start.make_start()

# Path finding algorithm
def Algorithm(draw, grid, start, end, al):
    # start timer
    t0 = time.time()
    # Operations start at 0
    operation = 0
    count = 0
    # Initialise priority queue
    open_set = PriorityQueue()
    # Add start point to priority queue
    open_set.put((0, count, start))
    # Initilaise path array
    came_from = {}
    # float("inf") unbound upper value used for comparison
    g_score = {cell: float("inf") for row in grid for cell in row}
    # G_score of startpoint set to 0
    g_score[start] = 0
    f_score = {cell: float("inf") for row in grid for cell in row}
    # F_score for start point is the manhattan distance from start to end coordinates
    f_score[start] = h(start.get_pos(), end.get_pos())
    open_set_hash = {start}
    # Detect a change to display
    global change
    # No current changes have been made during the running process
    change = False

    # Searching loop
    while not open_set.empty():
        operation +=1 # Add operation
        # Inputs while running
        for event in pygame.event.get():
            # Quit
            if event.type == pygame.QUIT:
                pygame.quit()

             # Place
            if pygame.mouse.get_pressed()[0] and not graph:
                change = True # Set change to true
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, 25, WIDTH)
                cell = grid[row][col] # Cell being interacted with
                if not start and cell != end: # Make start
                    start = cell
                    start.make_start()
                elif not end and cell != start: # Make end
                    end = cell
                    end.make_end()
                elif cell != end and cell != start: # Make barrier
                    cell.make_barrier()
            # Remove
            elif pygame.mouse.get_pressed()[2] and not graph:
                change = True # Set change to true
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, 25, WIDTH)
                cell = grid[row][col] # Cell being interacted with
                if cell != start and cell != end: # Only clear barrier
                    cell.reset()


        current = open_set.get()[2]
        open_set_hash.remove(current)

        # End conditions
        if current == end and change == False:
            # Make path
            reconstruct_path(came_from, end, draw, start)
            # Set end
            end.make_end()
            # End time
            t1 = time.time()
            # Time taken
            timeTaken = "{:.2f}".format(t1-t0)
            print("Time: "+timeTaken+"s")
            global Time
            Time = timeTaken # Set time taken
            if graph: # If graph store time
                global TimeArray
                TimeArray.append(Time)
            print("Operations: " +str(operation))
            global Operations
            Operations = operation # Set operations
            print("------------------------")
            return True
        elif change == True: # If change was made dont meet end condition
            return False

        # Set scores for current cells being searched
        for neighbour in current.neighbours:
            # Increase g_score
            temp_g_score = g_score[current] + 1

            # If temp_g_score is smaller than current neighbour g_score then store new score
            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                if(al == "A*"):# Do A* equation
                    f_score[neighbour] = temp_g_score + h(neighbour.get_pos(), end.get_pos())
                elif(al == "BFS"):# Do BFS equation
                    f_score[neighbour] = h(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    if(al == "Dijkstras"):# Do Dijikstra's equation
                        open_set.put((g_score[neighbour], count, neighbour))
                    else:# Store A* and BFS 
                        open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()
        draw()

        # close cell
        if current != start:
            current.make_closed()

    return False

# Layout grid on screen
def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            cell = Cell(i, j, gap, rows)
            grid[i].append(cell)
    
    return grid

# Draw grid on pygame window
def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

# Instruction Page
def instructions(win):
    win.fill(WHITE)
    # Text
    text1 = font.render("-------------------------PATH FINDING ALGORITHMS-------------------------", True, DARKBLUE)
    text2 = font.render("-------------CONTROLS-------------", True, BLACK)
    text3 = font.render("LEFT Click: PLACE start point, end point, walls", True, BLACK)
    text4 = font.render("RIGHT Click: REMOVE start point, end point, walls", True, BLACK)
    text5 = font.render("SPACE: RUNS search algorithm", True, BLACK)
    text6 = font.render("C: CLEARS screen", True, BLACK)
    text7 = font.render("--------CHANGE ALGORITHM--------", True, BLACK)
    text8 = font.render("1: DIJKSTRA'S algorithm", True, BLACK)
    text9 = font.render("2: A* algorithm", True, BLACK)
    text10 = font.render("3: BEST-FIRST-SEARCH algorithm", True, BLACK)
    text11 = font.render("----------PRE-MADE MAPS----------", True, BLACK)
    text12 = font.render("G: GENERATES RANDOM maze", True, BLACK)
    text13 = font.render("H: GENERATES FIXED maze", True, BLACK)
    text14 = font.render("----------GRAPH----------", True, BLACK)
    text15 = font.render("M: GENERATES GRAPH", True, BLACK)
    text16 = font.render("PRESS P TO OPEN & CLOSE", True, RED)

    # Locaions
    Tlocations = [[text1, 25], [text2, 90], [text3, 120], [text4, 150], 
    [text5, 180], [text6, 210], [text7, 270], [text8, 300], [text9, 330], 
    [text10, 360], [text11, 410], [text12, 440], [text13, 470], 
    [text14, 520], [text15, 550], [text16, 640]]
    
    # Render
    for l in Tlocations:
        text_rect = l[0].get_rect(center=(WIDTH/2, l[1]))
        win.blit(l[0], text_rect)

    pygame.display.update()   

# Display the results on screen
def results(win, Distance, Time, Operations, algorithm):
    textAl = font.render(algorithm+" Algorithm", True, DARKBLUE)
    textD = font.render("Distance: "+str(Distance)+" Blocks", True, DARKBLUE)
    textT = font.render("Time: "+Time+"s", True, DARKBLUE)
    textO = font.render("Operations: "+str(Operations), True, DARKBLUE)
    text_rect = textAl.get_rect(center=(WIDTH/2, 12.5))
    win.blit(textAl, text_rect)
    win.blit(textD, (5, 0))
    win.blit(textT, (5, 30))
    win.blit(textO, (5, 60))
    pygame.display.update()

# Draw on the pygame window
def draw(win, grid, rows, width, Distance, Time, Operations, algorithm, instruct):
    win.fill(WHITE)

    for row in grid: 
        for cell in row:
            cell.draw(win)

    if(instruct):
        instructions(win)
    else:
        draw_grid(win, rows, width)
        results(win, Distance, Time, Operations, algorithm)

    pygame.display.update()

# Return cell clicked in
def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

# Returns Largest number
def largest(num1, num2, num3):
    if (num1 > num2) and (num1 > num3):
        largest_num = num1
    elif (num2 > num1) and (num2 > num3):
        largest_num = num2
    else:
        largest_num = num3
    return largest_num

# Returns smallest number
def smallest(num1, num2, num3):
    if (num1 < num2) and (num1 < num3):
        smallest_num = num1
    elif (num2 < num1) and (num2 < num3):
        smallest_num = num2
    else:
        smallest_num = num3
    return smallest_num

# Main program
def main(win, width):
    # Setting main variables
    ROWS = 25                     # Grid rows
    grid = make_grid(ROWS, WIDTH) # Layout grid
    start = None                  # Set start to none
    end = None                    # Set end to none
    algorithm = "Dijkstras"       # Set initial algorithm
    instruct = True               # Start instruction page
    global graph
    graph = False                 # Dont display graph
    run = True                    # Run program

    # Display defult algorithm in terminal window
    print("---------------Defult DIJKSTRA'S-----------------")

    # Run program
    while run:
        # Draw to pygame window
        draw(win, grid, ROWS, width, Distance, Time, Operations, algorithm, instruct)

        # If a change occurs start the algorithm again
        if change == True:
            for row in grid:
                for cell in row:
                    cell.update_neighbours(grid)
                    if cell.is_open():
                        cell.reset()
                    elif cell.is_closed():
                        cell.reset()
            # Lambda used to summarize argument and return the result
            # Alogithm() runs the chosen algorithm
            Algorithm(lambda: draw(win, grid, ROWS, width, Distance,
                     Time, Operations, algorithm, instruct), grid , start, end, 
                     algorithm)
        
        # Scan events
        for event in pygame.event.get():
            # Quit
            if event.type == pygame.QUIT:
                run = False
            # Place
            if pygame.mouse.get_pressed()[0] and not instruct:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                cell = grid[row][col] # Cell being interated with
                if not start and cell != end: # Make start
                    start = cell
                    start.make_start()
                elif not end and cell != start: # Make end
                    end = cell
                    end.make_end()
                elif cell != end and cell != start: # Make barrier
                    cell.make_barrier()
            # Remove
            elif pygame.mouse.get_pressed()[2] and not instruct:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                cell = grid[row][col] # Cell being interated with
                cell.reset() # Clear cell
                if cell == start: # If start set start to none
                    start = None
                elif cell == end: # If end set end to none
                    end = None
            # Keyboard input
            if event.type == pygame.KEYDOWN:
                # Change to Dijkstra's
                if event.key == pygame.K_1 and not instruct:
                    algorithm = "Dijkstras"
                    print("---------------DIJKSTRA'S ACTIVE-----------------")
                # Change to A*
                if event.key == pygame.K_2 and not instruct:
                    algorithm = "A*"
                    print("-------------------A* ACTIVE---------------------")
                # Change to Best-First Search
                if event.key == pygame.K_3 and not instruct:
                    algorithm = "BFS"
                    print("---------------BEST FIRST ACTIVE-----------------")
                # Start pathfinding algorithm 
                if event.key == pygame.K_SPACE and start and end and not instruct:
                    for row in grid:
                        for cell in row:
                            cell.update_neighbours(grid)
                            # Reset screen
                            if cell.is_open():
                                cell.reset()
                            elif cell.is_closed():
                                cell.reset()
                            elif cell.is_path():
                                cell.reset()
                    # Lambda used to summarize argument and return the result
                    # Alogithm() runs the chosen algorithm
                    Algorithm(lambda: draw(win, grid, ROWS, width, Distance,
                     Time, Operations, algorithm, instruct), grid , start, end, 
                     algorithm)
                # Clear window
                if event.key == pygame.K_c and not instruct:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
                # Display or hide instruction window
                if event.key == pygame.K_p:
                    if(instruct == False):
                        instruct = True
                    else:
                        instruct = False
                # Display fixed map
                if event.key == pygame.K_h and not instruct:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

                    # Wall coordinates
                    if(ROWS < 50):
                        walls = [(10,19),(11,19),(12,19),(13,19),
                        (14,19),(10,20),(10,21),(10,22),(10,23),
                        (14,20),(14,21),(14,22),(14,23)]
                    elif(ROWS >= 50):
                        walls = [(23,44),(24,44),(25,44),(26,44),
                        (27,44),(23,45),(23,46),(23,47),(23,48),
                        (27,45),(27,46),(27,47),(27,48)]

                    # Make barrier
                    for w in walls:
                        cell = grid[w[0]][w[1]]
                        cell.make_barrier()

                    for i in range(ROWS):
                        if(i > 3 and i < ROWS - 4):
                            cell = grid[i][ROWS//2]
                            cell.make_barrier()

                        cell = grid[i][ROWS//4]
                        cell.make_barrier()

                        if(i > 7 and i < ROWS - 8):
                            cell = grid[i][ROWS//4]
                            cell.reset()

                    # Set start and end
                    cell = grid[ROWS//2][3]
                    start = cell
                    start.make_start()
                    cell = grid[ROWS//2][ROWS - 4]
                    end = cell
                    end.make_end()
                # Display random maze
                if event.key == pygame.K_g and not instruct:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

                    # Fill barriers
                    for i in range(ROWS):
                        for j in range(ROWS):
                            cell = grid[i][j]
                            cell.make_barrier()

                    # Clear sections of barries
                    for i in range(ROWS):
                        for j in range(ROWS):
                            if i > 0 and i < ROWS - 1 and j > 0 and j < ROWS - 1:
                                position = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
                                for p in position:
                                    cell = grid[round(ROWS*p)][j]
                                    cell.reset()
                                    cell = grid[i][round(ROWS*p)]
                                    cell.reset()

                                # Randomly set barriers
                                if not grid[i][j].is_barrier():
                                    num = random.randint(0, 5)
                                    if num == 0:
                                        cell = grid[i][j]
                                        cell.make_barrier()
                                        i += 2
                                        j += 2
                    
                    # Clear barriers around start and end
                    for i in range(6):
                        for j in range(6):
                            if i > 0 and i < ROWS - 1 and j > 0 and j < ROWS - 1:
                                cell = grid[i][j]
                                cell.reset()
                                cell = grid[ROWS - 1 - i][ROWS - 1 - j]
                                cell.reset()

                    # Set start and end   
                    cell = grid[1][1]
                    start = cell
                    start.make_start()
                    cell = grid[ROWS - 2][ROWS - 2]
                    end = cell
                    end.make_end()

                # Generate Time Graph
                if event.key == pygame.K_m and start and end and not instruct and not graph:
                    graph = True
                    al = ["BFS", "A*", "Dijkstras"]
                    # Run each algorithm 3 times
                    for each in al:
                        print("---------------"+each+"-----------------")
                        for i in range(3):
                            for row in grid:
                                for cell in row:
                                    cell.update_neighbours(grid)
                                    if cell.is_open():
                                        cell.reset()
                                    elif cell.is_closed():
                                        cell.reset()
                                    elif cell.is_path():
                                        cell.reset()
                            # Lambda used to summarize argument and return the result
                            # Alogithm() runs the chosen algorithm
                            Algorithm(lambda: draw(win, grid, ROWS, width, Distance,
                            Time, Operations, each, instruct), grid , start, end, 
                            each)

                    plt.style.use('ggplot')
                    # x axis values and the path distance found
                    x1 = al[0]+" ("+str(DistanceArray[0])+")"
                    x2 = al[1]+" ("+str(DistanceArray[3])+")"
                    x3 = al[2]+" ("+str(DistanceArray[6])+")"
                    x = [x1, x2, x3]
                    # corresponding y axis values ( average of 3 times )
                    T1avg = (float(TimeArray[0])+float(TimeArray[1])+float(TimeArray[2]))/3
                    T2avg = (float(TimeArray[3])+float(TimeArray[4])+float(TimeArray[5]))/3
                    T3avg = (float(TimeArray[6])+float(TimeArray[7])+float(TimeArray[8]))/3
                    time = [T1avg,T2avg,T3avg]
                    # Varience in results
                    v1 = largest(float(TimeArray[0]),float(TimeArray[1]),float(TimeArray[2])) - smallest(float(TimeArray[0]),float(TimeArray[1]),float(TimeArray[2]))
                    v2 = largest(float(TimeArray[3]),float(TimeArray[4]),float(TimeArray[5])) - smallest(float(TimeArray[3]),float(TimeArray[4]),float(TimeArray[5]))
                    v3 = largest(float(TimeArray[6]),float(TimeArray[7]),float(TimeArray[8])) - smallest(float(TimeArray[6]),float(TimeArray[7]),float(TimeArray[8]))
                    variance = [v1, v2, v3]
                    # Bar Graph
                    x_pos = [i for i, _ in enumerate(x)]
                    plt.bar(x_pos, time, color='green', yerr=variance)
                    # Labels
                    plt.xlabel("Path Find Algorithm (Distance)")
                    plt.ylabel("Time Taken (s)")
                    plt.title("Time Taken by Various Pathfinding Algorithms")
                    # Display graph
                    plt.xticks(x_pos, x)
                    plt.show()
                    # Reset time and distance arrays
                    TimeArray.clear()
                    DistanceArray.clear()
                    graph = False
    # Quit pygame
    pygame.quit()
# Run the main program
main(WIN, WIDTH)