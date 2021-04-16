# Imported libaries
import pygame
import random
import time

from queue import PriorityQueue

# Creating the pygame window
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Path Finding Algorithms")

# Creating the font
pygame.init()
font=pygame.font.SysFont('dejavusans', 25)

# Setting global variables
Distance = 0
Time = '0'
Operations = 0

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
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows

    # Cell properties
    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == BLUE

    def is_open(self):
        return self.color == LIGHTGREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == GREEN

    def is_end(self):
        return self.color == RED

    def reset(self):
        self.color = WHITE

    def make_closed(self):
        self.color = BLUE

    def make_open(self):
        self.color = LIGHTGREEN

    def make_barrier(self):
        self.color = BLACK
    
    def make_start(self):
        self.color = GREEN

    def make_end(self):
        self.color = RED

    def make_path(self):
        self.color = YELLOW

    # Draw cell
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    # Check neighbours around cell arent barriers 
    def update_neighbours(self, grid):
        self.neighbours = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): #down
            self.neighbours.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): #up
            self.neighbours.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): #right
            self.neighbours.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): #left
            self.neighbours.append(grid[self.row][self.col - 1])

# Manhattan Heuristic
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


# Retracing the end to start path
def reconstruct_path(came_from, current, draw, start):
    num = 0
    while current in came_from:
        num +=1
        current = came_from[current]
        current.make_path()
        draw()
    print("Distance: " +str(num)+" Blocks")
    global Distance
    Distance = num
    current.make_end()
    start.make_start()

# Path finding algorithm
def Algorithm(draw, grid, start, end, al):
    t0 = time.time()
    operation = 0
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {cell: float("inf") for row in grid for cell in row}
    g_score[start] = 0
    f_score = {cell: float("inf") for row in grid for cell in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        operation +=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw, start)
            end.make_end()
            t1 = time.time()
            timeTaken = "{:.2f}".format(t1-t0)
            print("Time: "+timeTaken+"s")
            global Time
            Time = timeTaken
            print("Operations: " +str(operation))
            global Operations
            Operations = operation
            print("------------------------")
            return True

        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                if(al == "A*"):
                    f_score[neighbour] = temp_g_score + h(neighbour.get_pos(), end.get_pos())
                elif(al == "BFS"):
                    f_score[neighbour] = h(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    if(al == "Dijkstras"):
                        open_set.put((g_score[neighbour], count, neighbour))
                    else:
                        open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()
        draw()

        if current != start:
            current.make_closed()

    return False

# Layout grid
def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            cell = Cell(i, j, gap, rows)
            grid[i].append(cell)
    
    return grid

# Draw grid
def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

# Instruction Page
def instructions(win):
    win.fill(WHITE)
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
    text14 = font.render("PRESS P TO OPEN & CLOSE", True, RED)

    Tlocations = [[text1, 25], [text2, 90], [text3, 120], [text4, 150], 
    [text5, 180], [text6, 210], [text7, 270], [text8, 300], [text9, 330], 
    [text10, 360], [text11, 410], [text12, 440], [text13, 470], [text14, 600]]
    
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

# Mouse clicked position
def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

# Main program
def main(win, width):
    # Setting main program variables
    ROWS = 25
    grid = make_grid(ROWS, WIDTH)

    start = None
    end = None

    algorithm = "Dijkstras"

    instruct = True

    run = True

    # Display defult algorithm in terminal window
    print("---------------Defult DIJKSTRA'S-----------------")

    # Run program
    while run:
        # Draw on window
        draw(win, grid, ROWS, width, Distance, Time, Operations, algorithm, instruct)
        # Get input on window
        for event in pygame.event.get():
            # Quit
            if event.type == pygame.QUIT:
                run = False
            # Place
            if pygame.mouse.get_pressed()[0] and not instruct:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                cell = grid[row][col]
                if not start and cell != end:
                    start = cell
                    start.make_start()
                elif not end and cell != start:
                    end = cell
                    end.make_end()
                elif cell != end and cell != start:
                    cell.make_barrier()
            # Remove
            elif pygame.mouse.get_pressed()[2] and not instruct:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                cell = grid[row][col]
                cell.reset()
                if cell == start:
                    start = None
                elif cell == end:
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
                # Start path finding algorithm 
                if event.key == pygame.K_SPACE and start and end and not instruct:
                    for row in grid:
                        for cell in row:
                            cell.update_neighbours(grid)
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

                    if(ROWS < 50):
                        walls = [(10,19),(11,19),(12,19),(13,19),
                        (14,19),(10,20),(10,21),(10,22),(10,23),
                        (14,20),(14,21),(14,22),(14,23)]
                    elif(ROWS >= 50):
                        walls = [(23,44),(24,44),(25,44),(26,44),
                        (27,44),(23,45),(23,46),(23,47),(23,48),
                        (27,45),(27,46),(27,47),(27,48)]

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

                    for i in range(ROWS):
                        for j in range(ROWS):
                            cell = grid[i][j]
                            cell.make_barrier()

                    for i in range(ROWS):
                        for j in range(ROWS):
                            if i > 0 and i < ROWS - 1 and j > 0 and j < ROWS - 1:
                                position = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
                                for p in position:
                                    cell = grid[round(ROWS*p)][j]
                                    cell.reset()
                                    cell = grid[i][round(ROWS*p)]
                                    cell.reset()

                                if not grid[i][j].is_barrier():
                                    num = random.randint(0, 5)
                                    if num == 0:
                                        cell = grid[i][j]
                                        cell.make_barrier()
                                        i += 2
                                        j += 2
                    
                    for i in range(6):
                        for j in range(6):
                            if i > 0 and i < ROWS - 1 and j > 0 and j < ROWS - 1:
                                cell = grid[i][j]
                                cell.reset()
                                cell = grid[ROWS - 1 - i][ROWS - 1 - j]
                                cell.reset()
                                
                    cell = grid[1][1]
                    start = cell
                    start.make_start()
                    cell = grid[ROWS - 2][ROWS - 2]
                    end = cell
                    end.make_end()
    # Quit pygame
    pygame.quit()
# Run the main program
main(WIN, WIDTH)