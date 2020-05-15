# ECE276B PR2 SP20 

## Overview
In this assignment, you will implement and compare the performance of search-based and sampling-based motion planning algorithms on several 3-D environments.

### 1. Final_A*_and_RRT*.py (This is my main code) 
This file contains the code for both A*(Self Implemented) and RRT* (using OMPL library). 
Under the main function uncomment the map whose path you want to generate and comment the other map codes. To maps will be generated one after the other.

### 2. ompl_Solution.py
This file contains the independent OMPL Shortest path solution of all the maps using RRT*. Just change the map
path in mapfile under the main function.

### 1. main.py (Reference file: I Didn't use it)
This file contains examples of how to load and display the 7 environments and how to call a motion planner and plot the planned path. Feel free to modify this file to fit your specific goals for the project. In particular, you should certainly replace Line 104 with a call to a function which checks whether the planned path intersects the boundary or any of the blocks in the environment.

### 2. Planner.py (Simple planner implemented. I Didn't use it)
This file contains an implementation of a baseline planner. The baseline planner gets stuck in complex environments and is not very careful with collision checking. Feel free to modify this file in any way necessary for your own implementation.

### 3. maps
This folder contains the 7 test environments described via a rectangular outer boundary and a list of rectangular obstacles. The start and goal points for each environment are specified in main.py.


