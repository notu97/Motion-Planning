# ECE276B PR2 SP20 

## Overview
In this assignment, I implemented and compared the performance of search-based and sampling-based motion planning algorithms on several 3-D environments.

### 1. Final_A*_and_RRT*.py (This is my main code) 
This file contains the code for both A*(Self Implemented) and RRT* (using OMPL library). 
Under the main function uncomment the map whose path you want to generate and comment the other map codes. To maps will be generated one after the other.

### 2. ompl_Solution.py
This file contains the independent OMPL Shortest path solution of all the maps using RRT*. Just change the map
path in mapfile under the main function.

### 3. maps
This folder contains the 7 test environments described via a rectangular outer boundary and a list of rectangular obstacles. The start and goal points for each environment are specified in main.py.


