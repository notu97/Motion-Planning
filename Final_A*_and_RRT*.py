
'''
This file contains the code for both A*(Self Implemented) and RRT* (using OMPL library). 
Under the main function uncomment the part whose map you want to generate and comment the other map codes.
'''

#In[]
# % matplotlib qt
import numpy as np
import time
import matplotlib.pyplot as plt  #plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Planner
from pqdict import PQDict
import sys
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
from math import sqrt
import argparse
# import numpy as np
import re
# import time
import matplotlib.pyplot as plt  #plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyrr

def tic():
    return time.time()
def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def ray_intersect_aabb(ray, aabb):
    """Calculates the intersection point of a ray and an AABB
    :param numpy.array ray1: The ray to check.
    :param numpy.array aabb: The Axis-Aligned Bounding Box to check against.
    :rtype: numpy.array
    :return: Returns a vector if an intersection occurs.
        Returns None if no intersection occurs.
    """
    """
    http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    """
    # this is basically "numpy.divide( 1.0, ray[ 1 ] )"
    # except we're trying to avoid a divide by zero warning
    # so where the ray direction value is 0.0, just use infinity
    # which is what we want anyway
    direction = ray[1]
    dir_fraction = np.empty(3, dtype = ray.dtype)
    dir_fraction[direction == 0.0] = 1e16
    dir_fraction[direction != 0.0] = np.divide(1.0, direction[direction != 0.0])

    t1 = (aabb[0,0] - ray[0,0]) * dir_fraction[ 0 ]
    t2 = (aabb[1,0] - ray[0,0]) * dir_fraction[ 0 ]
    t3 = (aabb[0,1] - ray[0,1]) * dir_fraction[ 1 ]
    t4 = (aabb[1,1] - ray[0,1]) * dir_fraction[ 1 ]
    t5 = (aabb[0,2] - ray[0,2]) * dir_fraction[ 2 ]
    t6 = (aabb[1,2] - ray[0,2]) * dir_fraction[ 2 ]


    tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
    tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))

    # if tmax < 0, ray (line) is intersecting AABB
    # but the whole AABB is behind the ray start
    if tmax < 0:
        return None

    # if tmin > tmax, ray doesn't intersect AABB
    if tmin > tmax:
        return None

    # t is the distance from the ray point
    # to intersection

    t = min(x for x in [tmin, tmax] if x >= 0)
    point = ray[0] + (ray[1] * t)
    return point


# In[115]:


def load_map(fname):
    '''
    Loads the bounady and blocks from map file fname.

    boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]

    blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
            ...,
            ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
    '''
    mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
    blockIdx = mapdata['type'] == b'block'
    boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
    blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
    return boundary, blocks


# In[116]:


def draw_map(boundary, blocks, start, goal):
    '''
    Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hb = draw_block_list(ax,blocks)
    hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
    hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(boundary[0,0],boundary[0,3])
    ax.set_ylim(boundary[0,1],boundary[0,4])
    ax.set_zlim(boundary[0,2],boundary[0,5])  
    return fig, ax, hb, hs, hg


# In[117]:


def draw_block_list(ax,blocks):
    '''
    Subroutine used by draw_map() to display the environment blocks
    '''
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
    f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
    clr = blocks[:,6:]/255
    n = blocks.shape[0]
    d = blocks[:,3:6] - blocks[:,:3] 
    vl = np.zeros((8*n,3))
    fl = np.zeros((6*n,4),dtype='int64')
    fcl = np.zeros((6*n,3))
    for k in range(n):
        vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
        fl[k*6:(k+1)*6,:] = f + k*8
        fcl[k*6:(k+1)*6,:] = clr[k,:]

    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
        pc.set_facecolor(fcl)
        h = ax.add_collection3d(pc)
        return h
    #In[]


# In[118]:


def runtest(mapfile, start, goal, path1, verbose = True):
    '''
    This function:
    * load the provided mapfile
    * creates a motion planner
    * plans a path from start to goal
    * checks whether the path is collision free and reaches the goal
    * computes the path length as a sum of the Euclidean norm of the path segments
    '''
    # Load a map and instantiate a motion planner
    boundary, blocks = load_map(mapfile)
    # MP = Planner.MyPlanner(boundary, blocks) # TODO: replace this with your own planner implementation

    # Display the environment
    if verbose:
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)  

    # Call the motion planner
    # t0 = tic()
    path = path1 #MP.plan(start, goal)
    # toc(t0,"Planning")

    # Plot the path
    if verbose:
        ax.plot(path[:,0],path[:,1],path[:,2],'r-')
        plt.show()

    # TODO: You should verify whether the path actually intersects any of the obstacles in continuous space
    # TODO: You can implement your own algorithm or use an existing library for segment and 
    #       axis-aligned bounding box (AABB) intersection 
    collision = False
    goal_reached = sum((path[-1]-goal)**2) <= 0.1
    success = (not collision) and goal_reached
    pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
    return success, pathlength


# In[143]:


def test_single_cube(res,epsi,verbose = False):
    print('Running single cube test...\n') 
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 5.5])
    t0 = tic()
    path=A_Star_Algo(res,epsi,start,goal,'./maps/single_cube.txt')
    toc(t0,"Planning")
    success, pathlength = runtest('./maps/single_cube.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

    
def test_maze(res,epsi,verbose = False):
    print('Running maze test...\n') 
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([12.0, 12.0, 5.0])
    t0 = tic()
    path=A_Star_Algo(res,epsi,start,goal,'./maps/maze.txt')
    toc(t0,"Planning")
    success, pathlength = runtest('./maps/maze.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

    
def test_window(res,epsi,verbose = False):
    print('Running window test...\n') 
    start = np.array([0.2, -4.9, 0.2])
    goal = np.array([6.0, 18.0, 3.0])
    t0 = tic()
    path=A_Star_Algo(res,epsi,start,goal,'./maps/window.txt')
    toc(t0,"Planning")
    success, pathlength = runtest('./maps/window.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')
    
def test_tower(res,epsi,verbose = False):
    print('Running tower test...\n') 
    start = np.array([2.5, 4.0, 0.5])
    goal = np.array([4.0, 2.5, 19.5])
    t0 = tic()
    path=A_Star_Algo(res,epsi,start,goal,'./maps/tower.txt')
    toc(t0,"Planning")
    success, pathlength = runtest('./maps/tower.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

     
def test_flappy_bird(res,epsi,verbose = False):
    print('Running flappy bird test...\n') 
    start = np.array([0.5, 2.5, 5.5])
    goal = np.array([19.0, 2.5, 5.5])
    t0 = tic()
    path=A_Star_Algo(res,epsi,start,goal,'./maps/flappy_bird.txt')
    toc(t0,"Planning")
    success, pathlength = runtest('./maps/flappy_bird.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength) 
    print('\n')


def test_room(res,epsi,verbose = False):
    print('Running room test...\n') 
    start = np.array([1.0, 5.0, 1.5])
    goal = np.array([9.0, 7.0, 1.5])
    t0 = tic()
    path=A_Star_Algo(res,epsi,start,goal,'./maps/room.txt')
    toc(t0,"Planning")
    success, pathlength = runtest('./maps/room.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


def test_monza(res,epsi,verbose = False):
    print('Running monza test...\n')
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])
    t0 = tic()
    path=A_Star_Algo(res,epsi,start,goal,'./maps/monza.txt')
    toc(t0,"Planning")
    success, pathlength = runtest('./maps/monza.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


# In[120]:


def visualize_3D(gridmap):
    x_max,y_max,z_max = gridmap.shape

    x_idx = np.arange(0,x_max)
    y_idx = np.arange(0,y_max)
    z_idx = np.arange(0,z_max)

    x_ids = np.tile(x_idx,(1,y_max*z_max)).reshape(1,-1,order='F')
    y_ids = np.tile(y_idx,(x_max,z_max)).reshape(1,-1,order='F')
    z_ids = np.tile(z_idx,(y_max*x_max,1)).reshape(1,-1,order='F')
    values = gridmap.reshape(1,-1,order='F')[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(x_ids, y_ids, z_ids, c=values, cmap=plt.hot(), alpha=0.01)
    fig.colorbar(img)
    plt.show()


# In[146]:


# boundary, blocks = load_map(mapfile)
# './maps/single_cube.txt', start, goal, verbose
def build_env(mapfile,res,start,goal):
    
    boundary, blocks = load_map(mapfile)
    
    start1=  np.ceil(((start-boundary[0][0:3])/res)+1)
    goal1= np.ceil(((goal-boundary[0][0:3])/res)+1)
    
    x_n= int(np.ceil(int( ( (boundary[0][3]-boundary[0][0])/res)+1 )))
    y_n= int(np.ceil(int( ( (boundary[0][4]-boundary[0][1])/res)+1 )))
    z_n= int(np.ceil(int( ( (boundary[0][5]-boundary[0][2])/res)+1 )))
    
    world= 0*np.ones((  x_n,y_n,z_n ))
    
    world[:,:,0]=np.inf
    world[0,:,:]=np.inf
    world[:,0,:]=np.inf
    print(world.shape)
    blocks[:,0]=blocks[:,0]-boundary[0][0]
    blocks[:,1]=blocks[:,1]-boundary[0][1]
    blocks[:,2]=blocks[:,2]-boundary[0][2]
    
    blocks[:,3]=blocks[:,3]-boundary[0][0]
    blocks[:,4]=blocks[:,4]-boundary[0][1]
    blocks[:,5]=blocks[:,5]-boundary[0][2]
        
    grid_block=(np.ceil(((blocks)/res)+1)).astype(np.int)
    
    for i in np.arange(len(blocks)):
        world[ grid_block[i][0]-1 : grid_block[i][3]+1,
               grid_block[i][1]-1 : grid_block[i][4]+1,
               grid_block[i][2]-1 : grid_block[i][5]+1] =np.inf

    return world, start1, goal1

    


# In[122]:


def children_of(cell):
    # numofdirs = 26
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1)
    
    return  (cell.reshape(3,1)+dR)


# In[123]:


def huristic(cell,goal,ep):
    h= round((ep*np.sqrt(np.sum((goal-cell)**2))),3)
    return h


# In[124]:


def c_ij(cell_i,cell_j):
    dist= round((np.sqrt(np.sum((cell_i-cell_j)**2))),3)
    return dist


# In[125]:


# OPEN_huri.update()
# def cost_Hur(OPEN_huri,epsi):
#     for d in OPEN_huri:
#         OPEN_huri.updateitem(d, new_world[d]+ epsi*(huristic(d,goal,epsi)))
#     return OPEN_huri


# In[126]:


def extract_path(Parent,start, goal, mapfile,res):
    boundary, blocks = load_map(mapfile)
    
    a=goal
    path=goal
    while not((a[0]==start[0])and(a[1]==start[1])and(a[2]==start[2])):
        b=np.array(Parent[tuple(a)])
#         print(b)
        path=np.vstack((path,b))
        a=b
    path=((path-1)*res)
    path[:,0]=path[:,0]+boundary[0][0]
    path[:,1]=path[:,1]+boundary[0][1]
    path[:,2]=path[:,2]+boundary[0][2]
    path=np.flip(path,0)
    return path
    


# In[127]:


def A_Star_Algo(res,epsi,start1,goal1,mapfile):

    new_world,start,goal = build_env(mapfile,res,start1,goal1) # Get the environment
    print(start,goal)
    x,y,z=np.shape(new_world)
    cost_grid=np.inf*np.ones((x,y,z))

    cost_grid[ tuple(start.astype(np.int)) ]=0

    OPEN = PQDict({tuple(start.astype(np.int)): cost_grid[tuple(start.astype(np.int))] }) # OPEN List
    CLOSED=PQDict() # Closed List
    PARENT={}
    itr=1

    while not(bool(CLOSED.get(tuple(goal) ))):
        print(itr)
    #     OPEN_huri=OPEN.copy()
    #     OPEN_huri=cost_Hur(OPEN,epsi)
    #     i=OPEN_huri.popitem()
        i=OPEN.popitem()

    #     del OPEN_huri
        CLOSED.additem(i[0],i[1])
        # i[0] location i[1] cost
        child_i= children_of(np.array([i[0]])[0])
        for j in (child_i.T):
    #         print(j)
            if(j[0]<x and j[1]<y and j[2]<z and j[0]>=0 and j[1]>=0 and j[2]>=0 and (new_world[tuple(j)]==0)):
                if (not(bool(CLOSED.get( tuple(j) )))):

                    if (cost_grid[tuple(j)]>(cost_grid[tuple(i[0])]+ c_ij(i[0],j)) ):
                        cost_grid[tuple(j)]=(cost_grid[tuple(i[0])]+ c_ij(i[0],j))
                        PARENT[tuple(j)]=tuple(i[0])

                        if ( bool(OPEN.get( tuple(j))) ):
                            OPEN.updateitem(tuple(j), cost_grid[tuple(j)]+huristic((j),list(goal),epsi ))
                        else:
                            OPEN.additem(tuple(j), cost_grid[tuple(j)]+huristic((j),list(goal),epsi ) )
    #         else:
    #             continue
        itr=itr+1           
    path=extract_path(PARENT,start,goal,mapfile,res)
    
    return path
#--------OMPL RRT--------------------------
def isValid(state):
    # print("validity",boundary[0])
    # print("blocks",blocks)
    bound=boundary[0]
    # print('jjjjjjjjjjjjjjjjjjjjjjjjjjjj',bound)
    # print('hhh:',state[0],state[1],state[2])
    if( state[0] > bound[0] and state[0] < bound[3] and \
        state[1] > bound[1] and state[1] < bound[4] and \
        state[2] > bound[2] and state[2] < bound[5] ):
    
        # print('IN BOUND')
        # print('inside bound',state[0],state[1],state[2])
        valid=True
        for k in range(blocks.shape[0]):
            # print(k,state[0],state[1],state[2])
            # print(blocks[k,:])
            if( (state[0] > blocks[k,0] and state[0] < blocks[k,3]) and (state[1] > blocks[k,1] and state[1] < blocks[k,4]) and (state[2] > blocks[k,2] and state[2] < blocks[k,5]) ):
                # print('INValid: ' ,state[0],state[1],state[2],k)
                valid =  True # State lies within kth block
                # return False
                break
            else:
                valid = False # state doesn't lie in kth block
                continue
                # return True
        # print('valid',valid)
        if(valid==True):
            return False # State is INVALID
        else:
            return True # State is VALID
    else:
        # print('Out of bound')
        return False # State is Invalid

  


def getPathLengthObjective(si):
    return ob.PathLengthOptimizationObjective(si)


class MyMotionValidator(ob.MotionValidator):
    def __init__(self, si):
        super(MyMotionValidator, self).__init__(si)
        self.si=si
    def checkMotion(self,s1, s2):
        # print(boundary[0])
        # print('hi....',self.si)
        # print('S1',s1[0],s1[1],s1[2])
        # print('S2',s2[0],s2[1],s2[2])
        # # valid=False

        point1 = np.array([s1[0],s1[1],s1[2]])
        point2 = np.array([s2[0],s2[1],s2[2]])
        line_seg = np.array([point1,point2])
        ray = pyrr.ray.create_from_line(line_seg)

        # line = np.array( s1[0], s1[1], s1[2], s2[0], s2[1], s2[2] )
        # pyrr.ray.create_from_line(line)
        
        # flag=True
        # statevalid=False
        b=[]
        for i in range(len(blocks)):
            # stateInvalid=False
            # valid=False
            aabb = pyrr.aabb.create_from_bounds(blocks[i][0:3],blocks[i][3:6])        
            # print('RAY', ray)
            # print('AABB: ',aabb )
            # print('RAY:  ', ray )
            result1 = (ray_intersect_aabb(ray,aabb))
            # result2 = (ray_intersect_aabb(-ray,aabb))
            # print(result1) # None= No intersection
            # print('S2 validity',isValid(s2))
            # if(isValid(s2)):
                
            if((np.all(result1)==None)):  #or(np.all(result2)!=None)) :
                b.append(True)
                # print('There is no intersection---------##########################################', i+1)
                # statevalid=True # State is either intersecting with a block or lies inside a block
                # continue
            else:
                # print('There is intersection with: ', i+1)
                b.append(False)
                # statevalid=False # State is valid
                # break
            # else:
            #     return False
        flag=True
        for i in b:
            # print(i)
            flag=flag and i
        # print('Final Flag: ', flag)
        # print('list b: ',b)
        if(flag==True):
            return True
        elif(flag==False):
            return False


def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj


# Keep these in alphabetical order and all lower case
def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")


# Keep these in alphabetical order and all lower case
def allocateObjective(si, objectiveType):
    # if objectiveType.lower() == "pathclearance":
    #     return getClearanceObjective(si)
    if objectiveType.lower() == "pathlength":
        return getPathLengthObjective(si)
    # elif objectiveType.lower() == "thresholdpathlength":
    #     return getThresholdPathLengthObj(si)
    # elif objectiveType.lower() == "weightedlengthandclearancecombo":
    #     return getBalancedObjective1(si)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")



def plan(runTime, plannerType, objectiveType, boundary, blocks, st,go,mapfile):
    # Construct the robot state space in which we're planning. We're
    # planning in [0,1]x[0,1], a subset of R^3.
    space = ob.RealVectorStateSpace(3)
    # space = ob.SE3StateSpace()
    # space=state(sp)
    bounds = ob.RealVectorBounds(3)

    # bd=np.empty(6)

    # print('plan func Boundary',boundary)
    bd=boundary.astype(np.float)

    bounds.low[0]=bd[0]
    bounds.high[0]=bd[3]

    bounds.low[1]=bd[1]
    bounds.high[1]=bd[4]

    bounds.low[2]=bd[2]
    bounds.high[2]=bd[5]

    # Set the bounds of space to be in [0,1].
    space.setBounds(bounds)
    
    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)
    
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isValid))
    

    mv = MyMotionValidator(si)    
    si.setMotionValidator(mv)

    
    si.setup()

    # Set our robot's starting state to be the bottom-left corner of
    # the environment, or (0,0).
    start = ob.State(space)
    start[0] = st[0]
    start[1] = st[1]
    start[2] = st[2]

    goal = ob.State(space)
    goal[0] = go[0]
    goal[1] = go[1]
    goal[2] = go[2]

    # Create a problem instance
    pdef = ob.ProblemDefinition(si)
    
    # Set the start and goal states
    pdef.setStartAndGoalStates(start, goal)

    # Create the optimization objective specified by our command-line argument.
    # This helper function is simply a switch statement.
    pdef.setOptimizationObjective(allocateObjective(si, objectiveType))

    # Construct the optimal planner specified by our command line argument.
    # This helper function is simply a switch statement.
    optimizingPlanner = allocatePlanner(si, plannerType)

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()
    
    count=0
    print(pdef.hasApproximateSolution())
    # attempt to solve the planning problem in the given runtime
    t0=tic()
    while(not pdef.hasExactSolution()) :
        print(count)
        count=count+1
        # solved = 
        optimizingPlanner.solve(runTime)
        # print(pdef.hasOptimizedSolution())
        if(pdef.hasExactSolution()): 
            toc(t0,"Planning")
            print('Solution Found.')
            path_str=pdef.getSolutionPath().printAsMatrix()
            q=re.findall(r"[-+]?\d*\.\d+|\d+", path_str) # extract floats
            w=[float(i) for i in q]   
            path=np.array( w[0:3] )
            #   while(np.not_equal(w[-3:], go) ):
            for i in np.arange(3,len(w),3):
                # print(i,i+3)
                path=np.vstack((path,w[i:i+3] ))

            # print(type(start))
            print('Path going to runtest',path)
            # return path

            runtest(mapfile, st, go, path, verbose = True)
        else:
            print("No solution found.")
            # return None
        




#---------OMPL RRT ENDS---------------------
# In[148]:


#In[]
# %matplotlib inline
if __name__=="__main__":
    res=0.1
    epsi=10
    #----------------------------------
    parser = argparse.ArgumentParser(description='Optimal motion planning demo program.')

    # Add a filename argument
    parser.add_argument('-t', '--runtime', type=float, default=1.0, help=\
        '(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.')

    parser.add_argument('-p', '--planner', default='RRTstar', \
        choices=['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', \
        'SORRTstar'], \
        help='(Optional) Specify the optimal planner to use, defaults to RRTstar if not given.')

    parser.add_argument('-o', '--objective', default='PathLength', \
        choices=['PathClearance', 'PathLength', 'ThresholdPathLength', \
        'WeightedLengthAndClearanceCombo'], \
        help='(Optional) Specify the optimization objective, defaults to PathLength if not given.')

    parser.add_argument('-f', '--file', default=None, \
        help='(Optional) Specify an output path for the found solution path.')

    parser.add_argument('-i', '--info', type=int, default=0, choices=[0, 1, 2], \
        help='(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG.' \
        ' Defaults to WARN.')

    # Parse the arguments
    args = parser.parse_args()
    # print(args.runtime, args.planner, args.objective, args.file)
    # Check that time is positive
    if args.runtime <= 0:
        raise argparse.ArgumentTypeError(
            "argument -t/--runtime: invalid choice: %r (choose a positive number greater than 0)" \
            % (args.runtime,))

    # Set the log level
    if args.info == 0:
        ou.setLogLevel(ou.LOG_WARN)
    elif args.info == 1:
        ou.setLogLevel(ou.LOG_INFO)
    elif args.info == 2:
        ou.setLogLevel(ou.LOG_DEBUG)
    else:
        ou.OMPL_ERROR("Invalid log-level integer.")
    #---------------------SINGLE_CUBE-------------
    
    test_single_cube(res,epsi,True) # A* Solution
    # OMPL Solution
    mapfile='./maps/single_cube.txt'
    boundary,blocks=load_map(mapfile)
    start = np.array([2.3, 2.3, 1.3]) # Singlr_cube #Working
    goal = np.array([7.0, 7.0, 5.5])
    plan(10, args.planner, args.objective, boundary[0], blocks,start,goal,mapfile)
    
    #---------------------MAZE--------------------------------------------------------------
    
    # test_maze(res,epsi,True) # 17 minutes # A* Solution
    ## OMPL Solution
    # mapfile='./maps/maze.txt'
    # boundary,blocks=load_map(mapfile)
    # start = np.array([0.0, 0.0, 1.0]) # Maze # Working 1301.2197613716125 sec.
    # goal = np.array([12.0, 12.0, 5.0])
    # plan(10, args.planner, args.objective, boundary[0], blocks,start,goal,mapfile)
    
    #----------------------FLAPPY_BIRD-------------------------------------------------------------
    
    # test_flappy_bird(res,epsi,True) # A* Solution
    # # OMPL Solution
    # mapfile='./maps/flappy_bird.txt' 
    # boundary,blocks=load_map(mapfile)
    # start = np.array([0.5, 2.5, 5.5]) # time=30-Flappy #Working
    # goal = np.array([19, 2.5, 5.5])
    # plan(10, args.planner, args.objective, boundary[0], blocks,start,goal,mapfile)
    
    #----------------------------MONZA-------------------------------------------------------
    
    # test_monza(res,epsi,True) #issue # A* Solution
    ## OMPL Solution
    # mapfile='./maps/monza.txt'
    # boundary,blocks=load_map(mapfile)
    # start = np.array([0.5, 1.0, 4.9]) # Monza   1793.9816465377808 sec Working
    # goal = np.array([3.8, 1.0, 0.1])
    # plan(10, args.planner, args.objective, boundary[0], blocks,start,goal,mapfile)
    
    #----------------------------WINDOW-------------------------------------------------------    
    
    # test_window(res,epsi,True) # A* Solution
    # ## OMPL Solution
    # mapfile='./maps/window.txt'
    # boundary,blocks=load_map(mapfile)
    # start = np.array([0.2, -4.9, 0.2]) #time=150 window #Working
    # goal = np.array([6.0, 18.0, 3.0])
    # plan(10, args.planner, args.objective, boundary[0], blocks,start,goal,mapfile)
    
    #--------------------------TOWER---------------------------------------------------------
    
    # test_tower(res,epsi,True) # A* Solution
    # # OMPL Solution
    # mapfile='./maps/tower.txt'
    # boundary,blocks=load_map(mapfile)
    # start = np.array([2.5, 4.0, 0.5]) # Tower #Working
    # goal = np.array([4.0, 2.5, 19.5])
    # plan(10, args.planner, args.objective, boundary[0], blocks,start,goal,mapfile)
    
    #---------------------------ROOM--------------------------------------------------------
    
    # test_room(res,epsi,True) # A* Solution
    ## OMPL Solution
    # mapfile='./maps/room.txt'
    # boundary,blocks=load_map(mapfile)
    # start = np.array([1, 5, 1.5]) # Room 
    # goal = np.array([9, 7, 1.5]) # change to pdef.hasApproximateSolution()
    # plan(10, args.planner, args.objective, boundary[0], blocks,start,goal,mapfile)
    

