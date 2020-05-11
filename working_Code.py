
# coding: utf-8

# In[114]:


#In[]
# % matplotlib qt
import numpy as np
import time
import matplotlib.pyplot as plt  #plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Planner
from pqdict import PQDict

def tic():
    return time.time()
def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))


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
    MP = Planner.MyPlanner(boundary, blocks) # TODO: replace this with your own planner implementation

    # Display the environment
    if verbose:
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)  

    # Call the motion planner
    t0 = tic()
    path = path1 #MP.plan(start, goal)
    toc(t0,"Planning")

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
    path=A_Star_Algo(res,epsi,start,goal,'./maps/single_cube.txt')
    success, pathlength = runtest('./maps/single_cube.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')
    
def test_maze(res,epsi,verbose = False):
    print('Running maze test...\n') 
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([12.0, 12.0, 5.0])
    path=A_Star_Algo(res,epsi,start,goal,'./maps/maze.txt')
    success, pathlength = runtest('./maps/maze.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

    
def test_window(res,epsi,verbose = False):
    print('Running window test...\n') 
    start = np.array([0.2, -4.9, 0.2])
    goal = np.array([6.0, 18.0, 3.0])
    path=A_Star_Algo(res,epsi,start,goal,'./maps/window.txt')
    success, pathlength = runtest('./maps/window.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')
    
def test_tower(res,epsi,verbose = False):
    print('Running tower test...\n') 
    start = np.array([2.5, 4.0, 0.5])
    goal = np.array([4.0, 2.5, 19.5])
    path=A_Star_Algo(res,epsi,start,goal,'./maps/tower.txt')
    success, pathlength = runtest('./maps/tower.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

     
def test_flappy_bird(res,epsi,verbose = False):
    print('Running flappy bird test...\n') 
    start = np.array([0.5, 2.5, 5.5])
    goal = np.array([19.0, 2.5, 5.5])
    path=A_Star_Algo(res,epsi,start,goal,'./maps/flappy_bird.txt')
    success, pathlength = runtest('./maps/flappy_bird.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength) 
    print('\n')


def test_room(res,epsi,verbose = False):
    print('Running room test...\n') 
    start = np.array([1.0, 5.0, 1.5])
    goal = np.array([9.0, 7.0, 1.5])
    path=A_Star_Algo(res,epsi,start,goal,'./maps/room.txt')
    success, pathlength = runtest('./maps/room.txt', start, goal, path, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


def test_monza(res,epsi,verbose = False):
    print('Running monza test...\n')
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])
    path=A_Star_Algo(res,epsi,start,goal,'./maps/monza.txt')
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




# In[148]:


#In[]
# %matplotlib inline
if __name__=="__main__":
    res=0.1
    epsi=1.5
    # test_single_cube(res,epsi,True)
    # test_maze(res,epsi,True)
    test_flappy_bird(res,epsi,True)
    # test_monza(res,epsi,True) #issue
    # test_window(res,epsi,True)
    # test_tower(res,epsi,True)
    # test_room(res,epsi,True)

