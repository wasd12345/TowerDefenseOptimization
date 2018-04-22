# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:19:36 2018

@author: GK




EE 4650 Convex Optimization Final Project:

Tower defense placement as a combinatorial optimization problem:

Look at several different variations of tower defense placement optimization,
i.e. how to optimally place defensive towers to protect various other point 
structures.


Several variations:
    - with obstacles that block line of sight
    - different kinds of towers with different coverage functions [e.g. 
    different range or different range profile]
    - different costs of towers, with a fixed overall budget
    - different quotas on kinds of towers
    - different values of structures you want to protect, maximizing the 
    expected value of saved structures
    - others to come.

"""



import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from sklearn.metrics.pairwise import pairwise_distances






class Environment():
    
    def __init__(self,params_dict):
        self.random_seed = params_dict['random_seed']
        self.map_dimensions = params_dict['map_dimensions']#height(y),width(x)
        self.N_obstacles = params_dict['N_obstacles']
        self.N_targets = params_dict['N_targets']
        
        
        self.N_tower_sites = params_dict['N_tower_sites']
        self.N_towers = params_dict['N_towers']
        
        #If manually specifying map:
        self.coordinates__obstacles = params_dict['coordinates__obstacles'] if\
        'coordinates__obstacles' in params_dict else None
        self.coordinates__targets = params_dict['coordinates__targets'] if\
        'coordinates__targets' in params_dict else None        
        self.coordinates__tower_sites = params_dict['coordinates__tower_sites'] if\
        'coordinates__tower_sites' in params_dict else None        

        self.obstructed_mask = None
        self.distances_towers_targets = None
        self.unobstructed_coverages = None
        self.coordinates__solved_towers = None
        
        self.coverage_matrix = None
        
        self.budget__total_cost = 100
        self.budget__tower_quotas = [999,999]
        self.budget__tower_unit_costs = [2,3]
        self.N_tower_kinds = len(self.budget__tower_quotas)
        
        self.N_iterations = 10 #Number iterations of Weighted L1. For this problem, seems to converge pretty fast (only a few iterations)
        
        self.coverage_profile_type = params_dict['coverage_profile_type']
        self.figsize = (18,12)
        
        
        
        

    #generate random map, obstacles, targets, allowed tower sites
    def initialize_random_map(self):
        np.random.seed(self.random_seed)
        def place_obstacles():
            """
            Place various obstacles.
            E.g. put in rectangles which block the line of site of the towers.
            """
            #Randomly generate different sized rectangles
            #Soem may overlap, which gives more variety in shape of obstacles
            xvals = np.random.randint(0,self.map_dimensions[1],size=self.N_obstacles)
            yvals = np.random.randint(0,self.map_dimensions[0],size=self.N_obstacles)
            lower_left = zip(xvals,yvals)
            rects = []
            for LL in lower_left:
                x = LL[0]
                y = LL[1]
                wmax = self.map_dimensions[1] - x
                w = np.random.randint(0,wmax,size=1)[0]
                hmax = self.map_dimensions[0] - y
                h = np.random.randint(0,hmax,size=1)[0]
                rects += [(x,y,w,h)]
            self.coordinates__obstacles = rects
#            #Generate randomly, but for now just use manually specified:
#            rect1 = [(10,10),(20,10),(10,20),(20,20)]
#            rect2 = [(50,50),(70,50),(50,100),(70,100)]
#            rect3 = [(150,50),(70,50),(150,100),(70,100)]
#            self.coordinates__obstacles = [rect1,rect2,rect3]
                
        def check_valid_placement(p,rect):
            x,y,w,h = rect
            valid = ~(x<=p[0] and y<=p[1] and p[0]<=x+w  and p[1]<=y+h)
            return valid
        
        
        
        
        def place_targets():
            """
            Place the target locations
            """
            #xvals = np.random.randint(0,self.map_dimensions[1]+1,size=self.N_targets)
            #yvals = np.random.randint(0,self.map_dimensions[0]+1,size=self.N_targets)
            coords = []
            while len(coords)<self.N_targets:
                x = np.random.randint(0,self.map_dimensions[1]+1,size=1)[0]
                y = np.random.randint(0,self.map_dimensions[0]+1,size=1)[0]
                p = (x,y)
                all_valid = True
                for rect in self.coordinates__obstacles:
                    if not check_valid_placement(p,rect):
                        all_valid = False
                        break
                if all_valid:
                    coords +=[p]
            self.coordinates__targets = coords
            
        def place_allowed_tower_sites():
            """
            Place the potential tower locations.
            These are the locations where towers can potentially be placed.
            Not every location is necesarily used
            (only when N_tower_sites = N_towers).
            THe optimization problem is to determine which of these possible 
            locations to use.
            """
            coords = []
            while len(coords)<self.N_tower_sites:
                x = np.random.randint(0,self.map_dimensions[1]+1,size=1)[0]
                y = np.random.randint(0,self.map_dimensions[0]+1,size=1)[0]
                p = (x,y)
                all_valid = True
                for rect in self.coordinates__obstacles:
                    if not check_valid_placement(p,rect):
                        all_valid = False
                        break
                if all_valid:
                    coords +=[p]
            self.coordinates__tower_sites = coords            
            
        #If not already mnually specified then do random placement:
        if not self.coordinates__obstacles: place_obstacles()
        if not self.coordinates__targets: place_targets()
        if not self.coordinates__tower_sites: place_allowed_tower_sites()
        print self.coordinates__targets
        print self.coordinates__tower_sites


    #Different coverage profiles: how well tower X covers (protects) target Y
    def coverage_profiles(self):
        def fixed_radius():
            radius = 10.
            return np.where(self.distances_towers_targets <= radius, 1., 0.)
        def inverse_r():
            K = 50.
            return 1. / (self.distances_towers_targets + K)
        def inverse_r2():
            K = 50.
            return 1. / (self.distances_towers_targets + K)**2
        
        if self.coverage_profile_type == 'fixed_radius': return fixed_radius()
        if self.coverage_profile_type == 'inverse_r': return inverse_r()
        if self.coverage_profile_type == 'inverse_r2': return inverse_r2()
            

    #Calculate the voerage values for each tower to each target, depdending on
    #coverage profiles and accountign for obstructions
    def get_tower_target_coverages(self):

        def check_obstructed(r1,r2):
            """
            return True if r1 - r2 line of sight is obstrucetd; oherwise False
            """            
            
            if r1==r2:
                return False
            
            #Densely sample line connecting r1 and r2.
            #If any of those sampled points is inside the rectangle, then the 
            #line of sight intersects the rectangle and the tower's view is
            #obstructed.
            NP = 1000
            sampled_x = np.linspace(r1[0],r2[0],NP)
            sampled_y = np.linspace(r1[1],r2[1],NP)
            for x,y,w,h in self.coordinates__obstacles:
                for pt in xrange(NP):
                    if (sampled_x[pt] > x) and (sampled_x[pt] < x+w) and \
                    (sampled_y[pt] > y) and (sampled_y[pt] < y+h):
                        return True
            return False        
            
        def get_occluded():
            self.obstructed_mask = np.zeros((self.N_targets,self.N_tower_sites))
            #for r1,r2 in (self.coordinates__targets, self.coordinates__tower_sites):
            for i, r1 in enumerate(self.coordinates__targets):
                for j, r2 in enumerate(self.coordinates__tower_sites):
                    obstructed = check_obstructed(r1,r2)
#                    print obstructed
#                    print
                    if not obstructed:
                        self.obstructed_mask[i][j] = 1.

        def get_tower_target_distances():
            #pairwise_distances
            self.distances_towers_targets = pairwise_distances(
                    self.coordinates__targets, 
                    self.coordinates__tower_sites)

        def get_unobstructed_coverages():
            self.unobstructed_coverages = self.coverage_profiles()
            
            
        def get_final_coverages():
            get_tower_target_distances()
            get_occluded()            
            get_unobstructed_coverages()
            return self.unobstructed_coverages *  self.obstructed_mask
        

        
        self.coverage_matrix = get_final_coverages()
        print self.obstructed_mask
        print self.distances_towers_targets
        print self.unobstructed_coverages
        print self.coverage_matrix
     



    def visualize_environment(self,env_state):
        """
        Visualize the map environment and solved tower locations.
        env_state = 'solved', 'initial'
        """
        fig=plt.figure(figsize=self.figsize)
        ax=plt.subplot(111)
        plt.plot([i[0] for i in self.coordinates__targets],\
                 [i[1] for i in self.coordinates__targets],\
                 marker='x',markersize=15,linestyle='None',color='r',label='Target')
        plt.plot([i[0] for i in self.coordinates__tower_sites],\
                 [i[1] for i in self.coordinates__tower_sites],\
                 marker='o',markersize=10,linestyle='None',color='k',label='Tower Sites')
        if env_state == 'solved':
            plt.plot([i[0] for i in self.coordinates__solved_towers],\
                     [i[1] for i in self.coordinates__solved_towers],\
                     marker='^',markersize=15,linestyle='None',color='g',label='Towers Placed')
        for x,y,w,h in self.coordinates__obstacles:
            r = plt.Rectangle((x,y),w,h,fc='c')
            ax.add_patch(r)
        plt.xlim(0,self.map_dimensions[1])
        plt.ylim(0,self.map_dimensions[0])
        plt.legend(numpoints=1,loc='best')
        savename = 'SolvedMap.png' if env_state == 'solved' else 'InitialMap.png'
        plt.savefig(savename)
        

    #Get the optimal placement of the towers at the allowed tower sites
    def solve_environment(self):
        """
        The placement problem is a combinatorial optimization problem, so in general
        will be NP hard. But we can use convex relaxation to reformulate it as 
        a convex optimization problem. Then we can use the Iterated Weighted 
        L1 Heuristic to encourage sparse solutions and recover a viable solution 
        to the placement problem.
        """
        a = 2. #1.
        tau = 1e-2
        m, n = self.coverage_matrix.shape
        w = np.zeros(n)
        for i in xrange(self.N_iterations):
            x = cvx.Variable(n)
            t = cvx.Variable(1)
        
            objective = cvx.Maximize(t - x.T*w)
            constraint = [0<=x, x<=1, t<=self.coverage_matrix*x, cvx.sum_entries(x)==self.N_towers]
            cvx.Problem(objective, constraint).solve(verbose=True)
            x = np.array(x.value).flatten()
            w = a/(tau+np.abs(x))
            plt.figure(figsize=(5,5))
            plt.plot(x,marker='o')
            plt.savefig('histrograms_{}.png'.format(i))
            
        #From the solution x, get the coordinates of those tower sites where we
        #really do want to place a tower
        #use = np.isclose(x,1.)
        inds = np.argsort(x)
        s = x[inds]
        use = np.where(s>.5)[0]
#        print inds
#        print s
#        print use
        if len(use) != self.N_towers:
            print 'Solution did not converge properly. Choosing the K best towers.'
            print self.N_towers, len(use)
        use = use[-self.N_towers:]
        self.coordinates__solved_towers = [self.coordinates__tower_sites[mm] for mm in inds[use]]
        
    
    
    
        
    
    
    
if __name__ == '__main__':
    # =============================================================================
    #     PARAMETERS
    # =============================================================================
    params_dict = {'random_seed':3312018,
                   'map_dimensions':(30,40),
                   'N_obstacles':8,
                   'N_targets':8,#266,
                   'N_tower_sites':11,#302,
                   'N_towers':2,
                   'coverage_profile_type':'fixed_radius',
#                   'coordinates__obstacles':[(0,0,44,44)],
#                   'coordinates__targets':[(0,0),(88,66)],
#                   'coordinates__tower_sites':[(99,107),(377,288)]
                   }
        
    
    
    # =============================================================================
    #     MAIN
    # =============================================================================
    env = Environment(params_dict)
    env.initialize_random_map()
    env.visualize_environment('initial')
    env.get_tower_target_coverages()

#    f=fffff    
    env.solve_environment()
    env.visualize_environment('solved')