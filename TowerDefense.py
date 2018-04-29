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



>activate cvxpy_env
"""



import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from sklearn.metrics.pairwise import pairwise_distances






class Environment():
    
    def __init__(self,params_dict):
        
        self.PROBLEM_VARIATION = params_dict['PROBLEM_VARIATION']
        
        self.random_seed = params_dict['random_seed']
        self.map_dimensions = params_dict['map_dimensions']#height(y),width(x)
        self.N_obstacles = params_dict['N_obstacles']
        self.N_targets = params_dict['N_targets']
        
        
        self.N_tower_sites = params_dict['N_tower_sites']
        self.N_towers = params_dict['N_towers']
        self.N_tower_kinds = len(self.N_towers)
        
        #If manually specifying map:
        self.coordinates__obstacles = params_dict['coordinates__obstacles'] if\
        'coordinates__obstacles' in params_dict else None
        self.coordinates__targets = params_dict['coordinates__targets'] if\
        'coordinates__targets' in params_dict else None        
        self.coordinates__tower_sites = params_dict['coordinates__tower_sites'] if\
        'coordinates__tower_sites' in params_dict else None        

        self.obstructed_masks = []
        self.distances_towers_targets = []
        self.unobstructed_coverages = []
        self.coordinates__solved_towers = []
        
        self.coverage_matrices = None
        
        self.budget__total_cost = 100
        self.budget__tower_quotas = [999,999]
        self.budget__tower_unit_costs = [2,3]
        
        
        self.N_iterations = 10 #Number iterations of Weighted L1. For this problem, seems to converge pretty fast (only a few iterations)
        
        self.coverage_profile_types = params_dict['coverage_profile_types']
        self.figsize = (18,12)
        
        self.VERBOSE = params_dict['VERBOSE']
        
        def summarize_environment():
            print self.PROBLEM_VARIATION
            print self.random_seed
            print self.map_dimensions
            print self.N_obstacles
            print self.N_targets
            print self.N_tower_sites
            print self.N_towers
            print self.coverage_profile_types
            print self.N_tower_kinds
            print self.budget__total_cost
            print self.budget__tower_quotas
            print self.budget__tower_unit_costs
            print '\n'*5
        summarize_environment()
        
        

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
            self.coordinates__tower_sites = []
            for tk in xrange(self.N_tower_kinds):
                #Each kind of tower will have the correct number of sites placed
                
                coords = []
                while len(coords)<self.N_tower_sites[tk]:
                    x = np.random.randint(0,self.map_dimensions[1]+1,size=1)[0]
                    y = np.random.randint(0,self.map_dimensions[0]+1,size=1)[0]
                    p = (x,y)
                    all_valid = True
                    for rect in self.coordinates__obstacles:
                        if not check_valid_placement(p,rect):
                            all_valid = False
                            break
                    if all_valid:
                        coords.append(p)
                self.coordinates__tower_sites.append(coords)
            
        #If not already mnually specified then do random placement:
        if not self.coordinates__obstacles: place_obstacles()
        if not self.coordinates__targets: place_targets()
        if not self.coordinates__tower_sites: place_allowed_tower_sites()
        print self.coordinates__targets
        print self.coordinates__tower_sites
    
    
    #Different coverage profiles: how well tower X covers (protects) target Y
    def coverage_profiles(self):
        def fixed_radius(A,radius):
            return np.where(A <= radius, 1., 0.)
        def inverse_r(A,K):
            return 1. / (A + K)
        def inverse_r2(A,K):
            return 1. / (A + K)**2
        
        self.unobstructed_coverages = []
        for tk in xrange(self.N_tower_kinds):
            A = self.distances_towers_targets[tk]
            radius = 15. #!!!!!!!
            K = 2.
            coverage_type = self.coverage_profile_types[tk]
            if coverage_type == 'fixed_radius': self.unobstructed_coverages.append(fixed_radius(A,radius))
            elif coverage_type == 'inverse_r': self.unobstructed_coverages.append(inverse_r(A,K))
            elif coverage_type == 'inverse_r2': self.unobstructed_coverages.append(inverse_r2(A,K))
            else: raise Exception('Must specify coverage profile types')
        #print self.unobstructed_coverages
        return self.unobstructed_coverages



    #Calculate the coverage values for each tower to each target, depdending on
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
            for tk in xrange(self.N_tower_kinds):
                mask = np.zeros((self.N_targets,self.N_tower_sites[tk]))
                for i, r1 in enumerate(self.coordinates__targets):
                    for j, r2 in enumerate(self.coordinates__tower_sites[tk]):
                        obstructed = check_obstructed(r1,r2)
#                        print obstructed
#                        print
                        if not obstructed:
                            mask[i][j] = 1.
                self.obstructed_masks += [mask]

        def get_tower_target_distances():
            #pairwise_distances
            for tk in xrange(self.N_tower_kinds):
                D = pairwise_distances(
                        self.coordinates__targets, 
                        self.coordinates__tower_sites[tk])
                self.distances_towers_targets += [D]

        def get_unobstructed_coverages():
            #for tk in xrange(self.N_tower_kinds):
            #    self.unobstructed_coverages += [self.coverage_profiles(ttttt)]
            self.unobstructed_coverages = self.coverage_profiles()
            
            
        def get_final_coverages():
            get_tower_target_distances()
            get_occluded()            
            get_unobstructed_coverages()
            """print '------'
            print self.unobstructed_coverages
            print self.obstructed_masks
            print self.N_tower_kinds
            print '------' """
            return [self.unobstructed_coverages[i] *  self.obstructed_masks[i] for i in xrange(self.N_tower_kinds)]
        

        
        self.coverage_matrices = get_final_coverages()
        print self.obstructed_masks
        print self.distances_towers_targets
        print self.unobstructed_coverages
        print self.coverage_matrices
     



    def visualize_environment(self,env_state):
        """
        Visualize the map environment and solved tower locations.
        env_state = 'solved', 'initial'
        """
        fig=plt.figure(figsize=self.figsize)
        ax=plt.subplot(111)
        #Plot the targets
        plt.plot([i[0] for i in self.coordinates__targets],\
                 [i[1] for i in self.coordinates__targets],\
                 marker='x',markersize=15,linestyle='None',color='k',label='Target')
        #Plot the towers
        tower_colors = ['r','b','g']
        for tk in xrange(self.N_tower_kinds):
            plt.plot([i[0] for i in self.coordinates__tower_sites[tk]],\
                     [i[1] for i in self.coordinates__tower_sites[tk]],\
                     marker='o',markersize=10,linestyle='None',color=tower_colors[tk],alpha=.5,label='Tower {} Sites'.format(tk+1))
        if env_state == 'solved':
            for tk in xrange(self.N_tower_kinds):
                plt.plot([i[0] for i in self.coordinates__solved_towers[tk]],\
                         [i[1] for i in self.coordinates__solved_towers[tk]],\
                         marker='^',markersize=20,linestyle='None',color=tower_colors[tk],label='Tower {} Placed'.format(tk+1))
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
        
        print 'PROBLEM_VARIATION: ', self.PROBLEM_VARIATION


        if self.PROBLEM_VARIATION == 1:
            #The first problem formulation
            #K kinds of towers
            #See more details about problem formulation in the writeup             
            
            #Get a full matrix of the concatenated coverage matrices for 
            #each tower type. THis new matrix has dimensions:
            #(Ntowers) x (sum(potential sites)), where the sum o=is over all tower types
            C = -np.hstack(i for i in self.coverage_matrices)
            print C
            print C.shape            
            
            a = 2. #1.
            tau = 1e-2
            N = sum(i for i in self.N_tower_sites)
#            print N
            w = np.zeros(N)
            for i in xrange(self.N_iterations):
                
                #
                t = cvx.Variable(1)
                #The concatenated vector of occupancies: Concatenated over all
                #of the kinds of towers.
                x = cvx.Variable(N)
                
                #Objective function includes penalty term for non-binary x values
                objective = cvx.Maximize(t - x.T*w)
                
                #Main constraints on 0<=x<=1 and on t
                constraints = [0<=x, x<=1, t<=C*x]
                #And then for each kind of tower, append the constraint that there
                #be exactly N_i towers 
                for tk in xrange(self.N_tower_kinds):
                    before_sum = np.concatenate(([0],np.cumsum(self.N_tower_sites)))[tk]
#                    print before_sum
#                    print before_sum + self.N_tower_sites[tk]
#                    print
                    constraints.append(cvx.sum_entries(
                                    x[before_sum : before_sum + self.N_tower_sites[tk]]
                                    )==self.N_towers[tk])
                
                print 'objective', objective
                print 'constraints', constraints

                cvx.Problem(objective, constraints).solve(verbose=self.VERBOSE)
                x = np.array(x.value).flatten()
                w = a/(tau+np.abs(x))
                plt.figure(figsize=(5,5))
                plt.plot(x,marker='o')
                plt.savefig('histrograms_{}.png'.format(i))         
            
            
            
            
            
            
            
            
        #From the solution x, get the coordinates of those tower sites where we
        #really do want to place a tower
        #use = np.isclose(x,1.)

        for tk in xrange(self.N_tower_kinds):
            before_sum = np.concatenate(([0],np.cumsum(self.N_tower_sites)))[tk]
            y = x[before_sum : before_sum + self.N_tower_sites[tk]]
            inds = np.argsort(y)
            s = y[inds]
            use = np.where(s>.5)[0]
#            print inds
#            print s
#            print use            
            if len(use) != self.N_towers[tk]:
                print 'Solution did not converge properly. Choosing the K best towers.'
                print self.N_towers[tk], len(use)
            use = use[-self.N_towers[tk]:]
            self.coordinates__solved_towers.append([self.coordinates__tower_sites[tk][mm] for mm in inds[use]])
#        print self.coordinates__solved_towers
#        print len(self.coordinates__solved_towers)
    
    
    
        
    
    
    
if __name__ == '__main__':
    # =============================================================================
    #     PARAMETERS
    # =============================================================================
    NT = 28
    params_dict = {'random_seed':3312018,
                   'PROBLEM_VARIATION':1,#2,3,4,5,6 #Which variation of the problem to do
                   'map_dimensions':(30,40),
                   'N_obstacles':18,
                   'N_targets':NT,#266,
                   'target_values':[1.]*NT,
                   'N_tower_sites':[10,13],
                   'N_towers':[3,4],
                   'coverage_profile_types':['fixed_radius','fixed_radius'], #['fixed_radius','inverse_r','inverse_r2']
#                   'coordinates__obstacles':[(0,0,44,44)],
#                   'coordinates__targets':[(0,0),(88,66)],
#                   'coordinates__tower_sites':[(99,107),(377,288)]
                   'VERBOSE':True
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