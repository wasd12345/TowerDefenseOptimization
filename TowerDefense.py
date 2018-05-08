# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:19:36 2018

@author: GK


Python 2.7.14 |Anaconda custom (64-bit)| (default, Nov  8 2017, 13:40:45) [MSC v
.1500 64 bit (AMD64)] on win32
cvxpy.__version__ = 0.4.8
numpy.__version__ = 1.12.1
sklearn.__version__ = 0.19.1
matplotlib.__version__ = 2.1.2



EE 4650 Convex Optimization Final Project:

Tower defense placement as a combinatorial optimization problem
(More generally, a combinatorial resource allocation problem):

Look at several different variations of tower defense placement optimization,
i.e. how to optimally place defensive towers to protect various other point 
structures.



Several variations:
    - with obstacles that block line of sight
    - different kinds of towers with different coverage functions [e.g. 
    different ranges / different range profile]
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

        self.random_seed = params_dict['random_seed']
        
        #Problem formulation        
        self.objective_type = params_dict['objective_type']
        self.penalty_type = params_dict['penalty_type']
        self.constraints__type = params_dict['constraints__type']
        
        #If doing the variation of the problem with cost budgets:
        self.budget__total_cost = params_dict['budget__total_cost'] if\
        'budget__total_cost' in params_dict else None
        self.budget__tower_quotas = params_dict['budget__tower_quotas'] if\
        'budget__tower_quotas' in params_dict else None        
        self.budget__tower_unit_costs = params_dict['budget__tower_unit_costs'] if\
        'budget__tower_unit_costs' in params_dict else None         
        
        #Map parameters
        self.map_dimensions = params_dict['map_dimensions']#height(y),width(x)
        self.BORDER_MARGIN = params_dict['BORDER_MARGIN'] #For visualizing / debugging purposes make easier to see placements by not having on edges
        self.N_obstacles = params_dict['N_obstacles']
        
        #Target parameters
        self.N_targets = params_dict['N_targets']
        self.target_values = params_dict['target_values']
        
        #Tower parameters
        self.N_tower_sites = params_dict['N_tower_sites']
        self.N_towers = params_dict['N_towers']
        self.N_tower_kinds = len(self.N_towers)
        self.coverage_profile_types = params_dict['coverage_profile_types']
        
        #If manually specifying map:
        self.coordinates__obstacles = params_dict['coordinates__obstacles'] if\
        'coordinates__obstacles' in params_dict else None
        self.coordinates__targets = params_dict['coordinates__targets'] if\
        'coordinates__targets' in params_dict else None        
        self.coordinates__tower_sites = params_dict['coordinates__tower_sites'] if\
        'coordinates__tower_sites' in params_dict else None        

        #Variables created to make coverage / protection matrices:
        self.obstructed_masks = []
        self.distances_towers_targets = []
        self.unobstructed_coverages = []
        self.coordinates__solved_towers = []
        self.coverage_matrices = None
        

        self.figsize = (18,12)
        
        
        #Number iterations of Weighted L1. For this problem, seems to converge pretty fast (only a few iterations)
        self.N_reweighting_iterations_max = params_dict['N_reweighting_iterations_max']         
        #Number of random independent starts (since we are not gauranteed to 
        #reach a global optimum, try a few times, and take the best result)
#        self.N_random_starts_max = params_dict['N_random_starts_max']         
        #CVXPY Optimizer output
        self.VERBOSE = params_dict['VERBOSE']
        
        def summarize_environment():
            print self.random_seed
            print self.map_dimensions
            print self.BORDER_MARGIN
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

            
            coords = []
            while len(coords)<self.N_targets:
                x = np.random.randint(self.BORDER_MARGIN,self.map_dimensions[1]+1-self.BORDER_MARGIN,size=1)[0]
                y = np.random.randint(self.BORDER_MARGIN,self.map_dimensions[0]+1-self.BORDER_MARGIN,size=1)[0]
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
                    x = np.random.randint(self.BORDER_MARGIN,self.map_dimensions[1]+1-self.BORDER_MARGIN,size=1)[0]
                    y = np.random.randint(self.BORDER_MARGIN,self.map_dimensions[0]+1-self.BORDER_MARGIN,size=1)[0]
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
            K = 1.
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


     
    def get_target_value_matrix(self):
        """
        asdasd
        """
        pass


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
        plot_target_values = True
        if plot_target_values:
            for i ,t in enumerate(self.coordinates__targets):
                plt.text(t[0],t[1],self.target_values[i])
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
        
        #The first problem formulation
        #K kinds of towers
        #See more details about problem formulation in the writeup             
        
        #Get a full matrix of the concatenated coverage matrices for 
        #each tower type. THis new matrix has dimensions:
        #(Ntowers) x (sum(potential sites)), where the sum o=is over all tower types
        coverage = np.hstack(i for i in self.coverage_matrices)
        print coverage
        print coverage.shape 
        
        #Diagonal matrix of the values of each target
        #(for the scenarios where we don't care about maximizing covered value,
        #target_values is just all ones, so this is just the identity matrix)
        V = np.diag(self.target_values)
        
        #If doing scenario where we want to fortify weakest link, only makes
        #sense if all targets are equal value:
        if self.objective_type == 'min_entries':
            V = np.eye(len(self.target_values))

        #Get the matrix of coverage values / expected value saved:
        C = np.dot(V,coverage)
        print 'V', V
        print 'coverage', coverage
        print 'C', C
        
        
        #Since not gauranteed to reach global optimum on any particular initialization,
        #run a few times and take the best result.
        #Just define "best result" as the result which had the most overall 
        #"converged" x, combined over all tower kinds. 
#        for j in xrange(self.N_random_starts_max):
            
        
        a = 2. #1.
        tau = 1e-4
        N = sum(i for i in self.N_tower_sites)
        w = np.zeros(N)
        ones = np.ones(N)
        p = 1. #the exponents power when doing he exponent method:
        
        for i in xrange(self.N_reweighting_iterations_max):
            #The concatenated vector of occupancies: Concatenated over all
            #of the kinds of towers.
            x = cvx.Variable(N)
            
            #Different objective functions depending on which optimization problem.
            #These are defined in the scenarios in the main function.
            if self.objective_type == 'min_entries':
                operation = cvx.min_entries
            elif self.objective_type == 'sum_entries':
                operation = cvx.sum_entries
            else:
                raise Exception('must specify valid objective_type')
                
            #Objective function includes penalty term for non-binary x values
            if self.penalty_type == 'reweighted_L1':
                #objective = cvx.Maximize(t - x.T*w)
                objective = cvx.Maximize(operation(C*x - x.T*w))


            #Main constraints on 0<=x<=1
            constraints = [0<=x, x<=1]
            
            
            #And then for each kind of tower, append the constraint that there
            #be exactly N_i towers, or <= quota (depending on constraint type)
            if self.constraints__type == 'fixed_N_towers' or self.constraints__type == 'tower_quotas':
                for tk in xrange(self.N_tower_kinds):
                    before_sum = np.concatenate(([0],np.cumsum(self.N_tower_sites)))[tk]
                    print before_sum
                    print before_sum + self.N_tower_sites[tk]
                    if self.constraints__type == 'fixed_N_towers':
                        constraints.append(cvx.sum_entries(
                                        x[before_sum : before_sum + self.N_tower_sites[tk]]
                                        )==self.N_towers[tk])
                    elif self.constraints__type == 'tower_quotas':
                        constraints.append(cvx.sum_entries(
                                        x[before_sum : before_sum + self.N_tower_sites[tk]]
                                        )<=self.budget__tower_quotas[tk])
                    print x[before_sum : before_sum + self.N_tower_sites[tk]]
                    
            elif self.constraints__type == 'total_cost':
                costs = np.hstack([np.repeat(self.budget__tower_unit_costs[tk],self.N_tower_sites[tk]) for tk in xrange(self.N_tower_kinds)])
                constraints.append(cvx.sum_entries(costs * x) <= self.budget__total_cost)                    
                    
                    
                


                
                
            print 'penalty_type', self.penalty_type
            print 'objective_type', self.objective_type
            print 'constraints__type', self.constraints__type
            print 'budget__tower_quotas', self.budget__tower_quotas
            print 'operation', operation
            print 'objective', objective
            print 'constraints', constraints
            cvx.Problem(objective, constraints).solve(verbose=self.VERBOSE)
            x = np.array(x.value).flatten()
            print 'x', x
            w = a/(tau+np.abs(x))
            p += 1.
            plt.figure(figsize=(5,5))
            plt.plot(x,marker='o')
            plt.savefig('histrograms_{}.png'.format(i))
            print            
            
            
            
            
        #From the solution x, get the coordinates of those tower sites where we
        #really do want to place a tower
        #use = np.isclose(x,1.)
        for tk in xrange(self.N_tower_kinds):
            before_sum = np.concatenate(([0],np.cumsum(self.N_tower_sites)))[tk]
            y = x[before_sum : before_sum + self.N_tower_sites[tk]]
            inds = np.argsort(y)
            s = y[inds]
            use = np.where(s>.5)[0]
            print inds
            print s
            print use            
            if self.constraints__type == 'fixed_N_towers':
                if len(use) != self.N_towers[tk]:
                    print 'Solution did not converge properly. Choosing the K best towers.'
                    print self.N_towers[tk], len(use)
    #                use = use[-self.N_towers[tk]:]
                    use = inds[-self.N_towers[tk]:]
            elif self.constraints__type == 'tower_quotas':
                pass #Just use the towers thresholded at > .5
            print use
            
            
            self.coordinates__solved_towers.append([self.coordinates__tower_sites[tk][mm] for mm in inds[use]])
    #        print self.coordinates__solved_towers
    #        print len(self.coordinates__solved_towers)
        
    
    
    
    
    def run_scenario(self):
        """
        Run the whole scenario.
        Initialize map, solve placement, visualize everything.
        """
        self.initialize_random_map()
        self.visualize_environment('initial')
        self.get_tower_target_coverages()
        self.solve_environment()
        self.visualize_environment('solved') 
    
    
    
if __name__ == '__main__':
    
    # =============================================================================
    #     PARAMETERS
    # =============================================================================
    
    #Which scenario to run
    SCENARIO = 1 #1 #2 #3
    
    
    
    #'objective_type':  'sum_entries' vs. 'min_entries':
    #min corresponds to the problem of fortifying the weakest link
    #sum gives makes objective function maximize the total overall coverage.
    
    
    
    
    
    # =============================================================================
    #     MAIN
    # =============================================================================    
    #SCENARIO !:
    #Explanation:
    #asdasdasdad
    if SCENARIO == 1:
        NTARGETS = 33
        params_dict = {'random_seed':33120181,
                       'objective_type':'sum_entries',#'min_entries','sum_entries'
                       'penalty_type':'reweighted_L1',#'exponential','reweighted_L1','double_square'
                       'constraints__type': 'total_cost',#'fixed_N_towers', 'tower_quotas', 'total_cost'
                       #If want to include quotas / budgets on towers:
                       'budget__total_cost': 30.,
                       'budget__tower_quotas': [1,1,1],#[6,3,2],#If don't want to limit number of a given tower type, just put in large value 999999
                       'budget__tower_unit_costs': [50.,50.,20.],

                       'N_reweighting_iterations_max':10,
#                       'N_random_starts_max':5,
                       'map_dimensions':(300,400),
                       'BORDER_MARGIN':3,
                       'N_obstacles':10,
                       'N_targets':NTARGETS,#266,
#                       'target_values': [1.]*(NTARGETS-1) + [9999.],
                       'target_values': [9999.] + [1.]*(NTARGETS-1),
#                       'target_values': [1.]*NTARGETS,
                       'target_health': [10.]*NTARGETS,#!!!!!!!!!!!!!!!!not implemented yet to turn into probability protected
                       'N_tower_sites':[27,15,17],#[10,13],
                       'N_towers':[9,7,5],
                       'coverage_profile_types':['inverse_r','inverse_r','inverse_r'], #['fixed_radius','inverse_r','inverse_r2']
    #                   'coordinates__obstacles':[(0,0,44,44)],
    #                   'coordinates__targets':[(0,0),(88,66)],
    #                   'coordinates__tower_sites':[(99,107),(377,288)]
                       'VERBOSE':True
                       }
        
    
    
    if SCENARIO == 2:
        params_dict = {
                       }
    
    
    
    

    env = Environment(params_dict)
    env.run_scenario()
