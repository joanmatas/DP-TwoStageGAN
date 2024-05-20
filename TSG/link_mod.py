import numpy as np
import torch
import time


class DFS():
    def __init__(self, mat: torch.Tensor) -> None:
        '''
        Class to perform Depth First Search on a NxN tensor with a trajectory drawn in it and returns a list of shape 1xKx3 with K being the length of the trajectory. 
        The format of the returned cordinates is as follows: [x,y,t] where x is the first coordinate of the trajectory point on the input matrix, y is the second
        coordinate of the trajectory point on the input matrix and t is the value of the input matrix in the position [x,y].

        In order to obtain the resulting trajectory in an array-like manner the DFS is performed once selecting a random point in the trajectory. Afterwards the first/last 
        point of the trajectory is obtained by calculating the centroid and obtaining the one furthest away. Once this is done, DFS is performed again without doing backtracking 
        to obtain the resulting trajectory from the selected point to the end of the trajectory.
        '''
        self.mat = mat
        self.max_len = len(self.mat.flatten().nonzero(as_tuple=True)[0])
        self.nodes_tup = self.mat.nonzero(as_tuple=True) # If we want to display the values of the nodes we can do self.mat[self.nodes.tup]
        self.nodes = self.mat.nonzero(as_tuple=False)
        self.current_node = self.nodes[np.random.randint(0,self.max_len)].tolist() #Randomly select the first node
        self.visited = [self.current_node] # Add the first node to the visited list
        self.stack = [self.current_node] # Add the first node to the stack list
        self.next_node = [] # We haven't selected a next node yet
        self.centroid = 0 # We dont know the centroid yet
    
    def step(self)-> None:
        '''
        A single step of the forward pass of DFS. The DFS is performed with priority on the central cross neighbours and afterwards choosing the neighbours with less neighbours.
        '''
        adj = self.get_adjacents(self.current_node[0], self.current_node[1])
        if len(adj)>0:
            priorities = self.priority_nodes(self.current_node, adj)
            num_adj = []
            if priorities == []:
                for a in adj:
                    num_adj.append([len(adj_n) for adj_n in self.get_adjacents(a[0], a[1]) if adj_n not in self.visited])
                self.next_node  = adj.pop(num_adj.index(min(num_adj))) 
            else:
                for p in priorities:
                    num_adj.append([len(adj_n) for adj_n in self.get_adjacents(p[0], p[1]) if adj_n not in self.visited])
                next_p = priorities.pop(num_adj.index(min(num_adj)))
                self.next_node = adj.pop(adj.index(next_p))

            while self.next_node in self.visited:
                try:
                    self.next_node = adj.pop(np.random.randint(0, len(adj)))
                except Exception as e:
                    self.next_node = []
                    return
    
            if self.next_node != []:
                self.visited.append(self.next_node)
                self.stack.append(self.next_node)
                self.current_node = self.next_node
        return

    def priority_nodes(self, current, adjacent):
        '''
        A function to select the central cross neighbours.
        '''
        priority = []
        for a in adjacent:
            if current[0] == a[0] or current[0] == a[1] or current[1] == a[0] or current[1] == a[1]:
                priority.append(a)
        return priority
        
    
    def forward(self) -> None:
        '''
        Doing steps of the DFS until we reach the end of the trajectory path.
        '''
        self.step()
        while self.next_node != []:
            self.step()
        return

    def unstep(self) -> None:
        '''
        A single backtracking step. Remove the last visited node from the stack and try to find new neigbours. If it is not possible start a backtracking step again.
        '''
        if len(self.stack)>0:
            self.current_node = self.stack.pop()
            self.step()
        else:
            self.current_node = []
        return
    
    def backward(self) -> None:
        '''
        Perform the backtracking steps until the stack is empty or we find a node that hasn't been visited.
        '''
        while self.current_node in self.visited and self.next_node == []:
            self.unstep()
        return      
    
    def get_adjacents(self, n: int, m: int) -> list:
        '''
        Function that returns the index and value of the adjaçent values to a specific index (n,m) in a matrix.
        Inputs:
        # n: first dimension of the index
        # m: second dimension of the index
        Outputs:
        adj: List of adjacent values to the spcific index in the shape of [(k,l), val], where k is the first dimension 
        of the position of the value on the matrix mat, l is the second dimension of the psoition of the value in the matrix mat
        and val is the value in said position.
        '''
        return [[k,l] for k in range(0,self.mat.shape[0]) if n-1 <= k <= n+1 for l in range(0,self.mat.shape[1]) if m-1 <= l <= m+1 and not (l == m and k == n) and not (l != m and k != n) and int(self.mat[k][l]) != 0]

    def calculate_centroid(self):
        '''
        Function to calculate the centroid of the trajectory.
        '''
        coords = np.array(self.visited)
        l = len(coords)
        sumx = np.sum(coords[:,0])
        sumy = np.sum(coords[:,1])
        self.centroid = [sumx/l, sumy/l]

    def dist(self, a:list , b:list):
        '''
        Function to calculate the euclidean distance between two points a and b.
        '''
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def get_distances_to_centroid(self):
        '''
        Function that obtains the distances to the centroid.
        '''
        self.calculate_centroid()
        return [self.dist(v,self.centroid) for v in self.visited]

    def get_furthest(self):
        '''
        Function that returns the furthest point from the current centroid.
        '''
        distances = self.get_distances_to_centroid()
        furthest = max(distances)
        first = self.visited[distances.index(furthest)]
        return first

    def fit(self)->list:
        '''
        Main code of the class. All the ordered steps to obtain a single trajectory from an NxN matrix are performed in this function. Returns a list of [x,y,t] points.
        '''
        while len(self.stack) > 0:
            self.forward()
            self.backward()
        self.current_node = self.get_furthest()
        self.visited = [self.current_node]
        self.stack = [self.current_node]
        self.next_node = []
        self.forward()
        ordered = torch.tensor(self.visited).T
        ordered = (ordered[0], ordered[1])
        return [[d[0], d[1],int(t)] for d,t in zip(self.visited, self.mat[ordered])]
        
        
if __name__=='__main__':
    '''
    To run a little test
    '''
    start = time.thread_time()
    not_found = 0
    short_traj = []
    for i in range(0,500):
        try:
            print(f'Trajectory number: {i}')
            real_traj2sgan = np.load(f'../data/grid32_train_filtered/{i}.npy') # Load the trajectories from your path
            
            # Reshape and create the tensor
            shape = [len(real_traj2sgan), len(real_traj2sgan[0]), len(real_traj2sgan[0][0])]
            flat_traj2sgan = real_traj2sgan.flatten()
            tensor_traj2sgan = torch.Tensor(flat_traj2sgan).reshape(shape)[0]

            # Call the class DFS
            dfs = DFS(tensor_traj2sgan)
            # Run the fit function to obtain the trajectory vector g
            g = dfs.fit()
            print(f'\t{g}')
            if len(g) <= 1: #Check how many trajectories are just one point (or 0)
                short_traj.append(i)
        except FileNotFoundError:
            not_found +=1
            print(f'\t[Err] File {i} does not exist') # In case the file number i does not exist
    finish = time.thread_time()

    # Check the time elapsed and the errors that may happen
    print(f'Time elapsed: {finish - start:.4}')
    print(f'Number of files not found: {not_found}')
    print(f'Trajectories that are too short {short_traj}')