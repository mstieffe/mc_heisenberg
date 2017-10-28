import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import pickle
#temperature
t = 1.06

#system size
N_x = 10
N_y = 10
N_z = 1

#total number of spins
N_total = N_x *N_y *N_z

#coupling constants starting value
J = np.array([1.0, 1.0, 1.0])

#anisotropy constant
K_ani = 1.0

#anisotropy axis
ani_ax = np.array([0,0,1])

#external field
field = np.array([0,0,0])

#number of single flip steps with rotation factor = 1.0
steps_singleflip = 100000

#number of cluster steps
steps_cluster = 10000

#number of steps after equlibration for recording
steps_recording = 320000

#steps between the recording of data points after qeuibration
steps_betw_rec = 10000

#rotation-factor for single spin flips
rot_fac = 1.0

#number of steps bevor the rotation factor decreases
rot_steps = 20000

#number of equilibrated systems fo each temperature step
steps_data = 1

#energy contribution for a given spin
def energy(spin, nn_spins, h):
    sum_vec = np.array([0,0,0])
    for s in nn_spins:
        sum_vec = sum_vec + spin*s
    return -np.dot(sum_vec,J)-np.dot(spin,h) - K_ani* np.dot(spin, ani_ax)*np.dot(spin, ani_ax)

#exchange energy term per spin
def energy_heis(spin, nn_spins):
    sum_vec = np.array([0,0,0])
    for s in nn_spins:
        sum_vec = sum_vec + spin*s
    return -sum_vec[0]*J[0]-sum_vec[1]*J[1]-sum_vec[2]*J[2]
    
#total energy of the system
def tot_energy(spinlattice,h):
    tot_en = 0
    for x in range(0,N_x):
        for y in range(0,N_y):
            for z in range(0,N_z):
                tot_en = tot_en + energy_heis(spinlattice[x][y][z], nearest_neigh(spinlattice, x,y,z)) -np.dot(spinlattice[x][y][z],h)- K_ani* np.dot(spinlattice[x][y][z], ani_ax)*np.dot(spinlattice[x][y][z], ani_ax)
    return tot_en/N_total

#prob for not pinning the cluster 
def prop_ani(spin, flip_spin, t):
    e_new = -K_ani* np.dot(flip_spin, ani_ax)*np.dot(flip_spin, ani_ax)    
    e_old = -K_ani* np.dot(spin, ani_ax)*np.dot(spin, ani_ax)        
    d_e = e_new - e_old
    if d_e < 0:
        return 1.0
    else:
        return math.exp(-d_e/t)

#prop for a frozen bond between neighboring spins for the cluster algorithm       
def prop_exchange(spin, direction, neigh_spin, t):
    d_e = 2*np.dot(spin,J*direction)*np.dot(neigh_spin,J*direction)
    if d_e < 0:
        return 0.0
    else:
        return (1 - math.exp(-d_e/t))
    
#magnetisation of the system
def magnetisation(spinlattice):
    magneti = np.array([0,0,0])
    for x in range(0,N_x):
        for y in range(0,N_y):
            for z in range(0,N_z):    
                magneti = magneti + 1/float(N_total)*spinlattice[x][y][z]
    return magneti
   
 #rotation of the given spin  
def rot_spin(spin, factor):
    s = 1
    while(s >= 1):
        v1 = random.uniform(-1.0,1.0)
        v2 = random.uniform(-1.0,1.0)
        s = v1*v1 + v2*v2
    root = math.sqrt(1-s)
    rot_spin = np.array([2*v1*root, 2*v2*root, 1-2*s])   
    new_spin = (1-factor) * spin + factor * rot_spin
    norm = math.sqrt(np.dot(new_spin,new_spin))
    return new_spin/norm

#create n random vector with length = 1
def create_spin():
    s = 1
    while(s >= 1):
        v1 = random.uniform(-1.0,1.0)
        v2 = random.uniform(-1.0,1.0)
        s = v1*v1 + v2*v2
    root = math.sqrt(1-s)
    return np.array([2*v1*root, 2*v2*root, 1-2*s])        

#seperate the given spin in a parallel and perpendicular vector to a given direction
def parallel_perpendic(spin, direction):
    parallel = np.dot(spin, direction) * direction
    perpen = spin - parallel
    return parallel, perpen
    
#reflect the given spin at the given plane
def flip_spin(spin, direction):
    para, perpen = parallel_perpendic(spin, direction)
    return perpen-para
   
#nearest neighbour spins
def nearest_neigh(spinlattice, x, y, z):
    #create list of nearest neighbours of the chosen spin (check periodic boundary)
    nn_spins = []
    if x == 0:
        nn_spins.append(spinlattice[N_x-1][y][z])
        nn_spins.append(spinlattice[x +1][y][z])     
    elif x == N_x-1:
        nn_spins.append(spinlattice[0][y][z])
        nn_spins.append(spinlattice[x -1][y][z])
    else:
        nn_spins.append(spinlattice[x -1][y][z])
        nn_spins.append(spinlattice[x +1][y][z])      
    if y == 0:
        nn_spins.append(spinlattice[x][N_y-1][z])
        nn_spins.append(spinlattice[x][y+1][z])     
    elif y == N_y-1:
        nn_spins.append(spinlattice[x][0][z])
        nn_spins.append(spinlattice[x][y-1][z])
    else:
        nn_spins.append(spinlattice[x][y-1][z])
        nn_spins.append(spinlattice[x][y+1][z])     
    if N_z > 2:
        if z == 0:
            nn_spins.append(spinlattice[x][y][N_z-1])
            nn_spins.append(spinlattice[x][y][z+1])     
        elif z == N_z-1:
            nn_spins.append(spinlattice[x][y][0])
            nn_spins.append(spinlattice[x][y][z-1])
        else:
            nn_spins.append(spinlattice[x][y][z-1])
            nn_spins.append(spinlattice[x][y][z+1]) 
    return nn_spins
    
#nearest neighbors indizes    
def nn(x,y,z):
    nn_spins = []
    if x == 0:
        nn_spins.append([N_x-1,y,z])
        nn_spins.append([x +1,y,z])     
    elif x == N_x-1:
        nn_spins.append([0,y,z])
        nn_spins.append([x -1,y,z])
    else:
        nn_spins.append([x -1,y,z])
        nn_spins.append([x +1,y,z])      
    if y == 0:
        nn_spins.append([x,N_y-1,z])
        nn_spins.append([x,y+1,z])     
    elif y == N_y-1:
        nn_spins.append([x,0,z])
        nn_spins.append([x,y-1,z])
    else:
        nn_spins.append([x,y-1,z])
        nn_spins.append([x,y+1,z])     
    if N_z > 2:
        if z == 0:
            nn_spins.append([x,y,N_z-1])
            nn_spins.append([x,y,z+1])     
        elif z == N_z-1:
            nn_spins.append([x,y,0])
            nn_spins.append([x,y,z-1])
        else:
            nn_spins.append([x,y,z-1])
            nn_spins.append([x,y,z+1]) 
    return nn_spins
            
#range function for floats
def frange(x, y, jump):
  while x <= y:
    yield round(x,2)
    x += jump

#average of a given list
def av_vec(list):
    r = np.array([0,0,0])
    mag = 0
    length = len(list)
    for m in list:
        r = r + abs(m)/length
        mag = mag + math.sqrt(np.dot(m,m))/length
    return r, mag
    
def av(list):
    r = 0
    length = len(list)
    for m in list:
        r = r + m/length
    return r
    
def kumu(list):
    length= len(list)
    m_4, m_2 = 0,0
    for i in list:
        m_4 = m_4 + i*i*i*i/length
        m_2 = m_2 + i*i/length
    return m_4/(m_2*m_2)

def kumu_vec(list):
    length= len(list)
    m_4, m_2 = 0,0
    for i in list:
        m_4 = m_4 + np.dot(i,i)*np.dot(i,i)/length
        m_2 = m_2 + np.dot(i,i)/length
    return m_4/(m_2*m_2)

#set seed for random number generator
random.seed()


#list for recording the energy for a given temperature
list_e = []
#list for recording the energy averages
list_e_av = []
#list for recording the magnetisation for a given temperature
list_m = []
#list for recording the magnetisation average vector
list_m_av_vec = []
#list for recording the magnetisation average aboslut value
list_m_av_abs = []
#list for recording the temperature
list_t = []
#list for recording the binder cumulant
list_k = []

num_samples = 0

print("temperature: "+str(t))
for i in range(0,steps_data):
    if i%10==0:
        print("data points: "+str(i))
    #initialize 3d lattice for the spins
    spinlattice = [[[]]]
    #and for the external field
    fieldlattice = [[[]]]    
    #initialize random spins with length one and the external field
    for x in range(0,N_x):
        spinlattice.append([])
        fieldlattice.append([])
        for y in range(0,N_y):
            fieldlattice[x].append([])
            spinlattice[x].append([])
            for z in range(0,N_z):
                spinlattice[x][y].append([])
                fieldlattice[x][y].append(field)
                s = 1
                while(s >= 1):
                    v1 = random.uniform(-1.0,1.0)
                    v2 = random.uniform(-1.0,1.0)
                    s = v1*v1 + v2*v2
                root = math.sqrt(1-s)
                #spinlattice[x][y][z] = np.array([1,0,0])
                spinlattice[x][y][z] = np.array([2*v1*root,2*v2*root,1-2*s])


###################################### metropolis with rotation factor = 1.0  #########################################

    #acceptance_rate
    acceptance = 0
    #set rotation factor to the saved value
    rot_fac_var = rot_fac
        
    step = 1               
    while(step <= steps_singleflip):        
       
    
        #pick random spin
        rand_x = random.randint(0,N_x-1)
        rand_y = random.randint(0,N_y-1)
        rand_z = random.randint(0,N_z-1)

        #create list of nearest neighbours of the chosen spin (check periodic boundary)
        nn_spins = nearest_neigh(spinlattice, rand_x, rand_y, rand_z)
                
        #random new spin
        new_spin = rot_spin(spinlattice[rand_x][rand_y][rand_z], rot_fac_var)   
        
        #calculate differenz of the energy contributions of the old and the new spin
        delta_e = energy(new_spin, nn_spins, fieldlattice[rand_x][rand_y][rand_z])-energy(spinlattice[rand_x][rand_y][rand_z], nn_spins, fieldlattice[rand_x][rand_y][rand_z])
        
        #Monte-Carlo move
        if delta_e < 0:
            spinlattice[rand_x][rand_y][rand_z] = new_spin
            acceptance = acceptance + 1/float(steps_singleflip)
        else:
            if random.uniform(0.0,1.0) <= math.exp(-delta_e/t):
                spinlattice[rand_x][rand_y][rand_z] = new_spin   
                acceptance = acceptance + 1/float(steps_singleflip)
        
        step = step +1


    #print data
#        print('temperature: '+str(t))
#        print('magnetisation: '+str(magnetisation(spinlattice)))
#        print('total energy: '+str(tot_energy(spinlattice,fieldlattice[rand_x][rand_y][rand_z])))
#        print('acceptance: '+str(acceptance))

###################################### wolff cluster algorithm #########################################
    
    step = 1
    while(step <= steps_cluster):

        #pick random spin
        rand_x = random.randint(0,N_x-1)
        rand_y = random.randint(0,N_y-1)
        rand_z = random.randint(0,N_z-1)

        #chose a random direction
        direction = create_spin()

        #create a list for the spins in the cluster
        cluster = []
        cluster.append([rand_x, rand_y, rand_z])
        
        #create a list for the spins in the cluster that has to be visited
        visit = []
        visit.append([rand_x, rand_y, rand_z])
        
        #create cluster
        reject = 0
        while(reject != 1 and visit != []):
            #check anisotropy
            if random.uniform(0.0,1.0) <= prop_ani(spinlattice[visit[0][0]][visit[0][1]][visit[0][2]], flip_spin(spinlattice[visit[0][0]][visit[0][1]][visit[0][2]], direction), t):
                for n in nn(visit[0][0],visit[0][1],visit[0][2]):
                    #check if the neighbour spins are included in the cluster (decide if bond is broken or not)
                    if n not in cluster and random.uniform(0.0,1.0) <= prop_exchange(spinlattice[visit[0][0]][visit[0][1]][visit[0][2]], direction, spinlattice[n[0]][n[1]][n[2]], t):
                        cluster.append(n)
                        visit.append(n)
                visit.remove(visit[0])
            else:
                reject = 1
        
        #flip cluster if not rejected
        if reject == 0:
            for s in cluster:
                spinlattice[s[0]][s[1]][s[2]] = flip_spin(spinlattice[s[0]][s[1]][s[2]], direction)
        else:
            reject == 0
                
        step= step+1

    #print data
#        print('magnetisation: '+str(magnetisation(spinlattice)))
#        print('total energy: '+str(tot_energy(spinlattice,fieldlattice[rand_x][rand_y][rand_z])))
#        print('acceptance: '+str(acceptance))

        
###################################### recording data  #########################################
        
    step = 1
    while(step <= steps_recording):

        #pick random spin
        rand_x = random.randint(0,N_x-1)
        rand_y = random.randint(0,N_y-1)
        rand_z = random.randint(0,N_z-1)

        #chose a random direction
        direction = create_spin()

        #create a list for the spins in the cluster
        cluster = []
        cluster.append([rand_x, rand_y, rand_z])
        
        #create a list for the spins in the cluster that has to be visited
        visit = []
        visit.append([rand_x, rand_y, rand_z])
        
        #create cluster
        reject = 0
        while(reject != 1 and visit != []):
            #check anisotropy
            if random.uniform(0.0,1.0) <= prop_ani(spinlattice[visit[0][0]][visit[0][1]][visit[0][2]], flip_spin(spinlattice[visit[0][0]][visit[0][1]][visit[0][2]], direction), t):
                for n in nn(visit[0][0],visit[0][1],visit[0][2]):
                    #check if the neighbour spins are included in the cluster (decide if bond is broken or not)
                    if n not in cluster and random.uniform(0.0,1.0) <= prop_exchange(spinlattice[visit[0][0]][visit[0][1]][visit[0][2]], direction, spinlattice[n[0]][n[1]][n[2]], t):
                        cluster.append(n)
                        visit.append(n)
                visit.remove(visit[0])
            else:
                reject = 1
        
        #flip cluster if not rejected
        if reject == 0:
            for s in cluster:
                spinlattice[s[0]][s[1]][s[2]] = flip_spin(spinlattice[s[0]][s[1]][s[2]], direction)
        else:
            reject == 0
                
        step= step+1    
        
        
        #record produced sample
        if step % steps_betw_rec == 0:     
            num_samples = num_samples +1
            print("samples: "+str(num_samples))
            with open('N'+str(N_x)+'_T'+str(t)+''+'.pickle', 'a') as f:
                pickle.dump(spinlattice, f)
#            with open("samples_"+str(N_x)+"x"+str(N_y)+"x"+str(N_z)+"J"+str(J[0])+"A"+str(K_ani)+"eq"+str(steps_cluster)+"rec"+str(steps_betw_rec)+".txt", "a") as myfile:
#                    myfile.write(str(spinlattice)+"\n")
        
