#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
#sys.path.insert(0,'/storage/emulated/0/github/cbse-papers/CoordGeo')         #path to my scripts
sys.path.insert(0,'/sdcard/fwc/circle/CoordGeo')


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if
def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def line_dir_pt(m,G,k1,k2):
   len = 10
   dim = G.shape[0]
   x_LC = np.zeros((dim,len))
   lam_1 = np.linspace(k1,k2,len)
   for i in range(len):
     temp1 = G + lam_1[i]*m
     x_LC[:,i]= temp1.T
   return x_LC

#for intersection tangent
x1 = np.array(([5,0],[0,5]))
e,p = np.linalg.eig(x1)
w = np.array(([np.sqrt(e[0]),np.sqrt(e[1])]))
q = np.array(([np.sqrt(e[0]),-np.sqrt(e[1])]))
K = np.array(([w,q]))
n3 = p@w
n4 = p@q
C = np.array(([9.8,2.8]))
N = np.linalg.inv(K)@C
print(N)
print("norm :",n3)
print("norm :",n4)
b = np.array(([2.23,-2.23]))
c = np.array(([2.23,2.23]))

#Standard basis vectors
e1 = np.array((1,0)).reshape(2,1)
e2 = np.array((0,1)).reshape(2,1)

#Input parameters
r1  = 1
r2 = 4
theta=np.pi/3
h=np.array((2/3,0)).reshape(2,1)
V1 = np.eye(2)
u1 = np.array((-2,-1)).reshape(2,1)
f1 =4
V2=np.eye(2)
u2=np.array((-6,-4)).reshape(2,1)
f2 = 36
S1 = (V1@h+u1)@(V1@h+u1).T-(h.T@V1@h+2*u1.T@h+f1)*V1
S2 = (V2@h+u2)@(V2@h+u2).T-(h.T@V2@h+2*u2.T@h+f2)*V2
O1 = -u1.T
O2 = -u2.T
print("S matrix :",S1,S2)
m = np.array(([6,-8]))
n = np.array(([8,6]))
p = np.array(([2,1]))
C = 32
f0 = u1.T@V1@u1-f1
i = (f0)/(n.T@V1@n)
ki = np.sqrt(i)
X = V1*ki*n-u1
P = np.array(([m@p,C]))
M = np.vstack(([m,n]))
X = np.linalg.inv(M)@P
A = np.array(([2/3,0]))
print("intersection :",X)

k1 = 0.5
k2 = -0.5
x_X = line_dir_pt(m,X,k1,k2)

k1 = 1
k2 = -1
x_A = line_dir_pt(b,N,k1,k2)

k1 = 3
k2 = -1.5
x_c = line_dir_pt(c,N,k1,k2)

#Intermediate parameters
f01 = np.abs(-f1+u1.T@LA.inv(V1)@u1)
f02 = np.abs(-f2+u2.T@LA.inv(V2)@u2)

#Eigenvalues and eigenvectors
D_vec1,P1 = LA.eig(S1)
lam1 = D_vec1[0]
lam2 = D_vec1[1]
p1 = P1[:,1].reshape(2,1)
p2 = P1[:,0].reshape(2,1)
D = np.diag(D_vec1)
t1= np.sqrt(np.abs(D_vec1))
negmat = np.block([e1,-e2])
t2 = negmat@t1


#Normal vectors to the conic
n1 = P1@t1
n2 = P1@t2
print(":",n1,n2)
x=n1@n2
y = np.linalg.norm(n1)*np.linalg.norm(n2)
theta = np.arccos(x/y)
theta1 = theta*180/np.pi
print("theta: ",theta1)

#kappa
den1 = n1.T@LA.inv(V1)@n1
den2 = n2.T@LA.inv(V1)@n2
k1 = np.sqrt(f01/(den1))
k2 = np.sqrt(f01/(den2))

#q11 = LA.inv(V1)@((k1*n1-u1.T).T)
q12 = LA.inv(V1)@((-k1*n1-u1.T).T)
#q21 = LA.inv(V1)@((k2*n2-u1.T).T)
q22 = LA.inv(V1)@((-k2*n2-u1.T).T)
print("point of contact :",q12,q22)


#Eigenvalues and eigenvectors
D_vec2,P2 = LA.eig(S2)
lam11 = D_vec2[0]
lam21 = D_vec2[1]
p11 = P2[:,1].reshape(2,1)
p21 = P2[:,0].reshape(2,1)
D1 = np.diag(D_vec2)
t11= np.sqrt(np.abs(D_vec2))
negmat = np.block([e1,-e2])
t21 = negmat@t11

#Normal vectors to the conic
n11 = P2@t11
n21 = P2@t21
print("normal :",n1,n11,n2,n21)
#kappa
den11 = n11.T@LA.inv(V2)@n11
den21 = n21.T@LA.inv(V2)@n21

k11 = np.sqrt(f02/(den11))
k21 = np.sqrt(f02/(den21))

#q11_1 = LA.inv(V2)@((k11*n11-u2.T).T)
q12_1 = LA.inv(V2)@((-k11*n11-u2.T).T)
#q21_1 = LA.inv(V2)@((k21*n21-u2.T).T)
q22_1 = LA.inv(V2)@((-k21*n21-u2.T).T)
print("point of contact :",q12_1,q22_1)

#Generating the lines
x_hq22 = line_gen(h,q22)
x_hq12 = line_gen(h,q12)
x_q22q22_1 = line_gen(q22,q22_1)
x_q12q12_1 = line_gen(q12,q12_1)


##Generating the circle
x_circ= circ_gen(O1,r1)
x_circ1=circ_gen(O2,r2)

##Plotting all lines
plt.plot(x_hq22[0,:],x_hq22[1,:],color='green')
plt.plot(x_hq12[0,:],x_hq12[1,:],color='red')
plt.plot(x_q22q22_1[0,:],x_q22q22_1[1,:],label='$tangent1$',color='green')
plt.plot(x_q12q12_1[0,:],x_q12q12_1[1,:],label='$tangent2$',color='red')
plt.plot(x_X[0,:],x_X[1,:],color='black',label='$tangent3$')
plt.plot(x_A[0,:],x_A[1,:],color='blue')
plt.plot(x_c[0,:],x_c[1,:],color='violet')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(x_circ1[0,:],x_circ1[1,:],label='$circle1$')

#Labeling the coordinates
tri_coords = np.vstack((h.T,q12.T,q22.T,O1,O2,q12_1.T,q22_1.T,X.T)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['h','q12','q22','O1','O2','q12_1','q22_1','X']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#
#if using termux
plt.savefig('/sdcard/fwc/circle/fig.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/fwc/circle/circle.pdf"))
plt.show()
