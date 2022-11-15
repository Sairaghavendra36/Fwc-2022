import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
#sys.path.insert(0, 'sdcard/nikhil/matrix')  #(0, '/storage/emulated/0/github/school/ncert-vectors/defs/codes/CoordGeo')        #path to my scripts

#local imports
#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen
omat = np.array([[0,1],[-1,0]])
def line_gen(A,B):
    len =10
    dim = A.shape[0]
    x_AB = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
      temp1 = A + lam_1[i]*(B-A)
      x_AB[:,i]= temp1.T
    return x_AB
def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB

#if using termux
import subprocess
import shlex
#end if

#Input parameters
a=2
t1=2
t2=4
t3=1
C = np.array(([a*t1*t2,a*(t1+t2)]))
A = np.array(([a*t2*t3,a*(t2+t3)]))
B = np.array(([a*t3*t1,a*(t3+t1)]))
#Intersection of two lines
n1 = B-C
m1 = omat@n1
n2 = A-C
m2 = omat@n2
A1 = A
A2 = B
N=np.array(([n1.T,n2.T]))
print(type(N))
p = np.zeros(2)
p[0] = n1@A1
p[1] = n2@A2
  #Intersection
X=np.linalg.inv(N)@p
print(X)
k1=2
k2=0
x_AP=line_dir_pt(m1,A1,k1,k2)
k1=3
k2=0
x_AQ=line_dir_pt(m2,A2,k1,k2)
# To plot the figure

X = np.array(([-2,30]))
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_AC = line_gen(A,C)
 

 #Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],color='red')
plt.plot(x_BC[0,:],x_BC[1,:],color='blue')
plt.plot(x_AC[0,:],x_AC[1,:],color='black')
plt.plot(x_AP[0,:],x_AP[1,:],color='green')
plt.plot(x_AQ[0,:],x_AQ[1,:],color='orange')


 #Labeling the coordinates
tri_coords = np.vstack((A,B,C,X)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','X(-2,30)']
for i, txt in enumerate(vert_labels):
     plt.annotate(txt, # this is the text
                  (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(0,5), # distance from text to points (x,y)
                  ha='center') # horizontal alignment can be left, right or center
 

plt.xlabel('$ X $')
plt.ylabel('$ Y $')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('Orthocenter of Triangle')
 #if using termux
plt.savefig('/home/user/fwc/line/fig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/fwc/line/fig.pdf"))
#plt.show()
