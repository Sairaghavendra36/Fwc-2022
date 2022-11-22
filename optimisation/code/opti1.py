import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# function
def f(x,l):
    return ((np.sin(x)**3)+(l)*(np.sin(x)**2))
  
#defining derivative of f(x)
def df(x,l): 
    return (np.sin(x)*np.cos(x))*(3*np.sin(x)+2*(l))

#for maxima using gradient ascent
test_pts = [-np.pi/2 + 0.25, np.pi/2 - 0.25]
#test_pts = [0, np.pi]

#for proper value of lamdav (-3/2,3/2), one of the element in max_pts will lie in (-pi/2, pi/2) indicating one maxima in range

#for other values of lamdav (like 4), both elements in max_pts will not lie in (-pi/2, pi/2) indicating no maxima in range
lamdav = 100 
previous_step_size1=1
iters1=0
precision=0.000000001
alpha=0.0001
max_iters=100000000
lam_max1 = np.array(([-1.5,-1,-0.5,0,0.5,1,1.5]))
#Gradiant ascent method for Maxima Calculation
max_v = []
max_pts = []
for j in test_pts:
    cur_x1 = j
    previous_step_size = 1
    iters1 = 0
    while (previous_step_size > precision and iters1<max_iters):
        prev_x = cur_x1
        cur_x1 += alpha*df(prev_x, lamdav)
        previous_step_size = abs(cur_x1-prev_x)
        iters1+= 1
    max_val1=f(cur_x1, lamdav)
    max_v.append(max_val1)
    max_pts.append(cur_x1)
print(max_v)
print(max_pts)

#Plotting for f(x)
x=np.linspace(-np.pi/2+0.25,np.pi/2-0.25,100)
y2=f(x,100)
label_str = "$maxima$"
plt.plot(x,y2,label=label_str)
##Labelling points
#plt.plot(cur_x1,max_pts,'.',label='point of minima')
#plt.text(cur_x1,max_pts,f'P({cur_y1:.4f},{min1:.4f})')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid()
plt.legend()
plt.savefig('/home/user/fwc/opti/fig2.pdf')
plt.show()

