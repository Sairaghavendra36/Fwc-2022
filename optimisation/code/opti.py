import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

# function
def f(x,l):
    return ((np.sin(x)**3)+(l)*(np.sin(x)**2))
  
#defining derivative of f(x)
def df(x,l): 
    return (np.sin(x)*np.cos(x))*(3*np.sin(x)+2*(l))

#for maxima using gradient ascent
cur_x1=np.pi/2
cur_y1=-np.pi/2
previous_step_size1=1
iters1=0
precision=0.000000001
alpha=0.0001
max_iters=100000000
lam_max1 = np.array(([-1.5,-1,-0.5,0,0.5,1,1.5]))
#Gradiant ascent method for Maxima Calculation
max_v = []
for i in np.arange(-1.5,2,0.5):
    cur_x1 = np.pi/2
    previous_step_size = 1
    iters1 = 0
    while (previous_step_size1>precision)&(iters1<max_iters):
        prev_x = cur_x1
        cur_x1+=alpha*df(prev_x,i)
        previous_step_size1 = abs(cur_x1-prev_x)
        iters1+= 1
    max_val1=f(cur_x1,i)
    max_v.append(max_val1)
print(cur_x1,i)
print(max_v)

max1 = max_v[0];
lam_max = i
for i in range(0, len(max_v)):        
   if(max_v[i] > max1):
       lam_max = i
       max1 = max_v[i];   
print(max1)
print(lam_max)

##Plotting f(x)
x=np.linspace(-np.pi/2,np.pi/2,100)
y1=f(x,lam_max1[lam_max])
label_str = "$maxima$"
plt.plot(x,y1,label=label_str)
#Labelling points
plt.plot(cur_x1,max1,'.',label='point of maxima')
plt.text(cur_x1,max1,f'P({cur_x1:.2f},{max1:.2f})',horizontalalignment='right',verticalalignment='center')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid()
plt.legend()
plt.savefig('/home/user/fwc/opti/fig1.pdf')
plt.show()


#Gradiant descent method for minima calculation
min_v = []
for i in np.arange(-1.5,2,0.5):
    while (previous_step_size1>precision)&(iters1<max_iters):
        prev_y = cur_y1
        cur_y1-=alpha*df(prev_y,i)
        previous_step_size1 = abs(cur_y1-prev_y)
        iters1+= 1
    min_val1=f(cur_y1,i)
    min_v.append(min_val1)
print(cur_y1,i)
print(min_v)

min1 = min_v[0]  
lam_min = 0
for j in range(0, len(min_v)):        
   if(min_v[j] < min1):
       lam_min = j
       min1 = min_v[j];   
print(min1)
print(lam_min)

##Plotting f(x)
x=np.linspace(-np.pi/2,np.pi/2,100)
y2=f(x,lam_max1[lam_min])
label_str = "$minima$"
plt.plot(x,y2,label=label_str)
#Labelling points
plt.plot(cur_y1,min1,'.',label='point of minima')
plt.text(cur_y1,min1,f'P({cur_y1:.4f},{min1:.4f})')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid()
plt.legend()
plt.savefig('/home/user/fwc/opti/fig.pdf')
plt.show()



#plt.subplot(211)
#plt.plot(x, y1, color='red', lw=5)
#plt.plot(x, y2, color='orange', lw=7)
#X, Y = [], []
#
#for lines in plt.gca().get_lines():
#   for x, y in lines.get_xydata():
#      X.append(x)
#      Y.append(y)
#
#idx = np.argsort(X)
#X = np.array(X)[idx]
#Y = np.array(Y)[idx]
#
#plt.subplot(212)
#
#plt.plot(X, Y, color='green', lw=0.75)
#
#plt.show()
