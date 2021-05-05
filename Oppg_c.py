import scipy.io as sio
import numpy as np
from numba import vectorize
from numpy import nan, sqrt
import matplotlib.pyplot as plt
import itertools as itr
from math import floor

data = sio.loadmat('data.mat')
x = data.get('x')
y = data.get('y')
u = data.get('u')
v = data.get('v')
xit = data.get('xit')
yit = data.get('yit')
Xit, Yit = xit[0], yit[0]

@vectorize
def cmp (yt, ym, U, V, up = True):
    if up == True:
        return (nan if yt > ym else sqrt(U**2 + V**2))
    else:
        return (nan if yt < ym else sqrt(U**2 + V**2))

def plot2():
    C2 = cmp (Yit, y, u, v, True)
    D2 = cmp (Yit, y, u, v, False)
    plt.contour (x, y, C2, cmap=plt.get_cmap())
    plt.contour (x, y, D2, cmap=plt.get_cmap())

def Oppg_c_I(k):
    @vectorize
    def cmp1(yt, y, l, up = True):
        if up == True:
            return (nan if yt > y else l)
        else:
            return (nan if yt < y else l)

    U_up, V_up = cmp1 (Yit, y, u, True), cmp1 (Yit, y, v, True)
    U_nd, V_nd = cmp1 (Yit, y, u, False), cmp1 (Yit, y, v, False)

    plt.quiver (x[::k, ::k], y[::k, ::k], U_up[::k, ::k], V_up[::k, ::k])
    plt.quiver (x[::k, ::k], y[::k, ::k], U_nd[::k, ::k], V_nd[::k, ::k])

def point(l):
    arr = np.zeros((4, 2))
    arrh = np.zeros(2)
    p1 = arrh + [l[0, 0], l[0, 1]]
    p2 = arrh + [l[1, 0], l[0, 1]]
    p3 = arrh + [l[1, 0], l[1, 1]]
    p4 = arrh + [l[0, 0], l[1, 1]]
    L = [p1, p2, p3, p4]
    for i in range(len (L)):
        m1, m2 = int (L[i][0]), int (L[i][1])
        arr[i] = np.array([x[0, m1], y[m2, 0]])

    plt.hlines (arr[0, 1], arr[0, 0], arr[1, 0], 'r')
    plt.hlines (arr[2, 1], arr[0, 0], arr[1, 0], 'b')

    plt.vlines (arr[1, 0], arr[1, 1], arr[2, 1], 'g')
    plt.vlines (arr[0, 0], arr[1, 1], arr[2, 1], 'black')

p1 = np.array ([35, 160, 70, 170])
p2 = np.array ([35, 85, 70, 100])
p3 = np.array ([35, 50, 70, 60])

P1 = p1.reshape((2, 2))
P2 = p2.reshape((2, 2))
P3 = p3.reshape((2, 2))

#plot2 ()
point (P1) ; point (P2) ; point (P3)
Oppg_c_I(11)
plt.xlabel('x')
plt.ylabel('y')
plt.title('oppg c')

plt.plot(Xit, Yit, 'r', linewidth=2)

'''
run Oppg_c.py
'''
