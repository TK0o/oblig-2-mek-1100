import scipy.io as sio
import numpy as np
from numba import vectorize
from numpy import nan
import matplotlib.pyplot as plt

data = sio.loadmat('data.mat')
x = data.get('x')
y = data.get('y')
u = data.get('u')
v = data.get('v')
xit = data.get('xit')
yit = data.get('yit')
Xit, Yit = xit[0], yit[0]

dudy = np.gradient(u, 0.5, axis=0)
dvdx = np.gradient(v, 0.5, axis=1)
curl = dvdx - dudy

@vectorize
def cmp (yt, ym, l, up = True):
    if up == True:
        return (nan if yt > ym else l)
    else:
        return (nan if yt < ym else l)

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

    plt.hlines (arr[0, 1], arr[0, 0], arr[1, 0], 'r', linewidth=1.5)
    plt.hlines (arr[2, 1], arr[0, 0], arr[1, 0], 'b', linewidth=1.5)

    plt.vlines (arr[1, 0], arr[1, 1], arr[2, 1], 'g', linewidth=1.5)
    plt.vlines (arr[0, 0], arr[1, 1], arr[2, 1], 'black', linewidth=1.5)

p1 = np.array ([35, 160, 70, 170])
p2 = np.array ([35, 85, 70, 100])
p3 = np.array ([35, 50, 70, 60])

P1 = p1.reshape((2, 2))
P2 = p2.reshape((2, 2))
P3 = p3.reshape((2, 2))
d_up = cmp (Yit, y, curl, True)
d_nd = cmp (Yit, y, curl, False)

U_up, V_up = cmp (Yit, y, u, True), cmp (Yit, y, v, True)
U_nd, V_nd = cmp (Yit, y, u, False), cmp (Yit, y, v, False)

plt.streamplot(x, y, U_up, V_up)
plt.streamplot(x, y, U_nd, V_nd)

plt.title('streamplot')
plt.xlabel('x')
plt.ylabel('y')
point (P1) ; point (P2) ; point (P3)

plt.plot(Xit, Yit, 'black', linewidth=2)
plt.show()

plt.title('kontur med fyll')
plt.xlabel('x')
plt.ylabel('y')
pcu = plt.contourf (x, y, d_up, cmap=plt.get_cmap('rainbow_r'))

plt.colorbar(pcu, label='over')
plt.contour(x, y, d_up, linewidths=0.1, colors='k')
plt.contour(x, y, d_nd, linewidths=0.1, colors='k')

pcd = plt.contourf (x, y, d_nd, cmap=plt.get_cmap('rainbow'))
plt.colorbar(pcd, label='under')
point (P1) ; point (P2) ; point (P3)

plt.plot(Xit, Yit, 'black', linewidth=2)
plt.show()

plt.title('kontur uten fyll')
plt.xlabel('x')
plt.ylabel('y')

plt.contour (x, y, d_up,cmap=plt.get_cmap('rainbow_r'))
plt.contour (x, y, d_nd,cmap=plt.get_cmap('rainbow'))
point (P1) ; point (P2) ; point (P3)

plt.plot(Xit, Yit, 'black', linewidth=2)

'''
run Oppg_e.py
'''
