import scipy.io as sio
import numpy as np
from numba import vectorize
from numpy import nan, sqrt
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

def func (l1, l2, l3, p):
    arr = np.zeros((4, 2))
    arrh = np.zeros(2)

    p1 = arrh + [p[0, 0], p[0, 1]]
    p2 = arrh + [p[1, 0], p[0, 1]]
    p3 = arrh + [p[1, 0], p[1, 1]]
    p4 = arrh + [p[0, 0], p[1, 1]]

    L = [p1, p2, p3, p4]
    for i in range(len (L)):
        m1, m2 = int (L[i][0]), int (L[i][1])
        arr[i] = np.array([x[0, m1], y[m2, 0]])

    x0, x1 = p[0, 0], p[1, 0]
    y0, y1 = p[0, 1], p[1, 1]

    s1 = np.sum (l1[y0 - 1, x0 - 1:x1] * 0.5)
    s2 = np.sum (l2[y0 -1:y1, x1 - 1] * 0.5)
    s3 = - np.sum (l1[y1 - 1, x0 - 1:x1] * 0.5)
    s4 = - np.sum (l2[y0 - 1:y1, x0 - 1] * 0.5)

    S = [s1, s2, s3, s4]

    c11, c12 = x[0,int(p1[0]) + 5], y[int(p1[1]) - 9, 0]
    c21, c22 = x[0,int(p2[0]) + 1], y[int(p2[1]) + 3,0]
    c31, c32 = x[0,int(p1[0]) + 5], y[int(p3[1]) + 2, 0]
    c41, c42 = x[0,int(p1[0]) - 30], y[int(p2[1]) + 4, 0]

    plt.hlines (arr[0, 1], arr[0, 0], arr[1, 0], 'r', linewidth=1.5)
    plt.annotate (f'{S[0]:.2g}', xy=(c11, c12))

    plt.hlines (arr[2, 1], arr[0, 0], arr[1, 0], 'b', linewidth=1.5)
    plt.annotate (f'{S[1]:.2g}', xy=(c21, c22))

    plt.vlines (arr[1, 0], arr[1, 1], arr[2, 1], 'g', linewidth=1.5)
    plt.annotate (f'{S[2]:.2g}', xy=(c31, c32))

    plt.vlines (arr[0, 0], arr[1, 1], arr[2, 1], 'black', linewidth=1.5)
    q = f'{S[3]:.4g}'
    plt.annotate (f'{q:>11}', xy=(c41, c42))

    crl = np.sum(l3[y0 - 1:y1, x0 - 1:x1])*0.5**2
    return sum (S), crl

p1 = np.array ([35, 160, 70, 170])
p2 = np.array ([35, 85, 70, 100])
p3 = np.array ([35, 50, 70, 60])

P1 = p1.reshape((2, 2))
P2 = p2.reshape((2, 2))
P3 = p3.reshape((2, 2))

r1, f1 = func (u, v, curl, P1)
r2, f2 = func (u, v, curl, P2)
r3, f3 = func (u, v, curl, P3)

Xit, Yit = xit[0], yit[0]

@vectorize
def replace (arr): return 0

plt.title('verdier for hver side')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(Xit, Yit, 'black', linewidth=0.5)
curln = replace (curl)
plt.contour (x, y, curln)

print ('\n' + 'kurveintegralet for rektanglene r1, r2 og r3')
print (f'r1 = {r1:.4f},', f'r2 = {r2:.4f},', f'r3 = {r3:.4f}', '\n')
print ('og flateintegralet for rektanglene r1, r2 og r3')
print (f'f1 = {f1:.4f},', f'f2 = {f2:.4f},', f'f3 = {f3:.4f}')

'''
run Oppg_f.py

kurveintegralet for rektanglene r1, r2 og r3
r1 = 2695.5141, r2 = -60976.6002, r3 = 9.5210

og flateintegralet for rektanglene r1, r2 og r3
f1 = 2621.5587, f2 = -61482.5410, f3 = -12.2143
'''
