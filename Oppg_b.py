import scipy.io as sio
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

@vectorize
def cmp (yt, ym, U, V, up = True):
    if up == True:
        return (nan if yt > ym else sqrt(U**2 + V**2))
    else:
        return (nan if yt < ym else sqrt(U**2 + V**2))

def plot2():
    C2 = cmp (Yit, y, u, v, True)
    D2 = cmp (Yit, y, u, v, False)
    plt.contourf (x, y, C2, cmap=plt.get_cmap('viridis'))
    plt.colorbar(label='over')
    plt.contourf (x, y, D2, cmap=plt.get_cmap('viridis'))
    plt.colorbar(label='under')
    plt.plot(Xit, Yit, 'black', linewidth=2)
    plt.title('oppg b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot2 ()

'''
run Oppg_b.py
'''
