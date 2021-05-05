import scipy.io as sio
import numpy as np

data = sio.loadmat('data.mat')
x = data.get('x')
y = data.get('y')
u = data.get('u')
v = data.get('v')
xit = data.get('xit')
yit = data.get('yit')
Xit, Yit = xit[0], yit[0]

def func (l1, l2, p):
    x0, x1 = p[0, 0], p[1, 0]
    y0, y1 = p[0, 1], p[1, 1]

    s1 = -np.sum (l2[y0 - 1, x0 - 1:x1]* 0.5)
    s2 = np.sum (l1[y0 - 1:y1, x1 - 1]* 0.5)
    s3 = np.sum (l2[y1 - 1, x0 - 1:x1]* 0.5)
    s4 = -np.sum (l1[y0 - 1:y1, x0 - 1]* 0.5)

    l = [s1, s2, s3, s4]
    return l


p1 = np.array ([35, 160, 70, 170])
p2 = np.array ([35, 85, 70, 100])
p3 = np.array ([35, 50, 70, 60])

P1 = p1.reshape((2, 2))
P2 = p2.reshape((2, 2))
P3 = p3.reshape((2, 2))

v1 = func(u, v, P1)
v2 = func(u, v, P2)
v3 = func(u, v, P3)
V = [v1, v2, v3]

for i in range (int(len(V))):
    print ('fluks av rektangel ' + f'{i + 1} ' + 'er '  +
           f'{np.sum(V[i]):.4f} ' +
           'med verdiene' + '\n'
           's1 = ' + f'{V[i][0]}' + '\n'
           's2 = ' + f'{V[i][1]}' + '\n'
           's3 = ' + f'{V[i][2]}' + '\n'
           's4 = ' + f'{V[i][3]}' + '\n')

'''
run Oppg_g.py

fluks av rektangel 1 er 104.8526 med verdiene
s1 = 1556.867943941396
s2 = 21664.567474322168
s3 = -2059.677184793871
s4 = -21056.905628561482

fluks av rektangel 2 er -6476.9392 med verdiene
s1 = -5187.564033067892
s2 = 14782.532896182347
s3 = -4074.0522144394345
s4 = -11997.85583077298

fluks av rektangel 3 er -124.5687 med verdiene
s1 = -195.57014792583357
s2 = 1536.8217966413547
s3 = 284.9436464350764
s4 = -1750.7639611955597
'''
