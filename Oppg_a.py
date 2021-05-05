import scipy.io as sio
import numpy as np
from numba import prange, njit

data = sio.loadmat('data.mat')
x = data.get('x')
y = data.get('y')
u = data.get('u')
v = data.get('v')
xit = data.get('xit')
yit = data.get('yit')

def oppg_A():
    matrise = [x, y, u, v]
    navn = ['x', 'y', 'u', 'v', 'xit', 'yit']
    index = 0
    a, b = np.shape(xit), np.shape(yit)
    py, pn = y[:, 0][0] / 10, y[:, 0][-1] / 10
    @njit(parallel=True)
    def test_con(l):
        n, N, s, S = len (l), len (l[0]), 0, 0
        si, Sj = n * ((N - 1) * 0.5), N * ((n - 1) * 0.5)

        for i in prange(n):
            for I in prange(N - 1):
                if l[i, I + 1] - l[i, I] == 0.5: s += 0.5
                else: s += 0

        for j in prange (n - 1):
            for J in prange (N):
                if l[j + 1, J] - l[j, J] == 0.5: S += 0.5
                else: S += 0

        if s == si or S == Sj:
            return True

    def statement(l):
        if test_con(l) == True:
            return 'har en avstand på 0.05 mm mellom hvert punkt'
        else:
            return 'har ikke en avstand på 0.05 mm mellom hvert punkt'

    print ('I)')
    for i in matrise:
        xret, yret = len (i[0]), len (i)
        t_a = f'{navn[index]} har {xret} punkter i x-retning'
        t_b = f'og {yret} punkter i y-retning'

        print (f'{t_a} {t_b}')
        index += 1

    print ('II)' + '\n' + f'xit har form {a}, og yit har form {b}')

    print ('III)' + '\n' + f'x {statement (x)} og' +
           '\n' + f'y {statement(y)}')

    print ('IV)' + '\n' +
    f'det ytterste punktet er {py} cm og {pn} cm fra midten av røret')
    return
oppg_A()

'''
run Oppg_a.py

I)
x har 194 punkter i x-retning og 201 punkter i y-retning
y har 194 punkter i x-retning og 201 punkter i y-retning
u har 194 punkter i x-retning og 201 punkter i y-retning
v har 194 punkter i x-retning og 201 punkter i y-retning
II)
xit har form (1, 194), og yit har form (1, 194)
III)
x har en avstand på 0.05 mm mellom hvert punkt og
y har en avstand på 0.05 mm mellom hvert punkt
IV)
det ytterste punktet er -5.0 cm og 5.0 cm fra midten av røret
'''
