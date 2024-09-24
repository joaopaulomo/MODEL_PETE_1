import numpy as np
import struct
import datetime

# Título : Modelagem sísmica com equação acústica para densidade constante
# Autor : Peterson Nogueira Santos
# Data : 08/11/2022 (ultima atualizacao)

# Função principal
def main():
    # Parâmetros iniciais
    nz = 300
    nx = 300
    dz = 10.0
    dx = 10.0
    nt = 2500
    dt = 0.001
    sz = 1
    ns = 1
    s0x = 150
    ds = 2
    gz = 1
    fpeak = 15.0
    nb = 100
    order = 8

    nzb = nz + 2 * nb
    nxb = nx + 2 * nb

    vel = np.zeros((nz, nx))
    fonte = np.zeros(nt)
    D = np.zeros((nt, nx))

    # Modelo de velocidades (m/s) com 3 camadas
    vel[0:10, :] = 1500
    vel[10:100, :] = 2000
    vel[100:200, :] = 3000

    # Fontes e receptores
    sz += nb
    gz += nb
    sx = np.zeros(ns, dtype=int)
    for i in range(ns):
        sx[i] = nb + s0x + ds * i

    for i in range(ns):
        print(f'Posicao fonte {i+1}: {sx[i]}')

    # Geração da fonte Ricker
    fonte = source(nt, dt, fpeak)

    D = np.zeros((nt, nx))
    with open('dado.bin', 'wb') as f:
        for is_ in range(ns):
            print(f'Tiro: {is_+1}')
            print(f'Fonte: {sx[is_]-nb}, {sz-nb}')
            # Modelagem sísmica
            modelagem(is_, nx, nz, nt, dx, dz, dt, fonte, nb, nxb, nzb, sx[is_], sz, gz, vel, D)
            for it in range(nt):
                f.write(struct.pack(f'{nx}f', *D[it, :]))

    print("\nProcesso concluído")
    timestamp()

# fonte Ricker
def source(nt, dt, fpeak):
    tdelay = 6.0 / (5.0 * fpeak)
    wpeak = 2.0 * np.pi * fpeak
    waux = 0.5 * wpeak
    fonte = np.zeros(nt)
    for i in range(nt):
        t = i * dt
        tt = t - tdelay
        fonte[i] = np.exp(-waux * waux * tt * tt / 4.0) * np.cos(wpeak * tt)
    return fonte

def calc_coef(a):
    coef = np.zeros(a + 1)
    if a == 2:
        coef = [1.0, -2.0, 1.0]
    elif a == 4:
        coef = [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
    elif a == 6:
        coef = [1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0, -49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0]
    elif a == 8:
        coef = [-1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0, -205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0]
    return coef

def op_l(a, coef, nz, nx, dz, dx, P, lap_P):
    in_n = (a // 2) + 1
    lim_nx = nx - (a // 2)
    lim_nz = nz - (a // 2)
    for i in range(in_n, lim_nx):
        for j in range(in_n, lim_nz):
            Pxx = 0.0
            Pzz = 0.0
            for k in range(a + 1):
                Pxx += coef[k] * P[j, i + k - in_n]
                Pzz += coef[k] * P[j + k - in_n, i]
            lap_P[j, i] = Pxx / (dx ** 2) + Pzz / (dz ** 2)
    return lap_P

# Função de modelagem sísmica
def modelagem(is_, nx, nz, nt, dx, dz, dt, fonte, nb, nxb, nzb, sx, sz, rsz, vel, scg):
    order = 8
    vel_ext = np.zeros((nzb, nxb))
    p0 = np.zeros((nzb, nxb))
    p1 = np.zeros((nzb, nxb))
    p2 = np.zeros((nzb, nxb))
    L = np.zeros((nzb, nxb))
    coef = calc_coef(order)

    makevelextend(nz, nx, nzb, nxb, nb, vel, vel_ext)

    for it in range(nt):
        p0 *= 0.0
        p1 *= 0.0
        p2 *= 0.0
        L *= 0.0

        p2[sz, sx] += fonte[it]
        scg[it, :] = p2[rsz, nb:nx+nb]

        p0, p1 = p1.copy(), p2.copy()
        p2 = 2 * p1 - p0 + (dt ** 2) * (vel_ext ** 2) * L
        atenuacao(nxb, nzb, nb, p2)

    return scg

# Função de atenuação
def atenuacao(nxb, nzb, nb, p2):
    lz = nzb
    for i in range(nb):
        for j in range(nxb):
            p2[i, j] *= np.exp(-0.0005 * (nb - i)) ** 2
            p2[lz - 1, j] *= np.exp(-0.0005 * (nb - i)) ** 2
        lz -= 1

    lx = nxb
    for i in range(nb):
        for j in range(nzb):
            p2[j, i] *= np.exp(-0.0005 * (nb - i)) ** 2
            p2[j, lx - 1] *= np.exp(-0.0005 * (nb - i)) ** 2
        lx -= 1

def makevelextend(nz, nx, nzb, nxb, nb, c, cext):
    cext[nb:nz+nb, nb:nx+nb] = c

    # Lado Superior e Inferior
    for ix in range(nx):
        for iz in range(nb):
            cext[iz, nb + ix] = c[0, ix]        # Lado Superior
            cext[nz + iz + nb, nb + ix] = c[nz - 1, ix]    # Lado Inferior

    # Lado Esquerdo e Direito
    for iz in range(nzb):
        for ix in range(nb):
            cext[iz, ix] = cext[iz, nb]        # Lado Esquerdo
            cext[iz, ix + nx + nb] = cext[iz, nb + nx]    # Lado Direito

# Função que imprime a data e hora
def timestamp():
    now = datetime.datetime.now()
    print(f'Data: {now.day}/{now.month}/{now.year} Hora: {now.hour}:{now.minute}:{now.second}')

if __name__ == "__main__":
    main()
