import numpy as np
import matplotlib.pyplot as plt
import sympy


RL = 25     # 回波损耗
w0 = [-1.3174, 0.2673]  # 传输零点
Nfz = len(w0)           # 零点个数
N = 5       # 阶数
w0 += [np.inf] * (N - Nfz)  # 补齐非零点
_w0 = [1 / i for i in w0]  # 零点的倒数
Er = 1      # 全规范型
# 构造频率变量与初始递归状态
w = sympy.symbols('w')
wp = (w ** 2 - 1) ** 0.5
Ut = w - _w0[0]
Vt = wp * (1 - _w0[0] ** 2) ** 0.5
# 递归计算
for i in range(1, len(w0)):
    U = Ut * (w - _w0[i]) + Vt * wp * (1 - _w0[i] ** 2) ** 0.5
    V = Ut * wp * (1 - _w0[i] ** 2) ** 0.5 + Vt * (w - _w0[i])
    Ut = U
    Vt = V

Ut = sympy.expand(Ut)
Vt = sympy.expand(Vt)
# 求F(s)的根
Froots = np.real(np.fromiter(sympy.solve(Ut, w), dtype=complex))
Froots = [complex(0, i) for i in Froots]
Fs = np.poly(Froots)
# 求P(s)的根
Proots = [complex(0, i) for i in w0[:Nfz]]
Ps = np.poly(Proots)
if (N - Nfz) % 2 == 0:
    Ps *= 1j
# 将无限远零点加入P(s)的根  
Plist = np.zeros(N + 1, dtype=complex)
for i in range(Nfz + 1):
    Plist[N - Nfz + i] = Ps[i]
# 求ε
E = 1 / (10 ** (RL / 10) - 1) ** 0.5 * np.abs(np.polyval(Ps, 1j) / np.polyval(Fs, 1j) * Er)
# 求E(s)的根
Eroots = np.roots(Plist * Er + Fs * E)
for i in range(N):
    if np.real(Eroots[i]) >= 0:
        Eroots[i] = complex(-np.real(Eroots[i]), np.imag(Eroots[i]))
Es = np.poly(Eroots)
# 构建S11与S21
F = np.poly1d(Fs)
P = np.poly1d(Ps)
En = np.poly1d(Es)
wr = np.linspace(-4, 4, 1000)
wr *= 1j
S11 = 20 * np.log10(np.abs(F(wr) / En(wr)) / Er)
S21 = 20 * np.log10(np.abs(P(wr) / En(wr)) / E)
# 绘图
plt.plot(wr, S11, label='S11')
plt.plot(wr, S21, label='S21')
plt.legend()
plt.show()
