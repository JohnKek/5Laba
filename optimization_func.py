import numpy as np
import matplotlib.pyplot as plt
import pylab
import control
import scipy
from scipy.optimize import minimize
import numpy.random


def build_real_and_get(X,y,h_t):
    plt.figure(figsize=(10, 5))
    plt.plot(X,y,label='Полученная характеристика')
    plt.plot(h_t[:, 0], h_t[:, 1],label='Исходная характеристика',linestyle = '-')
    plt.legend()
    plt.title('Step Responsse ')
    plt.ylabel('Amplitude')
    plt.xlabel('Time(sec)')
    plt.grid(True)
    plt.show()

## Переходные характеристики 2-3 порядка  при заданом векторе параметров
def transfer_2_power(x,T,power):
    syst=None
    s = control.TransferFunction.s
    if power == 2:
        syst =  (x[3]*(s**2) + x[4]*s + x[5])/(x[0]*(s**2) + x[1]*s + x[2])
    elif power == 3:
        syst = ((s**3)*x[4] + (s**2)*x[5] + s*x[6] + x[7])/((s**3)*x[0] + (s**2)*x[1] + s*x[2] + x[3])

    return control.step_response(syst,T)



def optimize_target(x,y_target,T,power):
    _,y_real = transfer_2_power(x,T,power)
    return np.linalg.norm(y_target - y_real) ** 2


def adam(x,alpha,gamma_1,gamma_2,y_target,T,power,eps,maxiter=1000):
    ## Вычисление градиента
    u = np.zeros(len(x))
    m = np.zeros(len(x))
    n_iter = 0
    while optimize_target(x,y_target,T,power) > eps and maxiter > n_iter: # for j in range(iterations):
        ## Градиенты
        x_grad = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] == 0:
                dx = 0.0001
            else:
                dx = 0.0001*x[i]
            d2x = dx*2
            x[i] += dx
            f1 = optimize_target(x,y_target,T,power)
            x[i] -= d2x
            f2 = optimize_target(x,y_target,T,power)
            x[i] += dx
            x_grad[i] = (f1 - f2)/d2x
        #print(m_k.shape,m.shape,x_grad.shape)
        m_k = m*gamma_1 + (1-gamma_1)*(x_grad)
        u_k = u*gamma_2 + (1-gamma_2)*(x_grad**2)
        x -= alpha*m_k/(np.sqrt(u_k + 10**(-8)))
        u = u_k
        m = m_k
        curr_res = optimize_target(x,y_target,T,power)
        n_iter = n_iter + 1
        #print('Целевая функция составляет:',curr_res,'номер итерации:',n_iter) ## это нужно выводить
        prev_res = curr_res


def optimization_circle(h_t, power, eps=0.001, xx=None, maxiter=50):
    '''
    Цикл обучения, его необходимо запускать по запросу пользователя,
    В конце выведется ошибка, и предложение пользователю заново запустить оптимизацию
    Если он согласен, запускать заново на этих полученном xx
    '''

    if (xx is not None):
        print("Not None")
        print(power)
        print(xx.shape[0])
        if (power==2):
            if  (xx.shape[0]!=6):
                print(2)
                xx = np.random.uniform(0, 1, 6)
        if (power==3):
            if  (xx.shape[0]!=8):
                print(3)
                xx = np.random.uniform(0, 1, 8)
    # if xx is not None:
    #     print(xx.shape[0])
    if xx is None:
        print("инициализация")
        if power == 2:
            xx = np.random.uniform(0, 1, 6)
        elif power == 3:
            xx = np.random.uniform(0, 1, 8)


    T, y_target = h_t[:, 0], h_t[:, 1]

    res = scipy.optimize.minimize(fun = optimize_target, x0=xx, args=(y_target, T, power), method='nelder-mead',
                   options={'xtol': 1e-8, 'disp': False})

    xx = np.copy(res.x)

    adam(xx, 0.001, 0.9, 0.999, y_target, T, power, 0.0001, maxiter)

    print('Итоговая ошибка составила:', optimize_target(xx, y_target, T, power))
    X, y = transfer_2_power(xx, T, power)
    error1=optimize_target(xx, y_target, T, power)
   # build_real_and_get(X, y, h_t)  # График

    return xx, X,y,error1