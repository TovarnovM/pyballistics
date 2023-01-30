from .termo cimport _convert_2_opts
from .options import get_full_options
from .termalconduct cimport fill_averages, TermalConductBarrel
from libc.math cimport fabs, sqrt, round, ceil, floor, trunc, pi, atan2, sin, cos
cimport cython
import numpy as np
import time

# def lagrange_termocond_barrel(opts_dict):
#     cdef:
#         LagrangeLayer layer
#         list results
#         list results_barrel
#         double t1 = time.perf_counter() 
#         double tau
#         TermalConductBarrel barrel
#     try:
#         layer = LagrangeLayer(opts_dict)
#         results = [layer.get_y0()]
#         barrel = TermalConductBarrel(
#             d = layer.opts_full['init_conditions']['d'],
#             d_1 = 3*layer.opts_full['init_conditions']['d'],
#             length0 = layer.xs[0],
#             length1 = layer.opts_full['stop_conditions'].get('x_p', 7),
#             n_cells = 100,
#             T_0 = layer.opts_full['heat']['T_w0'],
#             delta_b = layer.opts_full['heat']['rho_c'],
#             c_b = layer.opts_full['heat']['c_c'],
#             lambda_b = layer.opts_full['heat']['lambda_c']
#         )
#         results_barrel = [barrel.get_state()]
        
#         while not layer.stop_reason():
#             barrel.fill_T_ws(layer.xs, layer.T_ws)
#             barrel.fill_self_q_ws(layer.xs, layer.qs)
#             tau = layer.get_tau()
#             layer.step(layer.tau_Ku_filter_W(tau))
#             barrel.step(tau)
#             results.append(layer.get_result())
#             results_barrel.append(barrel.get_state())
#         return {
#             'stop_reason': layer.get_stop_reason(),
#             'layers': results,
#             'barrel': results_barrel,
#             'execution time': time.perf_counter() - t1
#         }

#     except Exception as e:
#         return {
#             'stop_reason': 'error',
#             'error_message': str(e),
#             'exception': e,
#             'execution time': time.perf_counter() - t1
#             }

def ozvb_lagrange(opts_dict):
    """
    Функция для решения ОЗВБ в газодинамической постановке в Лагранжевых координатах. В качестве входных данных может быть использован 
    словарь с неполными начальными данными (оставшиеся данные будут заполнены значениями по-умолчанию и проверены на правильность 
    при помощи функции get_full_options)

    :param opts: Словарь с начальными данными (может быть неплоным)
    :type opts: dict
    :return: Словарь с результатами расчета
    :rtype: dict


    Структура словаря с разультатами.
    В зависимости от результатов расчета, словарь может быть двух видов. 
    Если в результате расчета произошла ошибка, то будет сформирован следующий словарь:

    {
        'stop_reason': 'error',   # показывает, что в процессе расчета произошла ошибка
        'error_message': '...',   # описание ошибки
        'exception': Error('...'),# ссылка на саму ошибку (ее можно вызвать при помощи raise для трассировки) 
        'execution_time': float   # время выполнения функции в секундках
    }

    Пример:

    >>> result = ozvb_lagrange({})  # передаем пустой словарь
    >>> print(result)
    {
        'stop_reason': 'error',
        'error_message': 'В словаре opts обязательно должно быть поле "powders", в котором указываются параметры заряда. Пример правильного словаря opts можно получить из функции get_termo_options_sample()',
        'exception': ValueError('В словаре opts обязательно должно быть поле "powders", в котором указываются параметры заряда. Пример правильного словаря opts можно получить из функции get_termo_options_sample()'),
        'execution_time': 1.7400000047018693e-05
    }


    Если расчет прошел без ошибок, то словарь с результатами будет следующий:

    {
        'stop_reason': str,     # причина остановки расчета ('t_max', 'steps_max', 'v_p', 'x_p', 'p_max')
        'execution_time': float,# время выполнения расчета в секундках
        'layers': [             # список со словарями. В каждом словаре хранятся данные одного временного слоя
            {                       # Словарь первого временного слоя. Слой состоит из N ячеек
                't': 0.0,               # время временного слоя в секундах
                'step_count': 0,         # номер шага по времени
                'x': np.array([...]),    # numpy-массив координатами по длине узлов сетки в метрах, длина массива N+1
                'u': np.array([...]),    # numpy-массив со скоростями узлов сетки в м/с, длина массива N+1
                'T': np.array([...]),    # numpy-массив с температурами ГПС в ячейках в Кельвинах. Длина массива N
                'rho': np.array([...]),  # numpy-массив с плотностями ГПС в ячейках в кг/м^3. Длина массива N
                'p': np.array([...]),    # numpy-массив с давлениями ГПС в ячейках в Па. Длина массива N
                'T_w':np.array([...]),   # numpy-массив с температурами стенок ствола в ячейках в Кельвинах. Длина массива N
                'k':  np.array([...]),   # numpy-массив с показателями адиабаты ГПС в ячейках. Длина массива N
                'z_1': np.array([...]),  # numpy-массив с относительными толщинами сгоревшего свода пороха навески №1 по ячейкам. Длина массива N 
                'psi_1': np.array([...]),# numpy-массив с относительными массами сгоревшего пороха навески №1 по ячейкам. Длина массива N 
                'z_2':np.array([...]),   # numpy-массив с относительными толщинами сгоревшего свода пороха навески №2 по ячейкам. Длина массива N 
                'psi_2': np.array([...]),# numpy-массив с относительными массами сгоревшего пороха навески №2 по ячейкам. Длина массива N 
                ... # и так до 'z_N', 'psi_N'
            },
            {...},                 # Словарь второго временного слоя. Слой состоит из N ячеек
            {...},                 # Словарь третьего временного слоя. Слой состоит из N ячеек
            ...,                   № и т.д.
        ]     # конец списка 'layers'
    }


    Пример:

    >>> opts = get_options_sample()
    >>> result = ozvb_lagrange(opts)
    >>> print(result)
    {
        'stop_reason': 'v_p',
        'execution_time': 0.167843300000186, 
        'layers': [
            {
                't': 0.0,
                'step_count': 0,
                'x': array([-2.78 , -2.762, ..., -0.019,  0.   ]),
                'u': array([0., 0., ..., 0., 0.]),
                'T': array([2427., 2427., ..., 2427., 2427.]),
                'rho': array([402.851, 402.851, ..., 402.851, 402.851]),
                'p': array([1000000., 1000000., ..., 1000000., 1000000.]),
                'T_w': array([293.15, 293.15, ..., 293.15, 293.15]),
                'k': array([1.22, 1.22, ..., 1.22, 1.22]),
                'z_1': array([0., 0., ..., 0., 0.]),
                'psi_1': array([0., 0., ..., 0., 0.]),
                'z_2': array([0., 0., ..., 0., 0.]),
                'psi_2': array([0., 0., ..., 0., 0.])
            },
            {
                't': 0.00026096741712768897,
                'step_count': 1,
                'x': array([-2.78 , -2.762, ..., -0.019,  0.   ]),
                'u': array([0., 0., ..., 0., 0.]),
                'T': array([2450.216, 2450.216, ..., 2450.216, 2450.216]),
                'rho': array([402.851, 402.851, ..., 402.851, 402.851]),
                'p': array([1114231.986, 1114231.986, ..., 1114231.986, 1114231.986]),
                'T_w': array([293.15, 293.15, ..., 293.15, 293.15]),
                'k': array([1.222, 1.222, ..., 1.222, 1.222]),
                'z_1': array([0., 0., ..., 0., 0.]),
                'psi_1': array([0., 0., ..., 0., 0.]),
                'z_2': array([0., 0., ..., 0., 0.]),
                'psi_2': array([0., 0., ..., 0., 0.])
            },
            ...
        ]
    }
    """
    cdef:
        LagrangeLayer layer
        list results
        double t1 = time.perf_counter() 
        double tau
        str stop_reason
    try:
        layer = LagrangeLayer(opts_dict)
        results = [layer.get_y0()]
        while not layer.stop_reason():
            tau = layer.get_tau()
            layer.step(layer.tau_Ku_filter_W(tau))
            results.append(layer.get_result())
        stop_reason = layer.get_stop_reason()
        layer.trim_results(results, stop_reason)
        return {
            'stop_reason': stop_reason,
            'execution_time': time.perf_counter() - t1,
            'layers': results
        }

    except Exception as e:
        return {
            'stop_reason': 'error',
            'error_message': str(e),
            'exception': e,
            'execution_time': time.perf_counter() - t1
            }


cdef inline double get_psi(double z, Opts* opts, Py_ssize_t i)  nogil:
    cdef double psi = 0.0
    if  z <= 1.0:
        psi = opts[0].powders[i].kappa_1 * z*(1 + opts[0].powders[i].lambda_1*z + opts[0].powders[i].mu_1*z*z) 
    elif z < opts[0].powders[i].z_e:
        psi = opts[0].powders[i].kappa_1 * (1 + opts[0].powders[i].lambda_1+ opts[0].powders[i].mu_1) + opts[0].powders[i].kappa_2 * (z-1)*(1 + opts[0].powders[i].lambda_2*(z-1)+ opts[0].powders[i].mu_2*(z-1)*(z-1)) 
    else: 
        return 1.0
    if psi > 1.0:
        return 1.0
    if psi < 0:
        return 0.0
    return psi


cdef inline double H(double x)  nogil:
    return 1.0 if x > 0 else 0.0


@cython.final
cdef class LagrangeLayer:
    def __cinit__(self, opts_dict: dict):
        self.opts_full = get_full_options(opts_dict)
        self.mem = Pool()
        self.opts = _convert_2_opts(self.opts_full, self.mem)
        self.meta = self._get_meta(self.opts_full)
        self.step_count = 0
        self.t = 0.0

    cdef MetaLagrangeOpts _get_meta(self, dict opts_full):
        if 'meta_lagrange' not in opts_full:
            raise ValueError(f'В словаре с начальными данными обязательно должен быть раздел "meta_lagrange", в котором указываются параметры интегрирования')
        cdef MetaLagrangeOpts meta = {
            'n_cells': opts_full['meta_lagrange']['n_cells'],
            'CFL': opts_full['meta_lagrange']['CFL'],
            'W': opts_full['meta_lagrange'].get('W', 2048.0)
        }
        return meta

    cpdef double get_omega_ign(self):
        cdef double om_delta_sum = 0.0
        cdef Py_ssize_t i
        for i in range(self.opts.n_powders):
            om_delta_sum += self.opts.powders[i].omega / self.opts.powders[i].rho
        cdef double omega_ign = self.opts.igniter.p_ign_0 / self.opts.igniter.f_ign * (self.opts.init_conditions.W_0 - om_delta_sum) / (1 + \
                        self.opts.igniter.b_ign * self.opts.igniter.p_ign_0 / self.opts.igniter.f_ign)
        return omega_ign

    cpdef dict get_y0(self):
        cdef:
            Py_ssize_t i, j
            int n_powders = self.opts.n_powders
            int n_cells = self.meta.n_cells
            double omega_ign = self.get_omega_ign()
            double l_0 = self.opts.init_conditions.W_0 / self.opts.init_conditions.S
        
        self.ps = np.zeros(n_cells, dtype=np.float64)
        self.xs = np.linspace(-l_0, 0, n_cells+1, dtype=np.float64)
        self.us = np.zeros(n_cells+1, dtype=np.float64) 
        self.Ts = np.zeros(n_cells, dtype=np.float64)
        self.es = np.zeros(n_cells, dtype=np.float64)
        self.rhos = np.zeros(n_cells, dtype=np.float64)
        self.cs = np.zeros(n_cells, dtype=np.float64)
        self.Ws = np.zeros(n_cells, dtype=np.float64)
        self.T_ws = np.zeros(n_cells, dtype=np.float64) 
        self.m_sums = np.zeros(n_cells, dtype=np.float64)
        self.ks = np.zeros(n_cells, dtype=np.float64)
        self.W_cs = np.zeros(n_cells, dtype=np.float64)
        self.qs = np.zeros(n_cells, dtype=np.float64)
        self.Rs = np.zeros(n_cells, dtype=np.float64)
        self.dEta = np.zeros(n_cells, dtype=np.float64)
        self.xs_tmp = np.zeros_like(self.xs)
        self.T_ws_tmp = np.zeros_like(self.T_ws)

        self.zs = np.zeros((n_cells, n_powders+1), dtype=np.float64)
        self.psis = np.zeros((n_cells, n_powders+1), dtype=np.float64)
        self.omegas = np.zeros((n_cells, n_powders+1), dtype=np.float64)

        self.zs[:, 0] = 1
        self.psis[: ,0] = 1
        
        cdef double om_sum = omega_ign
        self.omegas[:, 0] = omega_ign/n_cells
        for i in range(self.opts.n_powders):
            self.omegas[:, i+1] = self.opts.powders[i].omega / n_cells
            om_sum += self.opts.powders[i].omega
        self.m_sums[:] = om_sum / n_cells
        self.ps[:] = self.opts.igniter.p_ign_0
        self.T_ws[:] = self.opts.heat.T_w0

        self.synch_Ws()
        self.synch_rhos()
        self.synch_ks()
        self.synch_es_Ts_Wcs_cs()
        self.synch_qs()

        self.step_count = 0
        self.t = 0
        self.tau_last = 999
        self.znam_eta = 2 * self.opts.heat.lambda_p**2 / (self.opts.init_conditions.d ** 2 * self.opts.heat.c_c * self.opts.heat.rho_c * self.opts.heat.lambda_c)
        return self.get_result()

    @cython.initializedcheck(False)
    cpdef dict get_state(self):
        return {
            't': self.t,
            'step_count': self.step_count,
            'tau_last': self.tau_last,
            'xs': np.array(self.xs),
            'us': np.array(self.us),
            'rhos': np.array(self.rhos),
            'es': np.array(self.es),
            'ps': np.array(self.ps),
            'zs': np.array(self.zs),
            'omegas': np.array(self.omegas),

            'Ts': np.array(self.Ts),
            'cs': np.array(self.cs),
            'psis': np.array(self.psis),
            'T_ws': np.array(self.T_ws)
        }

    cpdef void set_state(self, dict state):
        self.t = state['t']
        self.step_count = state['step_count']
        self.tau_last = state['tau_last']
        self.xs = state['xs']
        self.us = state['us']
        self.ps = state['ps']
        self.zs = state['zs']
        self.omegas = state['omegas']
        self.T_ws = state['T_ws']
        self.synch_Ws()
        self.synch_rhos()
        self.synch_psis()
        self.synch_ks()
        self.synch_es_Ts_Wcs_cs()
        self.synch_qs()

    @cython.initializedcheck(False)
    cpdef dict get_result(self):
        cdef dict res = {
            't': self.t,
            'step_count': self.step_count,
            'x': np.array(self.xs),
            'u': np.array(self.us),
            'T': np.array(self.Ts),
            'rho': np.array(self.rhos),
            'p': np.array(self.ps),
            'T_w': np.array(self.T_ws),
            'k': np.array(self.ks)
        }
        cdef:
            Py_ssize_t i, j
        for i in range(1, self.zs.shape[1]):
            res[f'z_{i}'] = np.array(self.zs[:, i])
            res[f'psi_{i}'] = np.array(self.psis[:, i])
        return res

    @cython.initializedcheck(False)
    cpdef void synch_rhos(self) nogil:
        cdef:
            Py_ssize_t i
            double S = self.opts.init_conditions.S
        for i in range(self.rhos.shape[0]):
            self.rhos[i] =  self.m_sums[i]/self.Ws[i]

    @cython.initializedcheck(False)
    cpdef void synch_psis(self) nogil:
        cdef:
            Py_ssize_t i, j
        for j in range(self.psis.shape[1]):
            self.psis[0, j] = 1.0 # воспламенитель
        for i in range(self.psis.shape[0]):
            for j in range(1, self.psis.shape[1]):
                self.psis[i, j] = get_psi(self.zs[i, j], &(self.opts), j-1)

    @cython.initializedcheck(False)
    cpdef void synch_Ws(self) nogil:
        cdef:
            Py_ssize_t i
            double S = self.opts.init_conditions.S
        for i in range(self.Ws.shape[0]):
            self.Ws[i] =  (self.xs[i+1] - self.xs[i])*S


    @cython.initializedcheck(False)
    cpdef void synch_ks(self) nogil:
        cdef:
            Py_ssize_t i, j
            double chisl, znam, tmp
        for i in range(self.ks.shape[0]):
            chisl = self.omegas[i, 0] * self.opts.igniter.R_ign
            znam = chisl / (self.opts.igniter.k_ign - 1)
            for j in range(1, self.psis.shape[1]):
                tmp = self.psis[i, j] * self.omegas[i, j] * self.opts.powders[j-1].R
                chisl += tmp
                znam += tmp / (self.opts.powders[j-1].k - 1)
            self.ks[i] = 1 + chisl / znam

    @cython.initializedcheck(False)
    cpdef void synch_es_Ts_Wcs_cs(self) nogil:
        cdef:
            Py_ssize_t i, j
            double s1, s2, s3, tmp, psi_j, W_c, s4, R
        for i in range(self.es.shape[0]):
            s1 = 0
            s2 = self.omegas[i, 0] / self.m_sums[i] * self.opts.igniter.b_ign
            s3 = 0
            W_c = s2 * self.m_sums[i]
            s4 = self.omegas[i, 0] * self.opts.igniter.R_ign
            for j in range(1, self.psis.shape[1]):
                tmp = self.omegas[i, j] / self.m_sums[i] 
                psi_j = self.psis[i, j]
                s1 += (1 - psi_j) * tmp / self.opts.powders[j-1].rho
                s2 += tmp * psi_j * self.opts.powders[j-1].b
                s3 += (1 - psi_j) * tmp * self.opts.powders[j-1].f / (self.opts.powders[j-1].k - 1)
            
                W_c += (1 - psi_j) * self.omegas[i, j] / self.opts.powders[j-1].rho +  self.omegas[i, j] * psi_j * self.opts.powders[j-1].b
                s4 += psi_j * self.omegas[i, j] * self.opts.powders[j-1].R
            self.es[i] = 1/(self.ks[i] - 1) * self.ps[i] * (1/self.rhos[i] - s1 - s2) + s3
            self.W_cs[i] = W_c
            self.Ts[i] = self.ps[i] * (self.Ws[i] - self.W_cs[i]) / s4
            self.cs[i] = sqrt(self.ks[i]* self.ps[i] /(1/self.rhos[i] - s1 - s2)) / self.rhos[i]
            self.Rs[i] = s4 / self.m_sums[i]

    @cython.initializedcheck(False)
    cpdef void synch_ps_Ts_Wcs_cs(self) nogil:
        cdef:
            Py_ssize_t i, j
            double s1, s2, s3, tmp, psi_j, W_c, s4
        for i in range(self.es.shape[0]):
            s1 = 0
            s2 = self.omegas[i, 0] / self.m_sums[i] * self.opts.igniter.b_ign
            s3 = 0
            W_c = s2 * self.m_sums[i]
            s4 = self.omegas[i, 0] * self.opts.igniter.R_ign
            for j in range(1, self.psis.shape[1]):
                tmp = self.omegas[i, j] / self.m_sums[i] 
                psi_j = self.psis[i, j]
                s1 += (1 - psi_j) * tmp / self.opts.powders[j-1].rho
                s2 += tmp * psi_j * self.opts.powders[j-1].b
                s3 += (1 - psi_j) * tmp * self.opts.powders[j-1].f / (self.opts.powders[j-1].k - 1)
            
                W_c += (1 - psi_j) * self.omegas[i, j] / self.opts.powders[j-1].rho +  self.omegas[i, j] * psi_j * self.opts.powders[j-1].b
                s4 += psi_j * self.omegas[i, j] * self.opts.powders[j-1].R
            self.ps[i] = (self.es[i] - s3) * (self.ks[i] - 1) / (1/self.rhos[i] - s1 - s2) 
            self.W_cs[i] = W_c
            self.Ts[i] = self.ps[i] * (self.Ws[i] - self.W_cs[i]) / s4
            self.cs[i] = sqrt(self.ks[i]* self.ps[i] /(1/self.rhos[i] - s1 - s2)) / self.rhos[i]
            self.Rs[i] = s4 / self.m_sums[i]

    @cython.initializedcheck(False)
    cpdef void step(self, double tau) nogil:
        cdef:
            Py_ssize_t i, j, i_1 = self.us.shape[0]-1
            double S = self.opts.init_conditions.S
            double d = self.opts.init_conditions.d
            double p_a = self.get_p_a(self.us[i_1])
            double eta, x_tmp, v_lft, v_rgt

        self.us[0] = 0
        for i in range(1, i_1):
            self.us[i] -= tau * S * (self.ps[i] - self.ps[i-1]) / (0.5 * (self.m_sums[i-1]+self.m_sums[i]))
        if self.ps[i_1-1] - p_a - self.opts.init_conditions.p_0 > 0 or fabs(self.us[i_1])>1e-8:
            self.us[i_1] += tau *(self.ps[i_1-1]-p_a) * S / (self.opts.init_conditions.q * self.opts.init_conditions.phi_1 + 0.5 * self.m_sums[i_1-1])
        else:
            self.us[i_1] = 0.0

        self.xs_tmp[:] = self.xs
        for i in range(self.xs.shape[0]):
            self.xs[i] += self.us[i] * tau

        if self.opts.heat.heat_barrel:
            for i in range(self.T_ws.shape[0]):
                eta = (self.T_ws[i] - self.opts.heat.T_w0) 
                eta *= eta
                eta += tau * self.dEta[i]
                self.T_ws[i] = sqrt(eta) + self.opts.heat.T_w0
            self.T_ws_tmp[:] = self.T_ws[:]
            fill_averages(self.T_ws_tmp, self.xs_tmp, self.xs, self.T_ws)
            i = self.T_ws.shape[0] - 1
            x_tmp = self.xs_tmp[i+1] - self.xs_tmp[i]
            v_lft = self.us[i]
            v_rgt = self.us[i+1]
            self.T_ws[i] = self.T_ws_tmp[i] * (x_tmp - tau * v_lft) / (x_tmp + tau * (-v_lft + v_rgt)) + self.opts.heat.T_w0 * (tau * v_rgt) / (x_tmp + tau * (-v_lft + v_rgt))
            
        self.synch_Ws()
        self.synch_rhos()
        
        for i in range(self.es.shape[0]):
            self.es[i] -= tau * (self.ps[i] * S * (self.us[i+1] - self.us[i])/self.m_sums[i] + 4 * self.qs[i] / self.rhos[i] / d)
        for i in range(self.zs.shape[0]):
            for j in range(1, self.zs.shape[1]):
                self.zs[i, j] += tau * (self.ps[i] ** self.opts.powders[j-1].nu) / self.opts.powders[j-1].I_k * H(self.opts.powders[j-1].z_e - self.zs[i, j])
       
        self.synch_psis()
        self.synch_ks()
        self.synch_ps_Ts_Wcs_cs()
        self.synch_qs()
        self.step_count += 1
        self.t += tau
        self.tau_last = tau

    @cython.initializedcheck(False)
    cpdef double get_p_a(self, double v_p) nogil:
        cdef:
            double p_a = self.opts.windage.p_0a
            double k_air, c_0a, mach, mach_
        if self.opts.windage.shock_wave:  
            k_air = self.opts.windage.k_air
            c_0a = self.opts.windage.c_0a
            mach = v_p/c_0a
            mach_ = (k_air+1)*v_p/4/c_0a           
            p_a *= 1 + k_air*(k_air+1)/4*(mach*mach) + k_air*mach*sqrt(1+mach_*mach_)
        return p_a
          
    @cython.initializedcheck(False)
    cpdef bint stop_reason(self) nogil:
        cdef double p_max
        cdef Py_ssize_t i
        if self.opts.stop_conditions.t_max_flag:
            if self.t >= self.opts.stop_conditions.t_max:
                return 1
        if self.opts.stop_conditions.steps_max_flag:
            if self.step_count >= self.opts.stop_conditions.steps_max:
                return 2
        if self.opts.stop_conditions.v_p_flag:
            if self.us[self.us.shape[0]-1] >= self.opts.stop_conditions.v_p:
                return 3
        if self.opts.stop_conditions.x_p_flag:
            if self.xs[self.xs.shape[0]-1] >= self.opts.stop_conditions.x_p:
                return 4
        if self.opts.stop_conditions.p_max_flag:
            p_max = self.opts.stop_conditions.p_max
            for i in range(self.ps.shape[0]):
                if self.ps[i] >= p_max:
                    return 5
        return 0

    @cython.initializedcheck(False)
    cpdef double get_tau(self) nogil:    
        cdef double tau_min = 1e10
        cdef double tau, CFL = self.meta.CFL
        cdef Py_ssize_t i
        for i in range(self.cs.shape[0]):
            tau = CFL * (self.xs[i+1] - self.xs[i]) / (fabs(0.5*(self.us[i] + self.us[i+1])) + self.cs[i])
            if tau < tau_min:
                tau_min = tau
        return tau_min

    @cython.initializedcheck(False)
    cpdef double tau_Ku_filter_W(self, double tau) nogil: 
        if tau > self.tau_last * self.meta.W:
            return self.tau_last * self.meta.W
        return tau

    cpdef str get_stop_reason(self):
        cdef int reason = self.stop_reason()
        if reason == 1:
            return 't_max'
        elif reason == 2:
            return 'steps_max'
        elif reason == 3:
            return 'v_p'
        elif reason == 4:
            return 'x_p'
        elif reason == 5:
            return 'p_max'
        return 'Unknown'

    @cython.initializedcheck(False)
    cpdef void synch_qs(self) nogil:
        cdef:
            Py_ssize_t i
            double mu, Re, S_w, v_eta, vi, Nu

        if not self.opts.heat.enabled:
            self.qs[:] = 0.0
            return

        for i in range(self.qs.shape[0]):
                                   # * self.opts.heat.Pr
            mu = self.opts.heat.mu_0 * ((self.Ts[i]/self.opts.heat.T_0)**1.5) * \
                (self.opts.heat.T_c + self.opts.heat.T_0)/(self.opts.heat.T_c + self.Ts[i])
            Re = self.rhos[i]* 0.5*fabs(self.us[i+1]+self.us[i]) * self.opts.init_conditions.d / mu
            Nu = 0.023*(Re**0.8)*(self.opts.heat.Pr**0.4)

            # # dQ_w/dt 
            # S_w = (self.xs[i+1] - self.xs[i]) * self.opts.init_conditions.d * pi
            if self.opts.heat.Sigma_T < 0:
                self.qs[i] = Nu * self.opts.heat.lambda_p / self.opts.init_conditions.d * (self.Ts[i] - self.T_ws[i])
            else:
                if fabs(self.opts.heat.vi) > 1e-8:
                    vi = self.opts.heat.vi
                else:
                    vi = 1 - self.T_ws[i]/ self.Ts[i]
                self.qs[i] = self.ps[i] * self.opts.heat.Sigma_T * vi / self.Rs[i]
            # d eta_T/dt
            if self.opts.heat.heat_barrel:
                self.dEta[i] = Nu**2 * self.znam_eta * (self.Ts[i] - self.T_ws[i]) * (self.Ts[i] - self.T_ws[i]) 

    @cython.wraparound(True)
    def trim_results(self, results, reason):
        if reason == 'steps_max':
            return

        layer1 = results[-2]
        layer2 = results[-1]
        if reason == 't_max':
            x = self.opts.stop_conditions.t_max
            x1 = layer1['t']
            x2 = layer2['t']

        elif reason == 'v_p':
            x = self.opts.stop_conditions.v_p
            x1 = layer1['u'][-1]
            x2 = layer2['u'][-1]

        elif reason == 'x_p':
            x = self.opts.stop_conditions.x_p
            x1 = layer1['x'][-1]
            x2 = layer2['x'][-1]

        elif reason == 'p_max':
            x = self.opts.stop_conditions.p_max
            x1 = np.max(layer1['p'])
            x2 = np.max(layer2['p'])
        else:
            raise ValueError(f'Неясная причина остановки reason = {reason}')
        
        t = (x - x1) / (x2 - x1)
        for k in layer2:
            if k == 'steps_max':
                continue
            layer2[k] = (1-t) * layer1[k] + t * layer2[k]
            



    
    