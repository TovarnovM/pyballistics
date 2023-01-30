
from libc.math cimport fabs, sqrt, round, ceil, floor, trunc, pi, atan2, sin, cos
cimport cython
import numpy as np
from .options import get_full_options
import time


def ozvb_termo(opts_dict):
    """
    Функция для решения ОЗВБ в термодинамической постановке. В качестве входных данных может быть использован 
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

    >>> result = ozvb_termo({})  # передаем пустой словарь
    >>> print(result)
    {
        'stop_reason': 'error',
        'error_message': 'В словаре opts обязательно должно быть поле "powders", в котором указываются параметры заряда. Пример правильного словаря opts можно получить из функции get_termo_options_sample()',
        'exception': ValueError('В словаре opts обязательно должно быть поле "powders", в котором указываются параметры заряда. Пример правильного словаря opts можно получить из функции get_termo_options_sample()'),
        'execution_time': 1.7400000047018693e-05
    }


    Если расчет прошел без ошибок, то словарь с результатами будет следующий:

    {
        't': np.array([...]),     # numpy-массив с точками по времени в секундах, в которых были рассчитаны остальные значения 
        'p_m': np.array([...]),   # numpy-массив со среднебаллистическим давлением в Па
        'T': np.array([...]),     # numpy-массив с темперетурой ГПС в Кельвинах
        'x_p':np.array([...]),    # numpy-массив с положением снаряда в метрах (в начальный момент x_p==0)
        'v_p': np.array([...]),   # numpy-массив со скоростью снаряда в м/c (в начальный момент v_p==0)
        'Q_pa': np.array([...]),  # numpy-массив с суммарной энергией в Дж, потраченной на преодоление сил сопротивления атмосфеному давлению перед снарядом
        'Q_w': np.array([...]),   # numpy-массив с суммарной энергией в Дж, отданой ГПС на нагрев ствола
        'W_p': np.array([...]),   # numpy-массив с заснарядным объемом в м^3
        'W_c': np.array([...]),   # numpy-массив с объемом в м^3, занятым коволюмом ГПС и конденсированной фазой ГПС
        'T_w': np.array([...]),   # numpy-массив со средней температурой ствола в К
        'k': np.array([...]),     # numpy-массив с показателями адиабаты ГПС
        'z_1': np.array([...]),   # numpy-массив с относительной толщиной сгоревшего свода пороха навески №1
        'psi_1': np.array([...]), # numpy-массив с относительной массой сгоревшего пороха навески №1
        'z_2': np.array([...]),   # numpy-массив с относительной толщиной сгоревшего свода пороха навески №2
        'psi_2': np.array([...]), # numpy-массив с относительной массой сгоревшего пороха навески №2
            ...                   # и так N раз
        'stop_reason': str,       # причина остановки расчета ('t_max', 'steps_max', 'v_p', 'x_p', 'p_max')
        'execution_time': float   # время, потраченное на расчет, в секундках
    }

    Пример:

    >>> opts = get_options_sample()
    >>> result = ozvb_termo(opts)
    >>> print(result)
    {
        't':    array([0.   , 0.   , ..., 0.027, 0.027]),
        'p_m':  array([ 1000000.   ,  1002189.433, ..., 90680294.893, 90629603.46 ]),
        'T':    array([2427.   , 2427.487, ..., 1824.249, 1823.988]),
        'x_p':  array([0.   , 0.   , ..., 6.394, 6.398]),
        'v_p':  array([  0.   ,   0.   , ..., 689.994, 690.085]),
        'Q_pa': array([    0.   ,     0.   , ..., 45159.509, 45195.554]),
        'Q_w':  array([      0.   ,       0.   , ..., 3447622.549, 3449318.738]),
        'W_p':  array([0.033, 0.033, ..., 0.107, 0.107]),
        'W_c':  array([0.008, 0.008, ..., 0.014, 0.014]),
        'T_w':  array([293.15 , 293.15 , ..., 315.661, 315.661]),
        'k':    array([1.22 , 1.22 , ..., 1.238, 1.238]),
        'z_1':  array([0.   , 0.   , ..., 0.954, 0.954]),
        'psi_1':array([0.   , 0.   , ..., 0.954, 0.954]),
        'z_2':  array([0.   , 0.   , ..., 1.343, 1.343]),
        'psi_2':array([0.   , 0.   , ..., 0.987, 0.987]),
        'stop_reason': 'v_p',
        'execution_time': 0.21484209999971426
    }

    """
    cdef:
        Pool mem = Pool()
        Opts opts
        Cached cache
        MetaTermoOpts meta_termo
        double[:] y, stuff, y1, stuff1,  
        double t1, t, t_ex, t_interpolate
        double[:,:] y_tmps 
        list ys, stuffs, ts
        int n_steps = 0
        int reason 
        dict results
        dict opts_full
    try:
        t_ex = time.perf_counter()
        opts_full = get_full_options(opts_dict)
        opts = _convert_2_opts(opts_full, mem)
        if 'meta_termo' not in opts_full:
            raise ValueError(f'В словаре с начальными данными обязательно должен быть раздел "meta_termo", в котором указываются параметры интегрирования')
        cache, meta_termo = _get_cache_meta(opts_full, opts, mem)
        y = _get_y0(&opts)
        stuff = _get_stuff(y, &opts, &cache)
        t = 0.0
        y_tmps = np.zeros((4,y.shape[0]), dtype=np.float64)
        ys = [y]
        stuffs = [stuff]
        ts = [t]
        while _stop_reason(t, y, stuff, &opts, n_steps) == 0:
            y1 = np.empty_like(y)
            stuff1 = np.empty_like(stuff)
            _step(t, y, stuff, &opts, &cache, y_tmps, &t1, y1, stuff1, &meta_termo)
            ys.append(y1)
            stuffs.append(stuff1)
            ts.append(t1)
            n_steps += 1
            t = t1
            y = y1
            stuff = stuff1
        reason = _stop_reason(t, y, stuff, &opts, n_steps)
        trim_interpolate_results(ts, ys, stuffs, &opts, n_steps, reason)
        results = _construct_results(ts, ys, stuffs, reason)
        results['execution_time'] = time.perf_counter() - t_ex
        return results
    except Exception as e:
        return {
            'stop_reason': 'error',
            'error_message': str(e),
            'exception': e,
            'execution_time': time.perf_counter() - t_ex
            }

cdef Opts _convert_2_opts(dict opts_full, Pool mem):
    cdef Opts opts 
    opts.n_powders = len(opts_full['powders'])
    opts.powders = <PowderOpts*>mem.alloc(opts.n_powders, sizeof(PowderOpts))
    cdef Py_ssize_t i
    for i in range(opts.n_powders):
        opts.powders[i].omega = opts_full['powders'][i]['omega']
        opts.powders[i].I_k =   opts_full['powders'][i]['I_e']
        opts.powders[i].nu =    opts_full['powders'][i]['nu']
        opts.powders[i].b =     opts_full['powders'][i]['b']
        opts.powders[i].rho =   opts_full['powders'][i]['delta']
        opts.powders[i].f =     opts_full['powders'][i]['f']
        opts.powders[i].k =     opts_full['powders'][i]['k']
        opts.powders[i].T_1 =   opts_full['powders'][i]['T_p']
        opts.powders[i].z_e =   opts_full['powders'][i]['z_e']
        opts.powders[i].kappa_1 =  opts_full['powders'][i]['kappa_1']
        opts.powders[i].lambda_1 = opts_full['powders'][i]['lambda_1']
        opts.powders[i].mu_1 = opts_full['powders'][i]['mu_1']
        opts.powders[i].kappa_2 =  opts_full['powders'][i]['kappa_2']
        opts.powders[i].lambda_2 = opts_full['powders'][i]['lambda_2']
        opts.powders[i].mu_2 = opts_full['powders'][i]['mu_2']
        opts.powders[i].k_I =   opts_full['powders'][i]['k_I']
        opts.powders[i].k_f =   opts_full['powders'][i]['k_f']
        opts.powders[i].R = opts_full['powders'][i]['R'] #opts.powders[i].f / opts.powders[i].T_1
        opts.powders[i].M = opts_full['powders'][i]['M'] # 8.31446261815324 / opts.powders[i].R

    opts.init_conditions.q =   opts_full['init_conditions']['q']
    opts.init_conditions.d =   opts_full['init_conditions']['d']
    opts.init_conditions.W_0 = opts_full['init_conditions']['W_0']
    opts.init_conditions.T_0 = opts_full['init_conditions']['T_0']
    opts.init_conditions.S =   opts_full['init_conditions']['S']
    opts.init_conditions.phi_1 = opts_full['init_conditions']['phi_1']
    opts.init_conditions.p_0 = opts_full['init_conditions']['p_0']

    for i in range(opts.n_powders):
        opts.powders[i].I_k *= 1 - opts.powders[i].k_I * (opts.init_conditions.T_0 - 293.15)
        opts.powders[i].f *= 1 + opts.powders[i].k_f * (opts.init_conditions.T_0 - 293.15)
    
    opts.igniter.p_ign_0 = opts_full['igniter']['p_ign_0']
    opts.igniter.k_ign = opts_full['igniter']['k_ign']
    opts.igniter.T_ign = opts_full['igniter']['T_ign']
    opts.igniter.f_ign = opts_full['igniter']['f_ign']
    opts.igniter.b_ign = opts_full['igniter']['b_ign']
    opts.igniter.R_ign = opts_full['igniter']['R_ign']
    opts.igniter.M_ign = opts_full['igniter']['M_ign']


    opts.windage.shock_wave = opts_full['windage']['shock_wave']
    opts.windage.p_0a = opts_full['windage']['p_0a']
    opts.windage.k_air = opts_full['windage']['k_air']
    opts.windage.c_0a = opts_full['windage']['c_0a']

    opts.heat.enabled = opts_full['heat']['enabled']
    opts.heat.heat_barrel = opts_full['heat']['heat_barrel']
    opts.heat.F_0 = opts_full['heat']['F_0']
    opts.heat.Pr = opts_full['heat']['Pr']
    opts.heat.T_w0 = opts_full['heat']['T_w0']
    opts.heat.mu_0 = opts_full['heat']['mu_0']
    opts.heat.T_c = opts_full['heat']['T_cs']
    opts.heat.T_0 = opts_full['heat']['T_0s']
    opts.heat.c_c = opts_full['heat']['c_b']
    opts.heat.rho_c = opts_full['heat']['rho_b']
    opts.heat.lambda_c = opts_full['heat']['lambda_b']
    opts.heat.lambda_p = opts_full['heat']['lambda_g']
    opts.heat.Sigma_T = opts_full['heat']['Sigma_T']
    opts.heat.vi = opts_full['heat']['vi']

    opts.stop_conditions.t_max_flag = 't_max' in opts_full['stop_conditions']
    if opts.stop_conditions.t_max_flag:
        opts.stop_conditions.t_max = opts_full['stop_conditions']['t_max']
    
    opts.stop_conditions.steps_max_flag = 'steps_max' in opts_full['stop_conditions']
    if opts.stop_conditions.steps_max_flag:
        opts.stop_conditions.steps_max = opts_full['stop_conditions']['steps_max']
    
    opts.stop_conditions.v_p_flag = 'v_p' in opts_full['stop_conditions']
    if opts.stop_conditions.v_p_flag:
        opts.stop_conditions.v_p = opts_full['stop_conditions']['v_p']
    
    opts.stop_conditions.x_p_flag = 'x_p' in opts_full['stop_conditions']
    if opts.stop_conditions.x_p_flag:
        opts.stop_conditions.x_p = opts_full['stop_conditions']['x_p']
    
    opts.stop_conditions.p_max_flag = 'p_max' in opts_full['stop_conditions']
    if opts.stop_conditions.p_max_flag:
        opts.stop_conditions.p_max = opts_full['stop_conditions']['p_max']
    
    return opts

cdef (Cached, MetaTermoOpts) _get_cache_meta(dict opts_full, Opts opts, Pool mem):
    cdef MetaTermoOpts meta_termo

    meta_termo.dt = opts_full['meta_termo']['dt']
    meta_termo.method = euler if opts_full['meta_termo']['method'] == 'euler' else \
                            rk2   if opts_full['meta_termo']['method'] == 'rk2' else \
                            rk4

    cdef Cached cache
    cdef double om_delta_sum = 0.0
    for i in range(opts.n_powders):
        om_delta_sum += opts.powders[i].omega / opts.powders[i].rho
    cache.omega_ign = opts.igniter.p_ign_0 / opts.igniter.f_ign * (opts.init_conditions.W_0 - om_delta_sum) / (1 + \
            opts.igniter.b_ign * opts.igniter.p_ign_0 / opts.igniter.f_ign)
    cdef double om_sum = cache.omega_ign
    cdef double E_00 = 0
    for i in range(opts.n_powders):
        om_sum += opts.powders[i].omega
        E_00 += opts.powders[i].f * opts.powders[i].omega / (opts.powders[i].k - 1)
    cache.E_00 = E_00
    cache.om_sum = om_sum
    cache.phi = opts.init_conditions.phi_1 + om_sum / 3 / opts.init_conditions.q
    cache.E_0 = opts.igniter.p_ign_0 / (opts.igniter.k_ign - 1) * (opts.init_conditions.W_0 - om_delta_sum - cache.omega_ign*opts.igniter.b_ign) 
    cache.znam_ign = cache.omega_ign / (opts.igniter.k_ign - 1) * opts.igniter.f_ign / opts.igniter.T_ign 
    cache.chisl = <double*>mem.alloc(opts.n_powders, sizeof(double))
    for i in range(opts.n_powders):
        cache.chisl[i] = opts.powders[i].f * opts.powders[i].omega / (opts.powders[i].k - 1)
    cache.znam = <double*>mem.alloc(opts.n_powders, sizeof(double))
    for i in range(opts.n_powders):
        cache.znam[i] = opts.powders[i].f / opts.powders[i].T_1 * opts.powders[i].omega / (opts.powders[i].k - 1)
    cache.znam_eta = 2 * opts.heat.lambda_p**2 / (opts.init_conditions.d ** 2 * opts.heat.c_c * opts.heat.rho_c * opts.heat.lambda_c)
    cache.l_0 = opts.init_conditions.W_0 / opts.init_conditions.S
    cache.E_kin_v2 = opts.init_conditions.q * cache.phi / 2
    return cache, meta_termo


cdef double[:] _get_y0(Opts *opts):
    # [Q_Ap, Q_w, eta_T, x, v, z1, z2, z3, ...]
    # [0,    1,   2,     3, 4,  5,  6,  ...]
    cdef double[:] res = np.zeros(opts[0].n_powders + 5, dtype=np.float64)
    return res

cdef double[:] _get_stuff(double[:] y, Opts* opts, Cached* cache):
    # opts.calc_type == 1
    # [p_m, T, T_w, W_p,  W_c, psi1, psi2, ..., k, R]
    # [  0, 1,   2,   3,    4,    5, 6,    ..., n+5, n+6]
    cdef double[:] res = np.zeros(opts[0].n_powders + 7, dtype=np.float64)
    _fill_stuff(y, opts, cache, res)

    return res

cdef void _fill_stuff(double[:] y, Opts* opts, Cached* cache, double[:] stuff)  nogil:
    # opts.calc_type == 0
    # [p_m, T, T_w, W_p,  W_c, psi1, psi2, ..., k, R]
    # [  0, 1,   2,   3,    4,    5, 6,    ..., n+5, n+6]
    cdef:
        double T
        int n = opts[0].n_powders
        Py_ssize_t i
    for i in range(5,n+5):
        # psi_i
        stuff[i] = get_psi(y[i], opts, i-5)
    cdef double W_c = cache[0].omega_ign * opts[0].igniter.b_ign
    cdef double chisl = cache[0].E_0 - y[0] - y[1] - cache[0].E_kin_v2 * y[4]**2
    cdef double znam = cache[0].znam_ign
    cdef double rho_1 
    cdef double k, vi, k_chisl, k_znam, R, omega_gas_sum
    k_chisl = cache[0].omega_ign / opts[0].igniter.M_ign
    k_znam = cache[0].omega_ign / opts[0].igniter.M_ign / (opts[0].igniter.k_ign - 1)
    R = opts[0].igniter.R_ign * cache[0].omega_ign 
    omega_gas_sum = cache[0].omega_ign 
    for i in range(n):
        chisl += cache[0].chisl[i] * stuff[i+5]
        znam += cache[0].znam[i] * stuff[i+5]
        rho_1 = 1 / opts[0].powders[i].rho
        W_c += opts[0].powders[i].omega * (rho_1 - (rho_1 - opts[0].powders[i].b)*stuff[i+5])
    
        vi = stuff[i+5] * opts[0].powders[i].omega / opts[0].powders[i].M
        k_chisl += vi 
        k_znam += vi / (opts[0].powders[i].k - 1)
        R += opts[0].powders[i].omega * stuff[i+5] * opts[0].powders[i].R
        omega_gas_sum += opts[0].powders[i].omega * stuff[i+5]

    # k
    k = 1 + k_chisl / k_znam
    stuff[n+5] = k

    # R
    R /= omega_gas_sum
    stuff[n+6] = R

    # T
    T = chisl / znam
    stuff[1] = T
    
    # W_c
    stuff[4] = W_c

    # T_w 
    stuff[2] = opts[0].heat.T_w0 + sqrt(y[2])

    # W_p
    stuff[3] = opts[0].init_conditions.W_0 + y[3] * opts[0].init_conditions.S 
    
    znam = stuff[3] - stuff[4]
    cdef double p_m = cache[0].omega_ign * opts[0].igniter.f_ign / opts[0].igniter.T_ign * T / znam 
    for i in range(n):
        p_m += stuff[i+5] * opts[0].powders[i].omega * opts[0].powders[i].f / opts[0].powders[i].T_1 * T / znam

    stuff[0] = p_m



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

cdef int _stop_reason(double t, double[:] y, double[:] stuff, Opts* opts, int n_steps)  nogil:
    if opts[0].stop_conditions.t_max_flag:
        if t >= opts[0].stop_conditions.t_max:
            return 1
    if opts[0].stop_conditions.steps_max_flag:
        if n_steps >= opts[0].stop_conditions.steps_max:
            return 2
    if opts[0].stop_conditions.v_p_flag:
        if y[4] >= opts[0].stop_conditions.v_p:
            return 3
    if opts[0].stop_conditions.x_p_flag:
        if y[3] >= opts[0].stop_conditions.x_p:
            return 4
    if opts[0].stop_conditions.p_max_flag:
        if stuff[0] >= opts[0].stop_conditions.p_max:
            return 5
    return 0

cdef int _step(double t, double[:] y, double[:] stuff, Opts* opts, Cached* cache, double[:,:] y_tmps,
                double* t1, double[:] y1, double[:] stuff1, MetaTermoOpts* meta_termo)  nogil except -1:
    cdef:
        Py_ssize_t i
        double dt = meta_termo[0].dt
    if meta_termo[0].method == 0:
        _fill_dy(t, y, stuff, opts, cache, y_tmps[0])
        for i in range(y.shape[0]):
            y1[i] = y[i] + dt* y_tmps[0,i]
        _fill_stuff(y1, opts, cache, stuff1)
        t1[0] = t + dt
        return 0
    elif meta_termo[0].method == 1:
        _fill_dy(t, y, stuff, opts, cache, y_tmps[0])
        for i in range(y.shape[0]):
            y1[i] = y[i] + 0.5* dt * y_tmps[0,i]
        _fill_stuff(y1, opts, cache, stuff1)
        _fill_dy(t+0.5* dt, y1, stuff1, opts, cache, y_tmps[0])
        for i in range(y.shape[0]):
            y1[i] = y[i] + dt* y_tmps[0,i]
        _fill_stuff(y1, opts, cache, stuff1)
        t1[0] = t + dt
        return 0
    elif meta_termo[0].method == 2:
        _fill_dy(t, y, stuff, opts, cache, y_tmps[0])
        for i in range(y.shape[0]):
            y1[i] = y[i] + 0.5* dt * y_tmps[0,i]
        _fill_stuff(y1, opts, cache, stuff1)

        _fill_dy(t + 0.5* dt, y1, stuff1, opts, cache, y_tmps[1])    
        for i in range(y.shape[0]):
            y1[i] = y[i] + 0.5* dt * y_tmps[1,i]
        _fill_stuff(y1, opts, cache, stuff1)

        _fill_dy(t + 0.5* dt, y1, stuff1, opts, cache, y_tmps[2])    
        for i in range(y.shape[0]):
            y1[i] = y[i] + dt * y_tmps[2,i]
        _fill_stuff(y1, opts, cache, stuff1)

        _fill_dy(t + dt, y1, stuff1, opts, cache, y_tmps[3])    
        
        for i in range(y.shape[0]):
            y1[i] = y[i] + dt /6 * (y_tmps[0,i] + 2*y_tmps[1,i] + 2*y_tmps[2,i] + y_tmps[3,i])
        _fill_stuff(y1, opts, cache, stuff1)
        t1[0] = t + dt
        return 0
    return -2

cdef void _fill_dy(double t, double[:] y, double[:] stuff, Opts* opts, Cached* cache, double[:] dy)  nogil:
    # y
    # [Q_Ap, Q_w, eta_T, x, v, z1, z2, z3, ...]
    # [0,    1,   2,     3, 4,  5,  6,  ...]
    #
    # stuff
    # [p_m, T, T_w, W_p,  W_c, psi1, psi2, ...]
    # [  0, 1,   2,   3,    4,    5, 6,    ...]
    cdef:
        double p_a = opts[0].windage.p_0a
        double k_air = opts[0].windage.k_air
        double c_0a = opts[0].windage.c_0a
        double mach = y[4]/c_0a
        double mach_ = (k_air+1)*y[4]/4/c_0a
    if opts[0].windage.shock_wave:  
        p_a *= 1 + k_air*(k_air+1)/4*(mach*mach) + k_air*mach*sqrt(1+mach_*mach_)
    
    # dA/dt
    dy[0] =p_a * opts[0].init_conditions.S * y[4]

    
    cdef:
        double mu, Re, Nu, S_w, v_eta, vi
    
    if opts[0].heat.enabled:
                             # * opts[0].heat.Pr
        mu = opts[0].heat.mu_0 * ((stuff[1]/opts[0].heat.T_0)**1.5) * \
            (opts[0].heat.T_c + opts[0].heat.T_0)/(opts[0].heat.T_c + stuff[1])
        Re = cache[0].om_sum * y[4] * opts[0].init_conditions.d / (stuff[3]*2*mu)
        Nu = 0.023*(Re**0.8)*(opts[0].heat.Pr**0.4)

        # dQ_w/dt 
        S_w = opts[0].heat.F_0 + y[3] * opts[0].init_conditions.d * pi
        if opts[0].heat.Sigma_T < 0:
            dy[1] = S_w * Nu * opts[0].heat.lambda_p / opts[0].init_conditions.d * (stuff[1] - stuff[2])
        else:
            if fabs(opts[0].heat.vi) > 1e-8:
                vi = opts[0].heat.vi
            else:
                vi = 1 - stuff[2] / stuff[1]
            dy[1] = S_w * stuff[0] * opts[0].heat.Sigma_T * vi / stuff[opts[0].n_powders+6]
        # d eta_T/dt
        if opts[0].heat.heat_barrel:
            v_eta = - 2 * y[4] * y[2] / (y[3] + cache[0].l_0)
            if v_eta > 0:
                v_eta = 0.0
            dy[2] = Nu**2 *cache[0].znam_eta * (stuff[1] - stuff[2]) * (stuff[1] - stuff[2]) + v_eta
        else:
            dy[2] = 0
    else:
        dy[1] = 0.0
        dy[2] = 0.0

    #dx/dt
    dy[3] = y[4]

    # dv_p/dt
    if stuff[0] - p_a - opts[0].init_conditions.p_0 > 0 or fabs(y[4])>1e-8:
        dy[4] = (stuff[0]-p_a) * opts[0].init_conditions.S / opts[0].init_conditions.q / cache[0].phi

    else:
        dy[4] = 0.0

    cdef Py_ssize_t i
    for i in range(opts[0].n_powders):
        # dz_i/dt
        dy[i+5] = ((stuff[0])**opts[0].powders[i].nu)/opts[0].powders[i].I_k * H(opts[0].powders[i].z_e - y[i+5])

@cython.wraparound(True)
cpdef dict _construct_results(list ts, list ys, list stuffs, int reason):
    t = np.asarray(ts)
    p_m = np.empty_like(t)
    T = np.empty_like(t)
    x_p = np.empty_like(t)
    v_p = np.empty_like(t)
    Q_pa = np.empty_like(t)
    Q_w = np.empty_like(t)
    W_p = np.empty_like(t)
    W_c = np.empty_like(t)
    # eta_T = np.empty_like(t)
    T_w = np.empty_like(t)
    k = np.empty_like(t)
    cdef:
        Py_ssize_t i
    
    for i in range(t.shape[0]):
        p_m[i] = stuffs[i][0]
        T[i] = stuffs[i][1]
        x_p[i] = ys[i][3]
        v_p[i] = ys[i][4]
        Q_pa[i] = ys[i][0]
        Q_w[i] = ys[i][1]
        W_p[i] = stuffs[i][3]
        W_c[i] = stuffs[i][4]
        # eta_T[i] = ys[i][2]
        T_w[i] = stuffs[i][2]
        k[i] = stuffs[i][-2]
    cdef dict res = {
        't': t,
        'p_m' : p_m,
        'T' : T,
        'x_p' : x_p,
        'v_p' : v_p,
        'Q_pa' : Q_pa,
        'Q_w' : Q_w,
        'W_p' : W_p,
        'W_c' : W_c,
        # 'eta_T' : eta_T,
        'T_w' : T_w, 
        'k': k       
    }
    for i in range(ys[0].shape[0]-5):
        res[f'z_{i+1}'] = np.array([y[i+5] for y in ys]) 
        res[f'psi_{i+1}'] = np.array([s[i+5] for s in stuffs]) 
    
    if reason == 1:
        res['stop_reason'] = 't_max'
    elif reason == 2:
        res['stop_reason'] = 'steps_max'
    elif reason == 3:
        res['stop_reason'] = 'v_p'
    elif reason == 4:
        res['stop_reason'] = 'x_p'
    elif reason == 5:
        res['stop_reason'] = 'p_max'

    return res

@cython.wraparound(True)
cdef void trim_interpolate_results(list ts, list ys, list stuffs, Opts* opts, int n_steps, int reason):
    if reason == 2:
        return
    if len(ts) < 2:
        return

    cdef:
        double x1, x, x2, t
        Py_ssize_t i
    if reason == 1:
        x = opts[0].stop_conditions.t_max
        x1 = ts[-2]
        x2 = ts[-1]

    elif reason == 3:
        x = opts[0].stop_conditions.v_p
        x1 = ys[-2][4]
        x2 = ys[-1][4]

    elif reason == 4:
        x = opts[0].stop_conditions.x_p
        x1 = ys[-2][3]
        x2 = ys[-1][3]

    elif reason == 5:
        x = opts[0].stop_conditions.p_max
        x1 = stuffs[-2][0]
        x2 = stuffs[-1][0]
    else:
        raise ValueError(f'Неясная причина остановки reason = {reason}')
    
    t = (x - x1) / (x2 - x1)
    y1 = ys[-2]
    y2 = ys[-1]
    for i in range(y1.shape[0]):
        y2[i] = (1-t) * y1[i] + t * y2[i]
    y1 = stuffs[-2]
    y2 = stuffs[-1]
    for i in range(y1.shape[0]):
        y2[i] = (1-t) * y1[i] + t * y2[i]