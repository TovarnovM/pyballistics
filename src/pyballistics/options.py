from copy import deepcopy
from schema import Schema, And, Use, Optional, Or
import os
from math import pi
import numpy as np



_schema_powder_dict = Schema({
    'omega': lambda x: x > 0,       # кг, масса навески пороха
    'I_e':   lambda x: x > 0,     # Па*с, импульс конца горения 
    Optional('nu', default=1): lambda x: x > 0,          # показатель в степенном законе горения (опционально, по-умолчанию 1)
    Optional('b', default=0): lambda x: x >= 0,      # м^3/кг, коволюм пороховых газов
    'delta': lambda x: x > 0,       # кг/м^3, плотность пороха   
    'f': lambda x: x > 0,     # Дж/кг, сила пороха 
    'k': lambda x: x > 0,        # коэффициент адиабаты пороховых газов
    'T_p': lambda x: x > 0,     # К, темп. горения пороха
    'z_e': lambda x: x > 0,     # относительная толщина сгоревшего слоя конца горения
    'kappa_1': Or(float, int), # коэффициенты в геометрическом законе горения
    'lambda_1': Or(float, int),
    Optional('mu_1', default=0): Or(float, int), 
    'kappa_2': Or(float, int),
    'lambda_2': Or(float, int),
    Optional('mu_2', default=0): Or(float, int), 
    Optional('k_I', default=0): lambda x: x >= 0,    # 1/K, коэффициент для пересчета импульса конца горения для других начальных температур
    Optional('k_f', default=0): lambda x: x >= 0   
}, ignore_extra_keys=True)


_schema_init_cond = Schema({
    'q': lambda x: x > 0,       # кг, масса снаряда
    'd': lambda x: x > 0,    # м, калибр
    'W_0': lambda x: x > 0,      # м^3, начальный объем каморы
    Optional('T_0', default=293.15): lambda x: x > 0, # К, начлаьная температура
    Optional('n_S', default=1.0): lambda x: x > 0,     #
    'phi_1': lambda x: x > 0,  # коэффициент, учитывающий силу трения в нарезах (участвует в вормуле расчета коэффициента фиктивности массы снаряда)
    'p_0': lambda x: x > 0,   # Па, давление форсирования
    Optional('S'): lambda x: x > 0
}, ignore_extra_keys=True)

_schema_igniter = Schema({
    'p_ign_0': lambda x: x > 0, # Па, Давление вспышки
    Optional('k_ign', default=1.22): lambda x: x > 0,  # коэффициент адиабаты газов воспламенителя
    Optional('T_ign', default=2427): lambda x: x > 0,  # температура горения воспралменителя
    Optional('f_ign', default=0.26e6): lambda x: x > 0, # Дж/кг, сила воспламенителя
    Optional('b_ign', default=0.0006): lambda x: x >= 0 # м^3/кг, коволюм газов воспламенителя
}, ignore_extra_keys=True)

_schema_windage = Schema({
    Optional('shock_wave', default=True): lambda x: isinstance(x, bool), # флаг, нужно ли считать сопротивление перед снарядом как ударную волну. если False, то сопротивление постоянно == p_0a 
    Optional('p_0a', default=1e5): lambda x: x >= 0,       # Па, давление воздуха перед снарядом
    Optional('k_air', default=1.4): lambda x: x > 0,       # показатель адиабаты воздуха
    Optional('c_0a', default=340): lambda x: x > 0         # м/с, скорость звука в воздухе
}, ignore_extra_keys=True)

_schema_heat = Schema({
    Optional('enabled', default=True): lambda x: isinstance(x, bool), 
    Optional('heat_barrel', default=True): lambda x: isinstance(x, bool), 
    Optional('F_0'): lambda x: x > 0, # м^2, наальная площадь теплоотдачи, если не указан, то будет рассчитываться по формуле 4*W_0 / d 
    Optional('Pr', default=0.74): lambda x: x >= 0, # число Прандля
    Optional('T_w0'): lambda x: x > 0, # температура стенки, если не указывть - то будет взята начальная температура
    Optional('mu_0', default=0.175e-4): lambda x: x > 0, # Па*с, Коэффициент динамической вязкости пороховых газов для формулы Сазерленда
    Optional('T_cs', default=628): lambda x: x > 0, # К, тоже для формулы Сазерленда
    Optional('T_0s', default=273): lambda x: x > 0, # K, тоже для формулы Сазерленда
    Optional('c_b', default=500): lambda x: x > 0, # Дж/(кг * град) теплоемкость материала ствола
    Optional('rho_b', default=7900):  lambda x: x > 0, # кг/м^3 плотность маетриала ствола
    Optional('lambda_b', default=40):  lambda x: x > 0, # Вт/(м·град), теплопроводность материала ствола
    Optional('lambda_g', default= 0.2218): lambda x: x > 0, # Вт/(м * К), теплопроводность пороховых газов
    Optional('Sigma_T') : lambda x: x >=0, # На случай использования старого неправильного закона теплопередачи
    Optional('vi'): lambda x: True # аналогично
}, ignore_extra_keys=True)


_schema_meta_termo = Schema({
    'dt': lambda x: x > 0,  #с,  шаг по времени
    Optional('method', default='rk2'): Or('euler', 'rk2', 'rk4') # метод интегрирования, Эйлер = 'euler', Рунге-Кутты 2 и 4 порядков = 'rk2', 'rk4'
}, ignore_extra_keys=True)


_schema_meta_lagrange = Schema({
    'n_cells': And(lambda x: x > 0, Use(lambda x: int(x))),  # количество ячеек
    'CFL': lambda x: 1 >= x > 0, # число Куранта 
    Optional('W'): lambda x: x > 1, # Дополнительное требование для повышения устойчивости: последующий шаг по времени не может быть больше текущего в W раз
}, ignore_extra_keys=True)


_schema_stop_conditions = Schema({
    Optional('t_max'): lambda x: x > 0, # с, прервать расчет при t > t_max
    Optional('steps_max'): And(int, lambda x: x > 0), # сделать максимум 'steps_max' шагов интегрирвоания
    Optional('v_p'): lambda x: x > 0,   # м/c, прервать расчет, когда скорость снаряда достигнет V_p
    Optional('x_p'): lambda x: x > 0,     # м, прервать расчет, когда снаряд пройдет x_p метров (в начальный момент снаряд прошел 0 м)
    Optional('p_max'): lambda x: x > 0, # Па, прервать расчет, если давление превысит p_max}
}, ignore_extra_keys=True) 

_agard_options = {
    'powders': [{
        'I_e': 250495 ,
        'T_p': 2585,
        'b': 0.0010838,
        'f': 1.009e6,
        'k': 1.27,
        'kappa_1': 0.7185,
        'kappa_2': 0.5386,
        'lambda_1': 0.2049,
        'lambda_2': -0.8977,
        'mu_1': -0.0217,
        'nu': 0.9,
        'omega': 9.5255,
        'delta': 1575,
        'z_e': 1.56}],
    'init_conditions': {
        'q': 45.359,
        'd': 0.132,
        'W_0': 9.5255 / 576,
        'T_0': 293.15,
        'phi_1': 1.0,
        'p_0': 13.79e6},
    'igniter': {
        'p_ign_0': 1000000.0,
        'k_ign': 1.25,
        'T_ign': 1706,
        'f_ign': 260000.0,
        'b_ign': 0.0006},
    'meta_termo': {
        'dt': 5e-06, 
        'method': 'rk2' },
    'meta_lagrange': {
        'CFL': 0.9, 
        'n_cells': 150 },
    'stop_conditions': {
        'x_p': 4.318 }
}

_sample_termo_options = {
    'powders': [
        {
            'omega': 7,             # кг, масса навески пороха
            'dbname': 'ДГ-4 15/1',   # имя пороха в БД, узнать все доступные имена можно из функции get_all_powder_names()
        },
        {
            'omega': 6,             # кг, масса навески пороха
            'dbname': '22/7',   # имя пороха в БД, узнать все доступные имена можно из функции get_all_powder_names()
        } ],
    'init_conditions': {     # блок начальных данных
        'q': 51.76,       # кг, масса снаряда
        'd': 0.122,    # м, калибр
        'W_0': 13/400,      # м^3, начальный объем каморы
        'phi_1': 1.02,  # коэффициент, учитывающий силу трения в нарезах (участвует в вормуле расчета коэффициента фиктивности массы снаряда)
        'p_0': 30e6,   # Па, давление форсирования
    },
    'igniter': {
        'p_ign_0': 1e6},
    'meta_termo': {
        'dt': 5e-06, 
        'method': 'rk2' },
    'meta_lagrange': {
        'CFL': 0.9, 
        'n_cells': 150 },
    'stop_conditions': {
        'v_p': 690,
        'p_max': 600e6,
        'x_p': 9
    }
}

_sample_termo_options_2 = {
    'powders': [
        {'omega': 5.7, 'dbname': '22/1 тр'}
    ],
    'init_conditions': {
        'q': 27.3,
        'd': 0.122,
        'W_0': 0.01092,
        'phi_1': 1.02,
        'p_0': 30e6,
        'n_S': 1.04
    },
    'igniter': {
        'p_ign_0': 1e6},
    'heat':{
        'enabled': False,
        'heat_barrel': False
    },
    'windage':{
        'p_0a': 1e-9
    },
    'meta_termo': {
        'dt': 1e-6, 
        'method': 'rk4'},
    'meta_lagrange': {
        'n_cells': 300, 
        'CFL': 0.9},
    'stop_conditions': {
        'x_p': 4.88, 
        'steps_max': 100000
        }
    }

def get_full_options(opts):
    """Функция для формирования полного словаря с начальными данными. С проверкой на правильность значений.

    :param opts: Словарь с начальными данными (может быть неплоным)
    :type opts: dict
    :raises ValueError: Если словарь по структуре не подходит, или в нем неправильные данные
    :return: Словарь, в котором есть ВСЕ исходные данные для задачи ОЗВБ
    :rtype: dict

    Пример:
    >>> opts = get_options_sample()
    >>> print(opts)
    'powders': [
        {
            'omega': 7,             
            'dbname': 'ДГ-4 15/1',   
        },
        {
            'omega': 6,            
            'dbname': '22/7',   
        } ],
    'init_conditions': {     
        'q': 51.76,       
        'd': 0.122,    
        'W_0': 13/400,      
        'phi_1': 1.02, 
        'p_0': 30e6,  
    },
    'igniter': {
        'p_ign_0': 1e6},
    'meta_termo': {
        'dt': 5e-06, 
        'method': 'rk2' },
    'meta_lagrange': {
        'CFL': 0.9, 
        'n_cells': 150 },
    'stop_conditions': {
        'v_p': 690,
        'p_max': 600e6,
        'x_p': 9
    }
}

    >>> opts_full = get_full_options(opts)
    >>> print(opts_full)
        {'powders': [
            {'I_e': 1120000.0,
            'f': 1004000.0,
            'k': 1.243,
            'T_p': 2650.0,
            'delta': 1520.0,
            'b': 0.001085,
            'z_e': 1.0,
            'kappa_1': 1.0,
            'lambda_1': 0.0,
            'kappa_2': 0.0,
            'lambda_2': 0.0,
            'k_I': 0.0022,
            'k_f': 0.00036,
            'omega': 7,
            'mu_2': 0,
            'nu': 1,
            'mu_1': 0,
            'R': 378.8679245283019,
            'M': 0.021945543763053868
            },
            {'I_e': 1530000.0,
            'f': 983000.0,
            'k': 1.232,
            'T_p': 2755.0,
            'delta': 1600.0,
            'b': 0.001029,
            'z_e': 1.501,
            'kappa_1': 0.749,
            'lambda_1': 0.155,
            'kappa_2': 0.53854,
            'lambda_2': -0.997,
            'k_I': 0.0016,
            'k_f': 0.0003,
            'omega': 6,
            'mu_2': 0,
            'nu': 1,
            'mu_1': 0,
            'R': 356.8058076225045,
            'M': 0.023302486788415237}],
        'init_conditions': {
            'q': 51.76,
            'd': 0.122,
            'W_0': 0.0325,
            'phi_1': 1.02,
            'p_0': 30000000.0,
            'T_0': 293.15,
            'n_S': 1.0},
        'windage': {
            'shock_wave': True, 
            'p_0a': 100000.0, 
            'k_air': 1.4, 
            'c_0a': 340},
        'heat': {
            'T_cs': 628,
            'heat_barrel': True,
            'T_0s': 273,
            'lambda_b': 40,
            'c_b': 500,
            'enabled': True,
            'lambda_g': 0.2218,
            'rho_b': 7900,
            'mu_0': 1.75e-05,
            'Pr': 0.74,
            'F_0': 1.0655737704918034,
            'T_w0': 293.15},
        'igniter': {
            'p_ign_0': 1000000.0,
            'f_ign': 260000.0,
            'T_ign': 2427,
            'b_ign': 0.0006,
            'k_ign': 1.22,
            'R_ign': 107.12814173877214,
            'M_ign': 0.07761231067022274},
        'stop_conditions': {'v_p': 690, 'p_max': 600000000.0, 'x_p': 9},
        'meta_termo': {'dt': 5e-06, 'method': 'rk2'},
        'meta_lagrange': {'CFL': 0.9, 'n_cells': 150}}
    """
    if not isinstance(opts, dict):
        raise ValueError('Неправильные данные. Параметр opts должен быть словарём. Пример правильного словаря opts можно получить из функции get_termo_options_sample()')
    res = {}
    _verify_powders(opts, res)
    _verify_opts_dict_stuff(opts, res, 'init_conditions', 'в котором указываются начальные параметры', _schema_init_cond)
    if 'windage' not in opts:
        opts['windage'] = {}
    _verify_opts_dict_stuff(opts, res, 'windage', 'в котором указываются параметры сопротивления столба воздуха', _schema_windage)
    if 'heat' not in opts:
        opts['heat'] = {}
    _verify_opts_dict_stuff(opts, res, 'heat', 'в котором указываются параметры теплоотдачи', _schema_heat)
    _verify_opts_dict_stuff(opts, res, 'igniter', 'в котором указываются параметры воспламенителя', _schema_igniter)
    _verify_opts_dict_stuff(opts, res, 'stop_conditions', 'в котором указываются условия остановки интегрирования', _schema_stop_conditions,
        condition_foo=lambda d: len(d) > 0, condition_foo_error_msg='Должно быть хотя бы одно условие остановки')
    
    meta_flag = False
    if 'meta_termo' in opts:
        _verify_opts_dict_stuff(opts, res, 'meta_termo', 'в котором указываются параметры интегрирования', _schema_meta_termo)
        meta_flag = True
    if 'meta_lagrange' in opts:
        _verify_opts_dict_stuff(opts, res, 'meta_lagrange', 'в котором указываются параметры интегрирования', _schema_meta_lagrange)
        meta_flag = True
    if not meta_flag:
        raise ValueError(f'В словаре с начальными данными обязательно должен быть раздел, в котором указываются параметры интегрирования. Например "meta_termo" или "meta_lagrange"')
    _fill_optionalz(res)
    _check_full_opts(res)
    return res

def _verify_powders(opts, res):
    if 'powders' not in opts:
        raise ValueError('В словаре opts обязательно должно быть поле "powders", в котором указываются параметры заряда. Пример правильного словаря opts можно получить из функции get_termo_options_sample()')
    if not isinstance(opts['powders'], list):
        raise ValueError('В словаре opts поле "powders" должно быть списком, в котором указываются параметры заряда. Пример правильного словаря opts можно получить из функции get_termo_options_sample()')
    if len(opts['powders']) == 0:
        raise ValueError('В словаре opts поле "powders" должно быть списком НЕНУЛЕВОЙ длины, в котором указываются параметры заряда. Пример правильного словаря opts можно получить из функции get_termo_options_sample()')
    res['powders'] = [_get_powder_dict(powder_dict) for powder_dict in opts['powders']]
             
def _get_powder_dict(powder_dict):
    if not isinstance(powder_dict, dict):
        raise ValueError('В словаре opts элементами списка opts["powders"] должны быть словарями, в которых указываются параметры заряда. Пример правильного словаря opts можно получить из функции get_termo_options_sample()')
    res = dict(powder_dict) 
    if 'dbname' in res:
        dbpowder = get_db_powder(res['dbname'])
        res = dict(dbpowder, **res)
    return _schema_powder_dict.validate(res)


def _verify_opts_dict_stuff(opts, res, field_name, message, schema, condition_foo=None, condition_foo_error_msg=''):
    if field_name not in opts:
        raise ValueError(f'В словаре opts обязательно должно быть поле "{field_name}", {message}. Пример правильного словаря opts можно получить из функции get_termo_options_sample()')
    if not isinstance(opts[field_name], dict):
        raise ValueError(f'В словаре opts поле "{field_name}" должно быть словарем, {message}. Пример правильного словаря opts можно получить из функции get_termo_options_sample()')
    res[field_name] = schema.validate(opts[field_name])
    if condition_foo:
        if not condition_foo(res[field_name]):
            raise ValueError(condition_foo_error_msg)

def _fill_optionalz(res):
    if 'S' not in res['init_conditions'] or res['init_conditions']['S'] is None:
        res['init_conditions']['S'] = res['init_conditions']['n_S'] * pi * (res['init_conditions']['d'])**2 / 4

    if 'F_0' not in res['heat'] or res['heat']['F_0'] is None:
        res['heat']['F_0'] = 4 * res['init_conditions']['W_0'] / res['init_conditions']['d']

    if 'T_w0' not in res['heat'] or res['heat']['T_w0'] is None:
        res['heat']['T_w0'] = res['init_conditions']['T_0'] 

    if 'Sigma_T' not in res['heat']:
        res['heat']['Sigma_T'] = -1

    if 'vi' not in res['heat']:
        res['heat']['vi'] = 0 
    
    for pd in res['powders']:
        pd['R'] = pd['f'] / pd['T_p']
        pd['M'] = 8.31446261815324 / pd['R']

    ign = res['igniter']
    ign['R_ign'] = ign['f_ign'] / ign['T_ign']
    ign['M_ign'] = 8.31446261815324 / ign['R_ign']

def _check_full_opts(full_opts):
    om_delta_sum = np.sum([powder['omega'] / powder['delta'] for powder in full_opts['powders']])
    omega_ign = full_opts['igniter']['p_ign_0'] / full_opts['igniter']['f_ign'] * (full_opts['init_conditions']['W_0'] - om_delta_sum) / (1 + \
            full_opts['igniter']['b_ign'] * full_opts['igniter']['p_ign_0'] / full_opts['igniter']['f_ign'])
    Wp_Wc = full_opts['init_conditions']['W_0'] - om_delta_sum - omega_ign * full_opts['igniter']['b_ign']
    if Wp_Wc <= 0:
        raise ValueError(f'Начальный свободный объем получается отрицательный. Масса навески слишком большая для объема каморы')

_powder_db_path = os.path.join( os.path.dirname(__file__), 'gpowders_si.csv')
_powder_db = None

def _init_powder_db(db_path):
    global _powder_db 
    _powder_db = {}
    headers = ['name', 'I_e', 'f', 'k', 'T_p', 'delta', 'b', 'z_e', 'kappa_1', 'lambda_1', 'kappa_2', 'lambda_2', 'k_I', 'k_f']
    with open(db_path, encoding='utf-8')  as f:
        f.readline()
        for line in f.readlines():
            ss = line.split(';')
            pd = {headers[0]: ss[0]}
            for h, s in zip(headers[1:], ss[1:]):
                pd[h] = float(s)
            _powder_db[ss[0]] = pd


def get_db_powder(powder_name):
    """Возвращает словарь с параметрами табличного пороха с именем powder_name

    :param powder_name: Имя табличного пороха (список доступных порохов можно узнать, вызвав функцию get_powder_names())
    :type powder_name: str
    :raises ValueError: Если в таблице нет пороха с указанным именем
    :return: Словарь с параметрами пороха
    :rtype: dict

    Пример:
    >>> get_db_powder('4/7')
    >>> {
            'name': '4/7',      # имя пороха в БД
            'I_e': 320000.0,    # Па*с, импульс конца горения 
            'f': 1027000.0,     # Дж/кг, сила пороха 
            'k': 1.228,         # коэффициент адиабаты пороховых газов
            'T_p': 3006.0,      # К, темп. горения пороха
            'delta': 1600.0,    # кг/м^3, плотность пороха 
            'b': 0.001008,      # м^3/кг, коволюм пороховых газов
            'z_e': 1.488,       # относительная толщина сгоревшего слоя конца горения
            'kappa_1': 0.811,   # коэффициент в геометрическом законе горения
            'lambda_1': 0.081,  # коэффициент в геометрическом законе горения
            'kappa_2': 0.50536, # коэффициент в геометрическом законе горения
            'lambda_2': -1.024, # коэффициент в геометрическом законе горения
            'k_I': 0.0016,      # 1/K, коэффициент для пересчета импульса конца горения для других начальных температур
            'k_f': 0.0003       # 1/K, коэффициент для пересчета силы пороха для других начальных температур
        }
    """
    if _powder_db is None:
        _init_powder_db(_powder_db_path)
    if powder_name not in _powder_db:
        raise ValueError(f'Такого пороха в таблице нет: {powder_name}. Список доступных имен можно получить из функции get_powder_names()')
    return deepcopy(_powder_db[powder_name])

def get_powder_names():
    """Возвращает список с доступными табличными порохами

    :return: Список с именами порохов. Для получения параметров пороха нужно воспользоваться функцией get_db_powder(powder_name)
    :rtype: list[str]
    """
    if _powder_db is None:
        _init_powder_db(_powder_db_path)
    return list(_powder_db.keys())


def get_options_sample():
    """Возвращает пример словаря с правильными начальными данными, который может быть
    использован для расчета в функциях ozvb_termo и ozvb_lagrange

    :return: 
    
    {
        'powders': [
            {
                'omega': 7,              # кг, масса навески пороха
                'dbname': 'ДГ-4 15/1',   # имя пороха в БД, узнать все доступные имена можно из функции get_all_powder_names()
            },
            {
                'omega': 6,         # кг, масса навески пороха
                'dbname': '22/7',   # имя пороха в БД, узнать все доступные имена можно из функции get_all_powder_names()
            } ],
        'init_conditions': {    # блок начальных данных
            'q': 51.76,         # кг, масса снаряда
            'd': 0.122,         # м, калибр
            'W_0': 13/400,      # м^3, начальный объем каморы
            'phi_1': 1.02,      # коэффициент, учитывающий силу трения в нарезах (участвует в вормуле расчета коэффициента фиктивности массы снаряда)
            'p_0': 30e6,        # Па, давление форсирования
        },
        'igniter': {
            'p_ign_0': 1e6},    # Па, Давление вспышки
        'meta_termo': {
            'dt': 5e-06,        #с, шаг по времени
            'method': 'rk2' },  # метод интегрирования, Эйлер = 'euler', Рунге-Кутты 2 и 4 порядков = 'rk2', 'rk4'
        'meta_lagrange': {
            'CFL': 0.9,         # число Куранта 
            'n_cells': 150 },   # количество ячеек
        'stop_conditions': {
            'v_p': 690,         # м/c, прервать расчет, когда скорость снаряда достигнет v_p
            'p_max': 600e6,     # Па, прервать расчет, если давление превысит p_max
            'x_p': 9            # м, прервать расчет, когда снаряд пройдет x_p метров (в начальный момент снаряд прошел 0 м)
        }
    }

    :rtype: dict
    """
    return deepcopy(_sample_termo_options)

def get_options_sample_2():
    """Возвращает пример словаря с правильными начальными данными, который может быть
    использован для расчета в функциях ozvb_termo и ozvb_lagrange

{
    'powders': [
        {'omega': 5.7, 'dbname': '22/1 тр'}
    ],
    'init_conditions': {
        'q': 27.3,
        'd': 0.122,
        'W_0': 0.01092,
        'phi_1': 1.02,
        'p_0': 30e6,
        'n_S': 1.04
    },
    'igniter': {
        'p_ign_0': 1e6},
    'heat':{
        'enabled': False,
        'heat_barrel': False
    },
    'windage':{
        'p_0a': 1e-9
    },
    'meta_termo': {
        'dt': 1e-6, 
        'method': 'rk4'},
    'meta_lagrange': {
        'n_cells': 300, 
        'CFL': 0.9},
    'stop_conditions': {
        'x_p': 4.88, 
        'steps_max': 100000
        }
    }

    :rtype: dict
    """
    return deepcopy(_sample_termo_options_2)

def get_options_agard():
    """Возвращает словарь с начальными данными для задачи AGARD
    Словарь может быть использован для расчета в функциях ozvb_termo и ozvb_lagrange

    :return:

    {
        'powders': [{
            'I_e': 250495 ,
            'T_p': 2585,
            'b': 0.0010838,
            'f': 1.009e6,
            'k': 1.27,
            'kappa_1': 0.7185,
            'kappa_2': 0.5386,
            'lambda_1': 0.2049,
            'lambda_2': -0.8977,
            'mu_1': -0.0217,
            'nu': 0.9,
            'omega': 9.5255,
            'delta': 1575,
            'z_e': 1.56}],
        'init_conditions': {
            'q': 45.359,
            'd': 0.132,
            'W_0': 9.5255 / 576,
            'T_0': 293.15,
            'phi_1': 1.0,
            'p_0': 13.79e6},
        'igniter': {
            'p_ign_0': 1000000.0,
            'k_ign': 1.25,
            'T_ign': 1706,
            'f_ign': 260000.0,
            'b_ign': 0.0006},
        'meta_termo': {
            'dt': 5e-06, 
            'method': 'rk2' },
        'meta_lagrange': {
            'CFL': 0.9, 
            'n_cells': 150 },
        'stop_conditions': {
            'x_p': 4.318 }
    }
    :rtype: dict
    """
    return deepcopy(_agard_options)
