cimport cython
from cymem.cymem cimport Pool

cdef struct PowderOpts:
    double omega, I_k, nu, b, rho, f, k, T_1, z_e, kappa_1, lambda_1, mu_1, kappa_2, lambda_2, mu_2, k_I, k_f, R, M

cdef struct InitCondOpts:
    double q, d, W_0, T_0, S, phi_1, p_0

cdef struct IgniterOpts:
    double p_ign_0, k_ign, T_ign, f_ign, b_ign, R_ign, M_ign

cdef struct WindageOpts:
    double p_0a, k_air, c_0a
    int shock_wave

cdef struct HeatOpts:
    int enabled, heat_barrel
    double F_0, Pr, T_w0, mu_0, T_c, T_0, c_c, rho_c, lambda_c, lambda_p, Sigma_T, vi

cdef enum IntegrMethod: 
    euler=0
    rk2=1
    rk4=2

cdef struct MetaTermoOpts:
    double dt
    IntegrMethod method

cdef struct StopConditions:
    int t_max_flag, steps_max, steps_max_flag, v_p_flag, x_p_flag, p_max_flag
    double t_max, v_p, x_p, p_max

cdef struct Opts:
    int n_powders
    PowderOpts* powders
    InitCondOpts init_conditions
    IgniterOpts igniter
    WindageOpts windage
    HeatOpts heat
    # MetaTermoOpts meta_termo
    StopConditions stop_conditions

cdef struct Cached:
    double E_00, phi, E_0, znam_ign, omega_ign, om_sum, znam_eta, l_0, E_kin_v2
    double *chisl
    double *znam 

cdef Opts _convert_2_opts(dict opts_full, Pool mem)
cdef (Cached, MetaTermoOpts) _get_cache_meta(dict opts_full, Opts opts, Pool mem)
cdef double[:] _get_y0(Opts* opts)
cdef double[:] _get_stuff(double[:] y, Opts* opts, Cached* cache)
cdef void _fill_stuff(double[:] y, Opts* opts, Cached* cache, double[:] stuff)  nogil
cdef double get_psi(double z, Opts* opts, Py_ssize_t i)  nogil
cdef double H(double x)  nogil
cdef int _stop_reason(double t, double[:] y, double[:] stuff, Opts* opts, int n_steps)  nogil
cdef int _step(double t, double[:] y, double[:] stuff, Opts* opts, Cached* cache, double[:,:] y_tmps, double* t1, double[:] y1, double[:] stuff1, MetaTermoOpts* meta_termo)  nogil except -1
cdef void _fill_dy(double t, double[:] y, double[:] stuff, Opts* opts, Cached* cache, double[:] dy)  nogil
cpdef dict _construct_results(list ts, list ys, list stuffs, int reason)
cdef void trim_interpolate_results(list ts, list ys, list stuffs, Opts* opts, int n_steps, int reason)