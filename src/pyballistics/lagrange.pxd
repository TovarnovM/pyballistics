from .termo cimport Opts
cimport cython
from cymem.cymem cimport Pool

cdef double get_psi(double z, Opts* opts, Py_ssize_t i)  nogil

cdef double H(double x)  nogil

cdef struct MetaLagrangeOpts:
    int n_cells
    double CFL, W

cdef class LagrangeLayer:
    cdef:
        Opts opts
        public dict opts_full
        MetaLagrangeOpts meta
        Pool mem
        public double[:] ps, xs, us, Ts, es, rhos, cs, Ws, T_ws, m_sums, ks, W_cs, qs, Rs, dEta, xs_tmp, T_ws_tmp
        public double[:, :] zs, psis, omegas
        public double t, tau_last, znam_eta
        public int step_count

    cdef MetaLagrangeOpts _get_meta(self, dict opts_full)
    cpdef dict get_y0(self)
    cpdef dict get_state(self)
    cpdef void set_state(self, dict state)
    cpdef dict get_result(self)
    cpdef void synch_rhos(self) nogil
    cpdef void synch_psis(self) nogil
    cpdef void synch_Ws(self) nogil
    cpdef void synch_ks(self) nogil
    cpdef void synch_es_Ts_Wcs_cs(self) nogil
    cpdef void synch_ps_Ts_Wcs_cs(self) nogil
    cpdef void step(self, double tau) nogil
    cpdef double get_p_a(self, double v_p) nogil
    cpdef double get_omega_ign(self)
    cpdef bint stop_reason(self) nogil
    cpdef double get_tau(self) nogil
    cpdef double tau_Ku_filter_W(self, double tau) nogil
    cpdef str get_stop_reason(self)
    cpdef void synch_qs(self) nogil