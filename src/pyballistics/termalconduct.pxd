
cpdef int binary_search(double[:] A, double T) nogil
cpdef void fill_averages(double[:] values, double[:] val_xs, double[:] aver_xs, double[:] averages) nogil

cdef class Termo1d:
    cdef:
        public double[::1] rs, Ts, alphas, bettas
        public double delta_b, c_b, lambda_b, time

    cpdef void step(self, double tau, double q_0, double T_up) nogil
    cpdef dict get_state(self)
    cpdef Termo1d copy(self)
    cpdef void set_state(self, dict state)


cdef class TermalConductBarrel:
    cdef:
        public int n_cells
        public list cells
        public double T_0
        public double[:] xs, cells_centers, T_ws, q_ws

    cpdef dict get_state(self)
    cpdef void fill_T_ws(self, double[:] xs, double[:] T_ws)
    cpdef void fill_self_q_ws(self, double[:] xs, double[:] q_ws)
    cpdef void step(self, double tau)