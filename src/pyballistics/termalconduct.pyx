import numpy as np
cimport cython
from libc.math cimport floor


@cython.initializedcheck(False)
cpdef int binary_search(double[:] A, double T) nogil:
    cdef:
        int L = 0
        int R = A.shape[0] - 1
        int m
    if T <= A[L]:
        return 0
    if T >= A[R]:
        return R
    while R  >= L:
        m = int(floor((L + R)/2))
        if A[m] < T:
            L = m + 1
        elif A[m] > T:
            R = m - 1
        else:
            return m
    return -1

@cython.initializedcheck(False)
cpdef void fill_averages(double[:] values, double[:] val_xs, double[:] aver_xs, double[:] averages) nogil:
    if values.shape[0] + 1 != val_xs.shape[0]:
        raise ValueError('Массивы values и val_xs имеют неправильные размерности!')
    if averages.shape[0] + 1 != aver_xs.shape[0]:
        raise ValueError('Массивы averages и aver_xs имеют неправильные размерности!')
    cdef:
        int j = 0
        int i = binary_search(val_xs, aver_xs[0])
        double s=0, val, x1, x2
    x1 = aver_xs[0]
    val = values[i]
    i += 1
    while j < averages.shape[0]:
        if i >= values.shape[0] or aver_xs[j+1] < val_xs[i]:
            x2 = aver_xs[j+1]
            s += (x2 - x1) * val
            averages[j] = s / (aver_xs[j+1] - aver_xs[j])
            s = 0
            j += 1
        else:
            x2 = val_xs[i]
            s += (x2 - x1) * val
            val = values[i]
            i += 1
        x1 = x2




@cython.final
cdef class Termo1d:
    @cython.wraparound(True)
    @classmethod
    def get_standart(cls, **kw):
        r_0 = kw.get('r_0', 0.023/2)
        r_1 = kw.get('r_1', 0.03/2)
        n = kw.get('n', 33)
        T_0 = kw.get('T_0', 293)
        s = r_1 - r_0
        q = kw.get('q', 0.85)
        b = s*(1-q)/(1-q**(n-1))
        rs = np.zeros(n)
        rs[-1] = r_1
        rs[-2] = r_1 - b
        for i in range(rs.shape[0]-3,0,-1):
            b *= q
            rs[i] = rs[i+1] - b
        rs[0] = r_0
        Ts = np.zeros(n)
        Ts[:] = T_0
        return cls(rs, Ts,  
            delta_b=kw.get('delta_b', 7800), 
            c_b=kw.get('c_b', 480),
            lambda_b=kw.get('lambda_b', 27),
            time=kw.get('time', 0.0))

    def __cinit__(self, rs, Ts, delta_b, c_b, lambda_b, time):
        self.rs = np.array(rs)
        self.Ts = np.array(Ts)
        self.delta_b = delta_b
        self.c_b = c_b
        self.lambda_b = lambda_b
        self.time = time
        self.alphas = np.zeros_like(self.rs)
        self.bettas = np.zeros_like(self.Ts)

    cpdef dict get_state(self):
        return {
            'rs': np.array(self.rs),
            'Ts': np.array(self.Ts),
            'time': self.time
        }

    cpdef void set_state(self, dict state):
        self.rs = np.array(state['rs'])
        self.Ts = np.array(state['Ts'])
        self.time = np.array(state['time'])
        self.alphas = np.zeros_like(self.rs)
        self.bettas = np.zeros_like(self.Ts)

    cpdef Termo1d copy(self):
        return Termo1d(self.rs, self.Ts, self.delta_b, self.c_b, self.lambda_b, self.time)

    @cython.initializedcheck(False)
    cpdef void step(self, double tau, double q_0, double T_up) nogil:
        cdef:
            double a = self.lambda_b/(self.c_b*self.delta_b)
            double B = self.rs[0] - self.rs[1]
            double C = -B
            double F = -q_0/self.lambda_b
            double T_k_i, r_k, r_km1, r_kp1, m_1, m_2, A
            int i
        self.alphas[0] = -C/B
        self.bettas[0] = F/B
        for i in range(1 ,self.rs.shape[0]-1):
            T_k_i = self.Ts[i]
            r_k = self.rs[i]
            r_km1=self.rs[i-1]
            r_kp1=self.rs[i+1]
            m_1 = a*(r_kp1+r_k)/(r_kp1-r_km1)/(r_kp1-r_k)
            m_2 = a*(r_k+r_km1)/(r_kp1-r_km1)/(r_k-r_km1)
            A = m_2
            B = -m_1 - m_2 - 1/tau
            C = m_1
            F = -T_k_i/tau
            self.alphas[i] = -C/(A*self.alphas[i-1] + B)
            self.bettas[i] = (F - A*self.bettas[i-1])/(A*self.alphas[i-1] + B)
        self.Ts[self.Ts.shape[0]-1] = T_up
        for i in range(self.Ts.shape[0]-2,-1,-1):
            self.Ts[i] = self.Ts[i+1]*self.alphas[i] + self.bettas[i]
        self.time += tau



@cython.final
cdef class TermalConductBarrel:
    @cython.wraparound(True)
    def __cinit__(self, d, d_1, length0, length1, n_cells, n_nodes=50, T_0=293, delta_b=7800, c_b=480, lambda_b=27, time=0):
        self.n_cells = n_cells
        self.T_0 = T_0
        self.cells = [Termo1d.get_standart(r_0=d/2, r_1=d_1/2, n=n_nodes, T_0=T_0, 
                delta_b=delta_b, c_b=c_b, lambda_b=lambda_b, time=time)
            for i in range(self.n_cells)]
        xs = np.linspace(-length0, length1, n_cells+1)
        self.xs = xs
        self.cells_centers = (xs[1:] + xs[:-1])/2
        self.T_ws = np.zeros_like(self.cells_centers)
        self.T_ws[:] = T_0
        self.q_ws = np.zeros_like(self.cells_centers)


    cpdef dict get_state(self):
        return {
            'cells_centers': np.array(self.cells_centers),
            'q_ws': np.array(self.q_ws),
            'termos': [c.get_state() for c in self.cells]
        }


    cpdef void fill_T_ws(self, double[:] xs, double[:] T_ws):
        cdef:
            int i
            Termo1d cell
        for i in range(self.n_cells):
            cell = <Termo1d>(self.cells[i])
            self.T_ws[i] = cell.Ts[0]
        fill_averages(self.T_ws, self.xs, xs, T_ws)

    cpdef void fill_self_q_ws(self, double[:] xs, double[:] q_ws):
        fill_averages(q_ws, xs, self.xs, self.q_ws)

    cpdef void step(self, double tau):
        cdef:
            int i
            Termo1d cell
        for i in range(self.n_cells):
            cell = <Termo1d>(self.cells[i])
            cell.step(tau, self.q_ws[i], self.T_0)

        
        

        


# if __name__ == "__main__":
#     l1 = Termo1d.get_standart()
#     for i in range(100):
#         l1 = l1.step_up(0.0001, 0.2, 273)
#     i=0

