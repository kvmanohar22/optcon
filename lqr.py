import numpy as np
from numpy import matmul as mm
from numpy.linalg import inv

import matplotlib.pyplot as plt
plt.style.use('ggplot')

class LQR_discrete:
    """ Discrete LQR finite horizon
    """
    def __init__(self,
            A, B, C, D,
            Q, Qf, R,
            X0, N):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        self.N = N

        self.X0 = X0
        self.Qf = Qf
        self.N  = N

        self.Pt = np.zeros((self.N+1, *self.Qf.shape), dtype=self.Qf.dtype)
        self.Kt = np.zeros((self.N, 1, 2))

        # optimal input
        self.Ut = np.zeros((self.N, 1, 1))

        # Trajectory
        self.Xt = np.zeros((self.N+1, 2, 1))
        self.Xt[0] = X0

    def summary(self):
        """ Print summary of the problem
        """
        pass

    def solve(self):
        """ Solve the finite horizon problem
        """
        # Compute Pt[...] matrices 
        self.Pt[self.N] = self.Qf
        for t in range(self.N, 0, -1):
            atpb = mm(mm(self.A.T, self.Pt[t]), self.B)
            mid = self.R+mm(mm(self.B.T, self.Pt[t]), self.B)
            self.Pt[t-1] = self.Q + mm(mm(self.A.T, self.Pt[t]), self.A) - mm(mm(atpb, inv(mid)), atpb.T)

        # Compute Kt[...] matrices
        for t in range(0, self.N):
            atpb = mm(mm(self.B.T, self.Pt[t+1]), self.A)
            mid = self.R+mm(mm(self.B.T, self.Pt[t+1]), self.B)
            self.Kt[t] = -mm(inv(mid), atpb)
        
        # Compute optimal input and trajectory
        for t in range(0, self.N):
            self.Ut[t] = mm(self.Kt[t], self.Xt[t])
            self.Xt[t+1] = mm(self.A, self.Xt[t])+mm(self.B, self.Ut[t])

    def analysis(self):
        """ Do some analysis
        """
        # plot norm of input
        plt.plot(np.arange(0, self.N), self.Ut[:, 0, 0])
        plt.xlabel('t')
        plt.ylabel('$u_t$')
        plt.title('Input as a function of time')
        plt.show()

        # plot trajectory
        Xs = self.Xt[:, 0, 0] 
        Ys = self.Xt[:, 1, 0] 
        plt.plot(np.arange(0, self.N+1), Xs, '-', label='Xt')
        plt.plot(np.arange(0, self.N+1), Ys, '-', label='Yt')
        plt.plot(np.arange(0, self.N+1), Xs, '.')
        plt.plot(np.arange(0, self.N+1), Ys, '.')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('X')
        plt.show()



def test1():
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0],[1]])
    C = np.array([[1, 0]])
    D = 0
    Q = Qf = np.matmul(C.T, C)
    R = 10 * np.eye(1)
    
    # Initial conditions
    X0 = np.array([[1], [0]])
    N  = 20

    # Instantiate
    lqr = LQR_discrete(A, B, C, D, Q, Qf, R, X0, N)
    lqr.summary()
    
    # Solve 
    lqr.solve()
    lqr.analysis()

if __name__ == '__main__':
    test1()

