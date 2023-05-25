import numpy as np

class circulant_matrix:

    def __init__(self, J, L):
        self.L = L
        self.J = J
        self.h = self.circulant_couplings( J, L )
        self.couplings_decoupled_flag = False

    def full_matrix( self ):
        self.couplings_calculated_flag = True 
        self.h = self.circulant_couplings( J )
        return self.h
    
    @staticmethod
    def circulant_couplings( J, N ):
        #This could be made more efficient?
        h = np.zeros( [N,N] )
        for i in range( 1, len(J)):
            h[ np.arange( N - i ), np.arange( N - i ) + i ]  = [ J[i] for j in range( N-i ) ]
            h[ np.arange( i ), np.arange( i ) + N - i]  = [ J[i] for j in range( i ) ]
            #h = h + J[i] * ( np.diag( [1]* (N - i), i) +  np.diag( [1]*i, N - i) + np.diag( [1]* (N - i), -i) +  np.diag( [1]*i, - N + i) )
        h = h + h.T
        h = h + J[0] * np.eye( N )
        return h
    @staticmethod
    def DFT_matrix( N = None  ):
        if N is None:
            N = self.L

        DFT = np.zeros( [N, N], dtype=np.complex128 )
        for k in range( N ):
            DFT[ k, : ] = [ np.exp( 2 * np.pi * 1j * k * z / N ) for z in range( N ) ]
        return DFT
                
    def diagonalize_couplings_py(self):
        self.e, self.R = np.linalg.eigh(self.h)
        print("Diagonalizing couplings. Check:", np.linalg.norm( self.R.T.dot( self.h.dot(self.R ) ) - np.diag(self.e)) )
        self.couplings_decoupled_flag = True
    
    def modes_matrix( self ):
        return 0

    def spectrum( self ):
        e = []
        for j in range(self.L):
            e.append( sum([ self.J[k] * np.exp( 1j * np.pi * k * j / self.L) for k in range(self.L)  ] ) ) 
        self.calculated_spectrum_flag = True
        self.e = e
        return e

    def check_if_circulant( self ):
        return 0
