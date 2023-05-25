from .hopping_Hamiltonian import *
from .circulant_matrix import *

class nearest_neighbor_hopping( hopping_Hamiltonian ):

    def __init__( self, L, PBC = None, Js = None ):
        #super.__init__( h ) TODO
        self.set_default_init_flags()
        if Js is not None:
             h = np.diag( Js, -1 ) 
             h = h + h.T.conj()
        else:
            if PBC is not None:
                h = circulant_matrix( self.J, L )            
                self.J = [ 0, 1 ] #Convention: only first non-trivial couplings are necessary
                h = circulant_matrix( self.J, L ).h
            else:
                h = np.diag( [1]*(L-1),-1) 
                h = h + h.T.conj()
        self.set_couplings( h )


