from .hopping_Hamiltonian import *
from .circulant_matrix import *

class dimerized_hopping( hopping_Hamiltonian ):

    def __init__( self, L, large_coupling = 1, small_coupling = 0.1, PBC = None ):
        #super.__init__( h ) TODO
        self.set_default_init_flags()
        n = int( L / 2 )
        Js = [ large_coupling, small_coupling ] * n
        Js.append(large_coupling)
        h = np.diag( Js, -1 ) 
        h = h + h.T
        self.set_couplings( h )


