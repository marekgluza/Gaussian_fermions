import numpy as np
import scipy as scp
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
import functools as fc

#Doesnt run TBD from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit

from .tunneling_covariance_matrix import *
from .circulant_matrix import *

class hopping_Hamiltonian:
    
    def __init__( self, h ):
        """
        Constructs the class simulating quasifree or quadratic particle number preserving (PNP) Hamiltonians.
        It uses the convention $H_\text{PNP}( h ) = \sum_{ j, k = 1 } ^ L h_{ j, k } \hat f^\dagger_j \hat f_k$


        The couplings h are stored as dense because propagators for long time are dense anyhow.

        The class assumes that $h$ will be implemented in a child class and provides:
        - Green function
        - single-particle eigenenergies and eigenmodes
        - covariance matrix of the ground state
        - covariance matrix of the thermal state at inverse temperature $beta$
        - evolution of an input covariance matrix under $h$
        - setup routines for typical couplings
        - visualization functions


        :param: h: Hermitian couplings
        """

        self.set_default_init_flags()#TODO: ensure that this runs first also in inheriting classes?
        self.set_couplings( h )

    def set_default_init_flags( self ):
        self.flags = {}
        self.flags['propagator_type'] = 'default'
        self.flags['trotterization_type'] = 'default'

    def set_couplings( self, h ):
        """
        Sets couplings and makes sure relevant flags are updated.

        :param: h: Hermitian couplings
        """

        #TBD FIX THIS self.check_couplings( h )

        self.L = h.shape[1]
        self.h = h
        self.flags['couplings_decoupled'] = False


    def check_couplings( self, h ):
        self.check_accuracy( np.linalg.norm( h - h.T.conj() ), "Couplings must be Hermitian.")
    
    
    def decouple_couplings(self, h = None ): #TODO: This should come from class normal_form_symmetric_matrix
        """
        Diagonalizes couplings.
        If h is None then the self couplings will be decoupled if they have not been decoupled already and
        the sorted energy self.e, corresponding eigenmodes self.R and the docoupled_couplings flag will be set.
        If h is not None then the passed couplings will be docoupled and sorted energy e and corresponding eigenmodes R will be returned.

        The convention is that np.diag( e ) = R.T.conj().dot( h.dot(R ) ) up to 10e-14 error.
        Equivalently h = R e R^dagger or R[ :, j ] for eigenmode j .

        :param: h: couplings
        """

        if h is None:
            if self.flags['couplings_decoupled'] == False:
                e, R = np.linalg.eigh( self.h )
                self.check_accuracy( np.linalg.norm( R.T.conj().dot( self.h.dot( R ) ) - np.diag( e ) ) , "Diagonalization unsuccessful") 
                e_argsort = e.argsort()
                R = R.T[e_argsort]
                R = R.T
                e.sort()
                self.e = e
                self.R = R.conj()
                self.flags['couplings_decoupled'] = True
            else:
                e = self.e
                R = self.R

        else:
            self.check_couplings( h )
            e, R = np.linalg.eigh( h )
            self.check_accuracy( np.linalg.norm( R.T.conj().dot( h.dot( R ) ) - np.diag( e ) ) , "Diagonalization of auxilary couplings unsuccessful" )
        
        return e, R
            
            
    def check_accuracy( self, value, threshold_value = 1e-14, msg = None ):
        """
        Tests if value is less than assigned threshold accuracy. If not raises Exception.

        :param: value: checked value
        :param: threshold_value: threshold value
        """
        return 
        if value > threshold_value:
                if msg is not None:
                    raise Exception ( msg + "Inaccuaracy detected val = " + str(value) + " > " + str(threshold_value) )
                else:
                    raise Exception ("Inaccuaracy detected val = " + str(value) + " > " + str(threshold_value) )
                
    def G( self, t, h = None):
        """
        Wrapper for propagator, in general could be e.g. a Trotterized propagator.

        :param: t: evolution time 
        :param: h: Hermitian couplings
        """
        return self.propagator( t, h )

    def propagator( self, t, h = None ):
        """
        Calculates the propagator/Green function $G(t)=\exp( - \i t h )$

        Possible calculation:
        1) self.flags['propagator'] == 'default' uses direct matrix exponential or use decoupled matrices
        2) via decoupled couplings multiplied with $e^{ - \i t E_k }$

        :param: t: evolution time 
        :param: h: Hermitian couplings
        """
                
        if self.flags['propagator_type'] == 'default':
            
            if h is None:
                #return scp.linalg.expm( -1j * t * self.h )
                return expm( -1j * t * self.h )

            else:
                #self.check_couplings( h )
                #return scp.linalg.expm( -1j * t * h )
                return expm( -1j * t * h )

        else:#TODO: test this
            e, R = self.diagonalize_couplings( h )
            return R.T.conj().dot( np.diag( np.exp( - 1j * t * e ).dot( R ) ) )



    def trotterized_propagator( self, h_t, t, nmb_steps ):
        """
        Trotterized propagator. The convention is that the gates are applied to the right e.g. G_trotterized(2,3) =  G(2) * G(1) * G(0).

        :param: h_t: time dependent couplings (function)
        :param: t: evolution time
        :param: nmb_steps: number of Trotter steps
        """
        assert h_t(0).shape == ( self.L, self.L ), "wrong couplings size" #This makes sure that the propagator will be of right size to interact with other object like the covariance matrix etc. but could in principle be less restrictive
        if nmb_steps <= 0:
            nmb_steps = 1
        dt = t / nmb_steps
        if self.flags['trotterization_type'] == 'default':
            G = np.eye( self.L )
            for n in range( nmb_steps ):
                G = self.propagator( dt, h_t( n * dt ) ).dot( G )
            return G

        #elif self.flags['trotterization_type'] == 'second_order': #TODO
        elif self.flags['trotterization_type'] == 'linearized':
            Id = np.eye( self.L )
            G = Id
            for n in range( nmb_steps ):
                h.check_couplings( h )
                G = ( Id - 1j * dt * h_t( n * dt ) ) * G
            return G

        else:
            raise Exception( "Trotterization type uknown" )


    def cov_thermal( self, beta, mu = None ):
        """
        Calculates the covariance matrix of a thermal state $\rho_\beta = exp ( -\beta H(h) ) / Z_\beta$ at inverse temperature $\beta$.

        :param: inverse temperature
        """

        if self.flags['couplings_decoupled'] == False:
            self.decouple_couplings()
        if mu == None:
            mu = 0;

        D = np.diag([ np.exp( - beta * E + mu ) / ( 1 + np.exp( - beta * E + mu ) ) for E in self.e ])
        return self.R.dot( D.dot( self.R.T.conj() ) ) 

    def cov_gnd( self, mu = None ):
        """
        Calculates the covariance matrix of a gnd state (degenaracy not resolved)
        :param: inverse temperature
        """

        if self.flags['couplings_decoupled'] == False:
            self.decouple_couplings()
        if mu == None:
            mu = 0;
        else:
            raise Exception("Gnd with chemical potential not implemented")

        N_k = [ np.heaviside( -E, 0 ) for E in self.e ]
        print( [ (np.heaviside( -E,0 ), E) for E in self.e ])
        D = np.diag( N_k )
        #fig= plt.figure()
        #plt.plot ([ np.heaviside( -E, .5 ) for E in self.e ] )
        
        return  self.R.dot( D.dot( self.R.T.conj() ) ) 



    def cov_evolve(self, C, t ):
        """
        Evolves an input covariance matrix $C$ forwards by time $t$ using $C(t) = G(t) C G(t)^\dagger$.

        :param: C: covariance matrix
        :param: t: evolution time
        """
       # if isinstance( C, tunneling_covariance_matrix ) is not True:
       #     raise Exception("Attempting to evolve a non PNP covariance matrix with a PNP quasifree Hamiltonian")
        G_t = self.G( t )
        C_evolved = ( G_t.dot( C.dot( G_t.T.conj() ) ) )
        #C.check( C_evolved )
        return  C_evolved  

    def cov_dephase( self, C ):
        raise Exception("Not implemented")
        
    def cov_dephase_translation_invariant_nondegenerate( self, C ):
        """
        Calculates the fully dephased covariance matrix under a translation invariant nondegenerate Hamiltonian.
        It is done by picking all off-diagonals with PBC and taking an average of the currents.

        :param: C: covariance matrix
        """
        if isinstance( C, tunneling_covariance_matrix ) is True:
            cov = C.cov
        else:
            cov = C

        I_j = [0]*self.L
        for j in range( 1, int(self.L) ):
            I_j[ j ] = 0.5 * ( np.sum( list( np.diag( cov, j ) ) + list( np.diag( cov, self.L - j ) ) ) / self.L )
        I_j[ 0 ] = 1.0 * np.sum(  np.diag( cov )  ) / self.L #The 1.0 avoids integer division! (Input 010101 was giving a 0!)
        return circulant_matrix.circulant_couplings( I_j, self.L ) 

    def cov_dephase_translation_invariant_degenerate( self, C ):
        """
        NAIVE Implementation

        :param: C: covariance matrix
        """
        if isinstance( C, tunneling_covariance_matrix ) is True:
            cov = C.cov
        else:
            cov = C
        ##Assert that size ok
        cov_dephased = np.zeros( cov.shape )
        for j in range( self.L ):
            for k in range( self.L ):
                d = abs( j - k )
                cov_dephased[ j, k ] = np.sum( [ cov[ x, (x + d)%self.L ] + cov[ (j + x)%self.L, (k - x)%self.L ] for x in range( 1, self.L ) ] ) 

        return cov_dephased / self.L

    def cov_calc_quasiparticle_occupation( self, C ):
        if isinstance( C, tunneling_covariance_matrix ) is True:
            cov = C.cov
        else:
            cov = C

        E, U = self.decouple_couplings()
        #n_qp = np.diag( U.dot( cov.dot( U.T.conj() ) ) )
        n_qp = np.diag( U.T.conj().dot( cov.dot( U ) ) )
        return n_qp
    def Fermi_Dirac( self,  E, beta, mu ):
            n_qp_E = np.exp(  -beta * E + mu ) / ( 1 + np.exp(  - beta * E + mu ) )
            return n_qp_E

    def cov_quasiparticle_occupation_fit_thermal( self, C ):
        
        n_qp = self.cov_calc_quasiparticle_occupation( C )

        def objective_function( params, e, n_qp ):
            beta = params['beta']
            mu = params['mu']
            FD = np.exp(  -beta * e + mu) / ( 1 + np.exp(  -beta * e + mu) )
            return FD - n_qp
        
        params = Parameters()
        params.add('beta',   value= 3,  min=0, max = 100 )
        params.add('mu',   value= 0,  min=-10, max = 10 )

        minner = Minimizer( objective_function, params, fcn_args = ( self.e, n_qp ) )
        result = minner.minimize()
        
        final = n_qp + result.residual
        beta_opt = result.params['beta'].value
        mu_opt = result.params['mu'].value
        print("Inverse temp fit: ", beta_opt)
        print("Chemical potential fit fit: ", mu_opt)
               
        n_qp_thermal = [ self.Fermi_Dirac( E, beta_opt, mu_opt ) for E in self.e]
        report_fit(result)
        return [ n_qp, n_qp_thermal, beta_opt, mu_opt ]
    def show_quasiparticle_occupation_fit_thermal( self, C ):
        fig = plt.figure()
        [ n_qp, n_qp_thermal, beta_opt, mu_opt ] = self.cov_quasiparticle_occupation_fit_thermal( C )
        plt.plot( range(self.L), n_qp, 'k+')
        plt.plot( range(self.L), n_qp_thermal, 'r')

        plt.show()
        return fig
   

    def show_cov_evolution(self, C, T , show_flag = None ):
        """
        Shows time evolution of covariance matrix $C$ with a time-slider.

        :param: C: PNP covariance matrix
        :param: T: upper time range [ 0, T = 1.5 L (as default) ]
        """
        def wrapper( t ):
            return abs( self.cov_evolve( C, t )  )
        self.show_matrix_slider( wrapper, T_min = 0, T_max = T, slider_label = 'Time t', show_flag = show_flag)

    def show_propagator(self, T_min = 0, T_max = None, show_flag = 1 ):
        """
        Shows the propagator matrix $G(t)$ with a time-slider.

        :param: T: upper time range [ 0, T = 1.5 L (as default) ]
        """
        if T_max is None:
            T_max = self.L
        def wrapper( t ):
            return abs( self.G( t ) )
        self.show_matrix_slider( wrapper, T_min = T_min, T_max = T_max,  slider_label = 'Time t', show_flag = 1 )

    def show_cov_thermal(self, beta_max ):
        """
        Shows the thermal covariance matrix $C_\beta$ with a time-slider.

        :param: beta_min: lower inverse temparature range 
        :param: beta_max: upper inverse temparature range 
        """

        show_matrix_slider( self.cov_thermal, T_min = 0, T_max = beta_max, slider_label = 'Inverse temp. beta'  )


    def show_matrix_slider(self, matrix_func, T_min, T_max, slider_label = None, show_flag = None ):
        """
        Shows a matrix.

        :param: matrix_func: returns the matrix at slider value
        :param: T_min: lower slider range 
        :param: T_max: upper slider range
        :param: slider_label: slider label
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.25, bottom=0.25)

        im1 = ax.imshow( matrix_func( 0 ), interpolation='none')
        fig.colorbar(im1)

        slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
        slider = Slider( slider_ax, slider_label,  T_min, T_max, valinit=0 )

        def update(val):
            im1.set_data( matrix_func( slider.val ) )
            fig.canvas.draw()

        slider.on_changed(update)
        if show_flag is not None:
            plt.show()
        

    def show_columns(self, M):
        raise Exception("Not implemented")
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        l, = plt.plot(M[0], lw=2, color='red')

        axcolor = 'lightgoldenrodyellow'
        axamp = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        samp = Slider(axamp, '# Eigenvector', 0, self.L, valinit=0)

        def update(val):
            amp = samp.val
            l.set_ydata(M[amp])
            fig.canvas.draw_idle()

        samp.on_changed(update)

        plt.show()


   
