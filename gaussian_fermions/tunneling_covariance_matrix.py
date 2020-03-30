import numpy as np
from hopping_Hamiltonian import *

class tunneling_covariance_matrix:
    
    def __init__( self, cov = None, L = None ):
        """
        Constructs the class simulating particle number preserving (PNP) covariance matrix i.e. having onlu $\langle \hat f^\dagger_j \hat f_k \rangle$ correlations and no pairing correlations of the form $\langle \hat f^\dagger_j \hat f^\dagger_k  + \text{h.c.} \rangle$.

        The couplings h are stored as dense because propagators for long time are dense anyhow.

        The class assumes that $h$ will be implemented in a child class and provides:
        - Green function
        - single-particle eigenenergies and eigenmodes
        - covariance matrix of the ground state
        - covariance matrix of the thermal state at inverse temperature $beta$
        - evolution of an input covariance matrix under $h$
        - setup routines for typical couplings
        - visualization functions


        :param: L: the dimension of the couplings matrix which is the system size
        """
        self.set_default_init_flags()

        if cov is not None:
            self.L = cov.shape[0]
            self.set_cov( cov )

        elif L is not None:
            #Set vacuum matrix by default
            self.L = L
            self.set_cov( np.zeros( [ L, L ] ) )

        else:
            raise Exception("Cannot initialize empty covariance matrix")
        
    def set_default_init_flags( self ):
        self.flags = {}

    def set_cov( self, cov, check_cov_admissible = False ):
        """
        Sets covariance matrix and makes sure relevant flags are updated.

        :param: cov: covariance matrix
        :param: check_cov_admissible: flag deciding if cov is check to be physically admissible
        """

        if cov is not None:
            if check_cov_admissible is True:
                self.check_cov_admissible( cov, save = True )
            self.cov = cov
            self.L = cov.shape[0]

            self.flags['cov_decoupled'] = False

        else:
            raise Exception("You cannot create an empty covariance matrix")
    def check( self, cov, save = False ):
        self.check_cov_admissible(  cov, save = save )

    def check_cov_admissible( self, cov, save = False ):
        #TODO check if is real
        n, R = self.decouple_cov( cov )
        self.check_accuracy( np.linalg.norm( n - abs( n ) ), "Occupation numbers must be non-negative" )

        #Getting here means cov admissible
        if save is True:
            self.n = n
            self.R = R
            self.flags['cov_decoupled'] = True

    def current( self, z, cov = None ):
        """
        Returns the z-current.
        
        """
        if cov is None:
            cov = self.cov
        L = cov.shape[0]
        if z == 0:
            return np.diag( cov )
        else:
            return np.diag( cov, z ).tolist() + np.diag( cov, L - z ).tolist()

    def decouple_cov( self, cov = None ): #TODO: This should come from normal_form_hermitian_matrix
        """
        Diagonalizes couplings.
        
        The convention is that np.diag( n ) = R.T.conj().dot( C.dot( R ) ) up to 10e-14 error.
        Equivalently $C = R \cdot n \cdot R^dagger$ or R[ :, j ] for eigenmode j .
        """

        if cov is None:
            if  self.flags['cov_decoupled'] is False:
                n, R = np.linalg.eigh( self.cov )
                self.check_accuracy( np.linalg.norm( R.T.dot( self.cov.dot( R ) ) - np.diag( e ) ), "Diagonalization unsuccessful") 
                self.n = n
                self.R = R
                self.flag['cov_decoupled'] = True
            else:
                n = self.n
                R = self.R

        else:
            n, R = np.linalg.eigh( cov )
            self.check_accuracy( np.linalg.norm( R.T.dot( cov.dot( R ) ) - np.diag( n ) ) , "Diagonalization unsuccessful" ) 
        
        return n, R
            
            
    def check_accuracy( self, value, threshold_value = 10e-14 ):
        """
        Tests if value is less than assigned threshold accuracy. If not raises Exception.

        :param: value: checked value
        :param: threshold_value: threshold value
        """

        if value > threshold_value:
                raise Exception ( "Inaccuaracy detected val = ", val, " > ", val_th )
                

    
    def evolve( self, H, t ):
        """
        Evolves an input covariance matrix $C$ forwards by time $t$ using $C(t) = G(t) C G(t)^\dagger$.

        :param: H: PNP quasifree Hamiltonian
        :param: t: evolution time
        """
        if isinstance( H, PNP_quasifree_Hamiltonian ) is not True:
            raise Exception("Trying to evolve PNP covariance matrix with a non-PNP Hamiltonian")
        return H.evolve_cov( self, t ) 
    

    def show_cov_evolution( self, H, T_max = None ):
        """
        Shows time evolution of covariance matrix $C$ with a time-slider.

        :param: H: PNP quasifree Hamiltonian
        :param: T: upper time range [ 0, T = 1.5 L (as default) ]
        """
        if isinstance( H, PNP_quasifree_Hamiltonian ) is not True:
            raise Exception("Trying to evolve PNP covariance matrix with a non-PNP Hamiltonian")
        if T_max is None:
            T_max = 1.5 * self.L
        H.show_cov_evolution( self, T_max )
        
    
    def show_eigenmodes( self ):
        self.decouple_cov()
        self.show_columns( self.R.T )
    

#TODO: Import this
    def show_matrix_slider(self, matrix_func, T_min, T_max, slider_label = '' ):
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

        slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg='white')
        slider = Slider( slider_ax, slider_label, 0, T_min, T_max, valinit=0 )

        def update(val):
            im1.set_data( matrix_func( slider.val ) )
            fig.canvas.draw()

        slider.on_changed(update)
        plt.show()

    @staticmethod
    def show_cov( cov, inset_size = None, save_path = None, title = None, show_flag = None ):
        
        L = cov.shape[0]
        # position of the inset
        inset_pos_x = 0.52
        inset_pos_y = 0.17
        inset_height = 0.28
        inset_width = 0.28

        font_size = 22
        
        # outer plot
        fig = plt.figure( figsize = ( 9, 6.75 ) )
        ax = fig.add_subplot(111)
        if title == None:
            title = r'$\Gamma^{(\beta)}$'
        plt.title( title )
        plt.xlim((1,L))
        plt.ylim((1,L))
        plt.xlabel( 'Lattice site $x$', fontsize = font_size )
        plt.ylabel( 'Lattice site $y$', fontsize = font_size )
        #plt.clabel = '|\Gamma_{'+str( int(L/2) )+',s}(t)|'
        im = ax.imshow( cov, cmap='RdBu', aspect='equal', interpolation = None, extent = [ 1, L, L, 1])
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)
        
        cbar = plt.colorbar(im, cax=cax)
        range_plot = np.max( np.abs( cov ) )
        cbar.set_clim( -range_plot, range_plot )
        #cbar.set_label( r'$\Gamma^{(\beta)}_{x,y}$', fontsize = font_size )
        
        if inset_size is not None:
            #inset 
            inset = fig.add_axes( [ inset_pos_x, inset_pos_y, inset_width, inset_height ] )
            im2 = inset.imshow( cov[0:inset_size,0:inset_size], extent = [ 1, inset_size, inset_size, 1], cmap='RdBu', aspect='equal', interpolation = None)
            #divider2 = make_axes_locatable(inset)
            #cax2 = divider2.append_axes("right", size="2.5%", pad=0.05)
            #cbar2 = plt.colorbar(im2, cax=cax2)
            im2.set_clim( -range_plot, range_plot )
            inset.set_xlim( ( 1, inset_size ) )
            inset.set_ylim( ( 1, inset_size) )
            #inset.set_xlabel( 'x', fontsize= font_size )
            #inset.set_ylabel( 'y', fontsize= font_size )
        
        if save_path is not None:
            plt.savefig( save_path, format='pdf')
        if show_flag is not None:
            plt.show()

    def show_cov_inset( self, cov = [], inset_size = 10, save_path = None, title = None, show_flag = None ):
        if cov == []:
            cov = self.cov
        return self.show_cov( cov = cov, inset_size = inset_size, save_path = save_path, title = title, show_flag = show_flag )

    def show_columns(self, M):
        raise Exception("Not implemented")
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        l, = plt.plot(M[0], lw=2, color='red')

        axcolor = 'lightgoldenrodyellow'
        axamp = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        samp = Slider(axamp, '# Eigenvector', 0, np.shape( M )[1], valinit=0)

        def update(val):
            amp = samp.val
            l.set_ydata(M[amp,:])
            fig.canvas.draw_idle()

        samp.on_changed(update)

        plt.show()



