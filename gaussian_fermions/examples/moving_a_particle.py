from gaussian_fermions import * #install with 'pip install gaussian_fermions'

L =14 #System length
cov_ini = np.diag([0]*L+[1]+[0]*(L-1))

H_quench = nearest_neighbor_hopping( cov_ini.shape[0] )
H_quench.h = H_quench.h + 10*np.diag( [x**2 for x in np.linspace(0,1, cov_ini.shape[0])])
H_quench.show_cov_evolution( cov_ini, 3*L, show_flag =1 )

