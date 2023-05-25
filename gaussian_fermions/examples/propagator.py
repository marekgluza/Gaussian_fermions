from gaussian_fermions import * #install with 'pip install gaussian_fermions'

L =140 #System length

H_quench = nearest_neighbor_hopping( L )
H_quench.show_propagator(0,100)

