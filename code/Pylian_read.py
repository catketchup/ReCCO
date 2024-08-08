import Pk_library as PKL

As= conf.As
ns = conf.ns
k_in = np.linspace(1e-3, 1e-1, 2000)
k_pivot = 0.05
Pk_in = As*(k_in/k_pivot)**(ns-1)*1e-9

if df_3D is not None:
    del df_3D
grid              = 56  #grid size
BoxSize           = 10000.0 #Mpc/h
seed              = 1      #value of the initial random seed
Rayleigh_sampling = 0      #whether sampling the Rayleigh distribution for modes amplitudes
threads           = 1      #number of openmp threads
verbose           = True   #whether to print some information

# read power spectrum; k and Pk have to be floats, not doubles
# k, Pk = np.loadtxt('my_Pk.txt', unpack=True)

k_in, Pk_in = k_in.astype(np.float32), Pk_in.astype(np.float32)
# generate a 2D Gaussian density field
# df_2D = DFL.gaussian_field_2D(grid, k, Pk, Rayleigh_sampling, seed,
                              # BoxSize, threads, verbose)
# Pk_in = Pk_in/k_in**3

# generate a 3D Gaussian density field
df_3D = DFL.gaussian_field_3D(grid, k_in, Pk_in, Rayleigh_sampling, seed,
                              BoxSize, threads, verbose)

Pk = PKL.Pk(df_3D, BoxSize)
# 3D P(k)
k       = Pk.k3D
Pk0     = Pk.Pk[:,0] #monopole
Pk2     = Pk.Pk[:,1] #quadrupole
Pk4     = Pk.Pk[:,2] #hexadecapole
Pkphase = Pk.Pkphase #power spectrum of the phases
Nmodes  = Pk.Nmodes3D
