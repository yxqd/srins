import numpy as np
import matplotlib.pyplot as plt


data = np.load('I(first column), E(second column), Q(third column).npy')

[Nx,Ny] = np.shape(data[0])

v1 = np.array([10,20])
v2 = np.array([-4,8])
BM = np.zeros([2,2])
BM[:,0] = v1/np.sqrt(np.sum(v1**2))
BM[:,1] = v2/np.sqrt(np.sum(v2**2))
RM = np.linalg.inv(BM)
# Parameters to optimize#
ix_c = 100
iy_c = 910
a1 = RM[0,0]
a2 = RM[0,1]
a3 = RM[1,0]
a4 = RM[1,1]
# Define data #
I = data[0]


# Do Rotation #
I_rot = np.zeros([Nx,Ny])

for ix in range(Nx):
    for iy in range(Ny):
        ixr = (ix-ix_c)*a1+(iy-iy_c)*a2+ix_c
        iyr = (ix-ix_c)*a3+(iy-iy_c)*a4+iy_c
        
        ix1 = np.int(np.floor(ixr))
        ix2 = np.int(np.ceil(ixr))
        if ix2==ix1:
            ix2 = ix2+1
        
        iy1 = np.int(np.floor(iyr))
        iy2 = np.int(np.ceil(iyr))
        if iy2==iy1:
            iy2 = iy2+1
        if iy2<=Ny-1 and iy1>=0 and ix2<=Nx-1 and ix1>=0:
            I_rot[ix,iy] = I[ix1,iy1]*(ix2-ixr)*(iy2-iyr)+\
                 I[ix2,iy1]*(ixr-ix1)*(iy2-iyr)+I[ix1,iy2]*(ix2-ixr)*(iyr-iy1)+\
                 I[ix2,iy2]*(ixr-ix1)*(iyr-iy1)
        

plt.imshow(data[0][75:125,800:1000], aspect='auto', interpolation='none', origin='lower')
plt.show()

plt.imshow(I_rot[75:125,800:1000], aspect='auto', interpolation='none', origin='lower')
plt.show()


