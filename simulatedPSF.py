## The process of obtaining the PSFs
## LightPipes
## Written by Haiquan Hu
from LightPipes import *
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.io import loadmat


# Loading the height maps of 2 DOEs
#Here, h1 and h2 belongs to the rotation angle of 30
path1 = 'h1.mat'
data1 = loadmat(path1)
h1 = data1['h1']
h1 = h1*um

path2 = 'h2.mat'
data2 = loadmat(path2)
h2 = data2['h2']
h2 = -h2*um                # The symbol here is important

## Parameters
deltaN=0.4506              # Refractive index difference
size=1*mm                  # The size of the square grid (the same as the size of the DOE here)
wavelength=0.7*um          # The wavelength selected
N=2000                     # The grid dimension
N2=1000
R=0.5*mm                   # The radius of the DOE
z = 100*mm                 # The distance of Fresnel diffraction
dz = 1.5*mm                # The distance between DOE1 and DOE2 (the ideal value is zero)

Int=np.empty([N,N])        # The amplitude of the input field
phase1=np.empty([N,N])     # The phase of DOE1
phase2=np.empty([N,N])     # The phase of DOE2
for i in range(0,N):
    for j in range(0,N):
        Int[i][j] = 1
        phase1[i][j] = 2*np.pi/wavelength*deltaN*h1[i][j]
        phase2[i][j] = 2*np.pi/wavelength*deltaN*h2[i][j]

# Initialize the input field
F=Begin(size,wavelength,N)
# The phase modulation by DOE1
F=SubIntensity(F,Int)
F=SubPhase(F,phase1)
F=CircAperture(R,0,0,F)
# The free space propagation (Forvard or Steps both OK)
F = Forvard(dz,F)
# The phase modulation by DOE2
F=MultIntensity(F,Int)
F=MultPhase(F,phase2)
F=CircAperture(R,0,0,F)
# The Fresnel diffraction
F = Fresnel(F,z)
I=Intensity(F,1)


# Results
fig=plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(I); ax1.axis('off'); ax1.set_title('Intensity')
X=np.linspace(-size/2,size/2,N)
ax2.plot(X/mm,I[N2]); ax2.set_xlabel('x[mm]'); ax2.set_ylabel('Intensity')
# ax2.set_xlim(-1,1)
name = f'd_{dz/mm}.png'
plt.imsave(name,I)
plt.show()
