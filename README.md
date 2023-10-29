# Hyperspectral-Moire-lenses

The process of simulation in Snapshot hyperspectral imaging based on tunable diffractive Moire-lenses.
##
get_H.m: get the height maps of the DOEs

forward.m: get the simulated RGB PSFs 

simulatedPSF.py: verify our PSF simulation by LightPipes

(You can find LightPipes here http://www.okotech.com/lightpipes)

h1.mat, h2.mat: the height maps of 2 DOEs when the rotation angle is 30

h_460.mat, h_700.mat: the height map of 2 DOEs as a whole when the rotation angle is 45 and 30, respectively

data_new.mat: the RGB response curves of the camera, range from 400nm-700nm (interval 10nm)

rotated_diff_2doe : the code for Algorithm 1

get_blurred_images： the code for Figure 7
