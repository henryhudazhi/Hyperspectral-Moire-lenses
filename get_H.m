%% get the height map of the DOEs
%% Written by Haiquan Hu
clear all; 
clc;
% Defining parameters
% unit um
N = 2000;                                       % The grid dimension
L = 1000;                                       % The length of the grid
dx = L/N;                                       
f = 100e3;                                      % focal length of the DOEs
lambda = 0.700;                                 % wavelength in micron

theta = 30/180*pi;                              % define an angle for a                           
a = pi/f/theta/lambda;                          % get the value of a


x =linspace(-L/2,L/2-dx,N);
y = x;
[X,Y]=meshgrid(x,y);
[t,r] = cart2pol(X,Y);
Fr = round(a*r.^2);                             % get F(r)
Fr(r>L/2)=0;

theta = pi/180*30;                              % the rotation angle (range from 30-45)
T = exp(-1i.*Fr*theta);                         % the transmittance of combined DOEs
T(r>N/2)=0;
figure();
imshow(T,[])

h = -mod(Fr*theta,2*pi)./2/pi*lambda/0.4506;    % the equivalent combined DOEs
figure();
imshow(h,[])
% save('h_twodoe700.mat','h')                   
T1 = exp(1i.*Fr.*t);                            % the transmittance of single DOE
T2 = exp(-1i.*Fr.*(t-theta));

figure();
imshow(T1,[])

figure();
imshow(T2,[])

h1 = -mod(Fr.*t,2*pi)./2/pi*lambda/0.4506;      % the height map of single DOE (the refractive index difference is a constant 0.4506 here for simplify)
save('h1.mat','h1')
h2 = -mod(Fr.*(t-theta),2*pi)./2/pi*lambda/0.4506;
save('h2.mat','h2')
figure();
imshow(h1,[])


figure();
imshow(h2,[])
