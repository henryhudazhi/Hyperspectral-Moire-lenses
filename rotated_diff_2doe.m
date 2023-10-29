
%% two DOEs realize rotation diffraction 
%% radial wavelength-dependent

%% parameters
lambda0 = 550e-9;
f = 100e-3;
theta = pi/6;
% lambda = 550e-3;
lambdamax = 700e-9;
lambdamin = 460e-9;
n = 1;
R = 1e-3;
N = 1000;
dx = R/N;

% a = 1.1e8;
% b = 7.42e4;
% c = 10;
         
a = pi/lambda0/f/theta;

b = 2*pi/lambda0/theta*(lambdamax-lambdamin)/R;
c = 2*pi/lambda0/theta*lambdamin;
d = b;

x = linspace(-R,R-dx,N);
y = x;
[X,Y] = meshgrid(x,y);
[t,r] = cart2pol(X,Y);

Fr = round(a*r.^2 + b*r + c);
Fr(r>R)=0;
theta = pi/180*32;

T = exp(1i.*Fr.*theta);
T(r>N/2)=0;
figure();
imshow(T,[])
title('Transmission function of joint DOEs')
set(gca,'FontName','Times New Roman','FontSize',15);

h = -mod(Fr*theta,2*pi)./2/pi*lambda0/0.4506;
figure();
imshow(h,[])
% xlabel('wavelength(nm)','FontName','Times New Roman','FontSize',25)
% ylabel('x(μm)','FontName','Times New Roman','FontSize',25);
save('h_test_rot.mat','h')

T1 = exp(1i.*Fr.*t);
T2 = exp(-1i.*Fr.*(t-theta));

figure();
imshow(T1,[])

figure();
imshow(T2,[])

h1 = -mod(Fr.*t,2*pi)./2/pi*lambda0/0.4506;
h2 = -mod(Fr.*(t-theta),2*pi)./2/pi*lambda0/0.4506;
figure();
imshow(h1,[])
figure();
imshow(h2,[])
h3 = h1-h2;
figure();
imshow(h3,[])

T3 = exp(1i.*2*pi/lambda0*0.4506.*(h1-h2));
figure();
imshow(T3,[])

