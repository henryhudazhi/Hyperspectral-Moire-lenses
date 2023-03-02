%% forward propagation
%% Written by Hao Zhou and Haiquan Hu

%unit um 
clear;
digits(64);
format long;

% load the RGB response curves of the camera, range from 400nm -700nm
load data_new.mat

% load the height map
load('h_700.mat');

lambda1 = linspace(460e-3, 700e-3,25);          % lambda range 460nm - 700nm
f=100e3;                                        % focal length
N=2000;                                         % grid dimension
L=1000;
dpixel = 5.5;                                   % pixel size
deltaN=0.4506;                                  % refractive index difference
 for i = 1:25;
    k=2*pi/lambda1(i);                          % wave number
    dx=L/N;
    x=linspace(-L/2,L/2-dx,N);
    y=-linspace(-L/2,L/2-dx,N);
    [x,y]=meshgrid(x,y);

    A=1;
    phi=2*pi/lambda1(i)*deltaN.*h;
    cof=exp(1i*pi/lambda1(i)/f*(x.^2+y.^2));
    amp=A.*exp(1i*phi).*cof;
    sensor=fft2(amp);
    sensor=fftshift(sensor);
    sensor1=sensor.*conj(sensor);

    rr = 1/(lambda1(i)*f) * L * dpixel;      %downsampling index

    Sensor = interp2(x,y,sensor1,x*rr,y*rr);
    Sensor(find(isnan(Sensor)==1)) = 0;

    Sensor=Sensor./max(max(Sensor(:)));
    Sensor=Sensor(975:1025,975:1025);
    SensorR = r(i+6) * Sensor;
    SensorG = g(i+6) * Sensor;
    SensorB = b(i+6) * Sensor;
    Sensor1 = cat(3,SensorR,SensorG,SensorB);
    Sensor1=Sensor1./max(Sensor1(:));

    outname=strcat(num2str(1000*lambda1(i)),'.bmp');
    imwrite(Sensor1,outname);
    PSF{i} = Sensor1./max(Sensor1(:));
 end

%% total PSF
IMG = sum(cat(4,PSF{:}), 4);
figure();
imshow(IMG./max(IMG(:)),[])
% imwrite(IMG./max(IMG(:)),'all.bmp');
%% single wavelength psf
for i = 1:25;
    psf(:,:,i,1) = PSF{i}(:,:,1)./sum(sum(IMG(:,:,1)));
    psf(:,:,i,2) = PSF{i}(:,:,2)./sum(sum(IMG(:,:,2)));
    psf(:,:,i,3) = PSF{i}(:,:,3)./sum(sum(IMG(:,:,3)));
end

psf = single(psf);
save('psf_700.mat','psf')