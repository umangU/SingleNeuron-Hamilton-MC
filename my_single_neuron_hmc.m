clc;
clear;
close all;
warning off;

% Loading the A3 dataset
data=load('A3.dat');

tI = 2;
tB = [3;0;1]';
tNS = 1;

% For reproducibility
rng('default') 
mu = data.*tB + tI;
targ = normrnd(mu,tNS);

IMean = 0;
ISigma = 10;
BMean = 0;
BSigma = 10;
LogMean = 0;
LogSigma = 3;

% Log Posterior function
logpdf = @(Parameters)logPosterior(Parameters,data,targ(:,3),IMean,ISigma,BMean,BSigma,LogMean,LogSigma);
Interceptpoint = randn;
Beta = randn(size(data,2),1);
LogVariance = randn;
startpoint = [Interceptpoint;Beta;LogVariance];
smplr = hmcSampler(logpdf,startpoint,'NumSteps',50);
[MAPp,fInfo] = estimateMAP(smplr,'VerbosityLevel',0);
MAPp(4,:) = [];
MAPInter = MAPp(1);
MAPBeta = MAPp(2:end-1);
MAPLogNoiseVariance = MAPp(end);

% Plotting HMC
figure
plot(fInfo.Iteration,fInfo.Objective,'ro-');
xlabel('Iteration');
ylabel('Negative log density');

% Auto-Correlation Plot
figure
autocorr(MAPp)
title('Auto correlation plot for Hamiltonian Monte Carlo');
% Contour Plot
figure
nn1=[sin(MAPp) cos(MAPp)];
contour(nn1)
title('Contour plot for Hamiltonian Monte Carlo');

nn1=[sin(MAPp);cos(MAPp)];
nn1=round(abs(nn1));
disp('FAR for Hamiltonian Monte Carlo ')
FAR=nnz(nn1~=data(:,3))/8;
disp(FAR)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Metropolis Monte Carlo
rng default;
% For reproducibility
alpha1 = mean2(data);
beta1 = std2(data);
pdf = @(x)gampdf(x,alpha1,beta1); 
% Target distribution
proppdf = @(x,y)gampdf(x,floor(alpha1),floor(alpha1)/alpha1);
proprnd = @(x)sum(exprnd(floor(alpha1)/alpha1,floor(alpha1),1));
nsamples = 4;
smpl = mhsample(1,nsamples,'pdf',pdf,'proprnd',proprnd,'proppdf',proppdf);

figure
% Auto Correlation plot for Metropolis MC
autocorr(smpl )
title('Auto correlation plot for Metropolis Monte Carlo');
figure
nn2=[sin(smpl) cos(smpl)];
contour(nn2)
title('Contour plot for Metropolis Monte Carlo');

nn2=[sin(smpl);cos(smpl)];
nn2=round(abs(nn2));
disp('FAR for Metropolis Monte Carlo')
FAR=nnz(nn2~=data(:,3))/8;
disp(FAR)
