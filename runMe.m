% run-me file for Sheffield University ptychography reconstruction code.
% Ensure the data file is stored in the working directory, or load the data
% seperately and comment out the 'load' statement.
% Parameters are set for running WASP.
% For RAAR: try recon.beta = 0.85
% For rPIE: try recon.alpha = 0.1, recon.beta = 1
% For ePIE: usually recon.alpha = recon.beta = 1
% DM and ER do not require tuning parameters.
% HIVE (parallel WASP) requires the additional fields "recon.numWorkers"
% and "recon.subIters"
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% Citation for this data and code:                                        %
% Andrew. M. Maiden, Wenjie Mei and Peng Li,                              %
% "WASP: Weighted Average of Sequential Projections for ptychographic     %
% phase retrieval,"                                                       %
% XXX, pp. XX-XX (2024).                                                  %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% change the filename here to load different datasests. 
load('OpticalPtychoDataExample.mat');

% set the reconstruction parameters
recon.iters      = 2000;
recon.gpu        = 1;            
recon.alpha      = 2;         
recon.beta       = 1;        
recon.upLimit    = 2;       

% run the algorithm
[obj, probe] = WASP(expt, recon, initProbe);