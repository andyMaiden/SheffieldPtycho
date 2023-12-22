function [obj, probe] = RAAR(expt, recon, probe)
% version 0: 11/12/2023.
% An implementation of the Relaxed Average Alternating Reflections
% ptychographic algorithm
%
% *** INPUTS ***
%
% expt: a structure containing the experimental parameters and data,
% with the following fields
%
% expt.dps              - the recorded diffraction intensities, held in an
%                         M x N x D array, where each of the D diffraction
%                         patterns has M x N pixels
% expt.positions.x(.y)  - the x/y scan grid positions recorded from the
%                         translation stage, in metres
% expt.wavelength       - the beam wavelength in metres
% expt.cameraPixelPitch - the pixel spacing of the detector
% expt.cameraLength     - the geometric magnification at the front face of
%                         the sample
%
% recon: a structure containing the reconstruction parameters, with the
% following fields
%
% recon.iters          - the number of iterations to carry out
% recon.gpu            - a flag indicating whether to transfer processing
%                        to a suitable CUDA-enabled graphics card
% recon.beta           - the RAAR tuning parameter (0.6 - 0.95. 1.0 = DM)
% recon.upLimit        - the maximum amplitude of the object - pixels above
%                        this value will be clipped
%
% probe: an initial model of the probe wavefront
%
% *** OUTPUTS ***
%
% obj: the reconstructed object
%
% probe: the reconstructed probe
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% Citations for this algorithm:                                           %
% Andrew. M. Maiden, Wenjie Mei and Peng Li,                              %
% "WASP: Weighted Average of Sequential Projections for ptychographic     %
% phase retrieval,"                                                       %
% XXX, pp. XX-XX (2024).                                                  %                                                 %
%                                                                         %
% Stefano Marchesini et al, "Augmented projections for ptychographic      %  
% imaging,"                                                               %
% Inverse Problems 29, 115009 (2013).                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pre-processing steps

% shift the positions to positive values
expt.positions.x = expt.positions.x - min(expt.positions.x,[],'all');
expt.positions.y = expt.positions.y - min(expt.positions.y,[],'all');

% compute pixel pitch in the sample plane
M   = size(expt.dps,1);
N   = size(expt.dps,2);
J   = size(expt.dps,3);

dx  = expt.wavelength*expt.cameraLength./...
    ([M,N]*expt.cameraPixelPitch);

% convert positions to top left (tl) and bottom right (br)
% pixel locations for each sample position
tlY = round(expt.positions.y/dx(1))+1;
tlX = round(expt.positions.x/dx(2))+1;
brY = tlY + M - 1;
brX = tlX + N - 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% variable initialisations

% initialise the "object" as free-space
obj = ones([max(brY,[],'all'),max(brX,[],'all')]);

% find suitable probe power from the brightest diffraction pattern
[~,b] = max(sum(expt.dps,[1,2]));
probePower = sum(expt.dps(:,:,b),'all');

% correct the initial probe's power
probe = probe*sqrt(probePower/(numel(probe)*sum(abs(probe(:)).^2)));

% initialise exit waves (the initial object is all ones, so the initial 
% exit waves are all just 1*probe)
EWs = zeros(M,N,J) + probe;

% pre-square-root and pre-fftshift the diffraction patterns (for speed)
expt.dps = fftshift(fftshift(realsqrt(expt.dps),1),2);

% zero-division constant
c = 1e-10;

% simple display
imH = imagesc(angle(obj));
axis image;
colormap gray;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load variables onto gpu if required
if recon.gpu
    obj      = gpuArray(obj);
    probe    = gpuArray(probe);
    expt.dps = gpuArray(expt.dps);
    EWs      = gpuArray(EWs);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 1:recon.iters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % exit wave update loop
    for j = 1:J

        % calculate the jth exit wave
        tempEW = probe.*obj(tlY(j):brY(j),tlX(j):brX(j));

        % update exit wave to conform with diffraction data
        revisedEW = ifft2(expt.dps(:,:,j).*sign(fft2(2*tempEW - EWs(:,:,j))));

        % calculate second relaxed reflection
        tempEW = 2*recon.beta*revisedEW + (1-2*recon.beta)*tempEW;

        % averaging step
        EWs(:,:,j) = 0.5*(tempEW + EWs(:,:,j));

    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Probe update
    numP = 0*probe; denP = 0*probe;
    absO2 = abs(obj).^2;
    conjO = conj(obj);

    for j = 1:J
        numP = numP + conjO(tlY(j):brY(j),tlX(j):brX(j)).*EWs(:,:,j);
        denP = denP + absO2(tlY(j):brY(j),tlX(j):brX(j));
    end

    probe = numP./(denP + c);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Object update (works best if the probe is calculated first)
    numO = 0*obj; denO = 0*obj;
    absP2 = abs(probe).^2;
    conjP = conj(probe);

    for j = 1:J
        numO(tlY(j):brY(j),tlX(j):brX(j))...
            = numO(tlY(j):brY(j),tlX(j):brX(j)) + conjP.*EWs(:,:,j);
        denO(tlY(j):brY(j),tlX(j):brX(j))...
            = denO(tlY(j):brY(j),tlX(j):brX(j)) + absP2;
    end

    obj = numO./(denO + c);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Apply additional constraints:

    % limit hot pixels
    tooHigh      = abs(obj) > recon.upLimit;
    obj(tooHigh) = recon.upLimit*sign(obj(tooHigh));

    % recentre probe/object using probe intensity centre of mass
    cp = ...
        fix([M,N]/2 - [M,N].*[mean(cumsum(sum(absP2,2))), mean((cumsum(sum(absP2,1))))]/sum(absP2,'all') + 1);

    if any(cp)
        probe = circshift(probe,-cp);
        obj   = circshift(obj,-cp);
        EWs   = circshift(EWs,[-cp 0]);
    end

    % update display
    set(imH,'cdata',gather(angle(obj)));
    drawnow();

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% format probe and obj for return

probe = gather(probe);
obj   = gather(obj);

end