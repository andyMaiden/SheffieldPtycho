function [obj, probe] = HIVE(expt, recon, probe)
% version 0: 11/12/2022.
% An implementation of the Weighted Average Stochastic Projections
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
% recon: a structure containing the reconstructino parameters, with the
% following fields
%
% recon.iters          - the number of iterations to carry out
% recon.gpu            - a flag indicating whether to transfer processing
%                        to a suitable CUDA-enabled graphics card
% recon.alpha          - the object step size parameter (~2)
% recon.beta           - the probe step size parameter (~1)
% recon.upLimit        - the maximum amplitude of the object - pixels above
%                        this value will be clipped
% recon.numWorkers     - the number of worker nodes to distribute over
% recon.subIters       - the number of sub-iterations run by each worker
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
% Citation for this algorithm:                                            %
% Andrew. M. Maiden, Wenjie Mei and Peng Li,                              %
% "WASP: Weighted Average of Sequential Projections for ptychographic     %
% phase retrieval,"                                                       %
% XXX, pp. XX-XX (2024).                                                  %                                                 
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pre-processing steps

% shift the positions to positive values
expt.positions.x = expt.positions.x - min(expt.positions.x,[],'all');
expt.positions.y = expt.positions.y - min(expt.positions.y,[],'all');

% compute pixel pitch in the sample plane
M   = size(expt.dps,1);
N   = size(expt.dps,2);
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

% find a suitable probe power from the brightest diffraction pattern
[~,b] = max(sum(expt.dps,[1,2]));
probePower = sum(expt.dps(:,:,b),'all');

% correct the initial probe's power
probe = probe*sqrt(probePower/(numel(probe)*sum(abs(probe(:)).^2)));

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
    obj      = gpuArray(single(obj));
    probe    = gpuArray(single(probe));
    expt.dps = gpuArray(single(expt.dps));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This parallel implementation of WASP uses the MATLAB 'Single Program
% Multiple Data' (spmd) construct available within the MATLAB Parallel
% Computing toolbox.

% determine the number of patterns per worker (some patterns may duplicate
% onto two workers)
recon.DPsPerWorker = ceil(size(expt.dps,3)/recon.numWorkers);

% stop any current parallel pool and start a new one
delete(gcp);
parpool(recon.numWorkers);

% randomly shuffle the data and allocate portions to the worker WASPs
shuffled = randperm(size(expt.dps,3));

for l = 1:recon.numWorkers
    dpList(l,:) = shuffled(1:recon.DPsPerWorker);
    shuffled    = circshift(shuffled,-recon.DPsPerWorker);
end

% load relevant data and variables to the workers
spmd
    dps    = expt.dps(:,:,dpList(spmdIndex,:));
    tlXs   = tlX(dpList(spmdIndex,:));
    tlYs   = tlY(dpList(spmdIndex,:));
    brXs   = brX(dpList(spmdIndex,:));
    brYs   = brY(dpList(spmdIndex,:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 1:recon.iters

    % collect partial numerators and denominators from each worker
    spmd
        [numP,denP,numO,denO] = WASPWorker(dps,obj,probe,tlXs,tlYs,brXs,brYs,recon);
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % weighted average update of object and probe
    obj         = sum(cat(3,numO{:}),3)./(sum(cat(3,denO{:}),3) + c);
    probe       = sum(cat(3,numP{:}),3)./(sum(cat(3,denP{:}),3) + c);

    % Apply additional constraints:

    % limit hot pixels
    tooHigh      = abs(obj) > recon.upLimit;
    obj(tooHigh) = recon.upLimit*sign(obj(tooHigh));

    % recentre probe/object using probe intensity centre of mass
    absP2 = abs(probe).^2;
    cp = ...
        fix([M,N]/2 - [M,N].*[mean(cumsum(sum(absP2,2))), mean((cumsum(sum(absP2,1))))]/sum(absP2,'all') + 1);

    if any(cp)
        probe = circshift(probe,-cp);
        obj   = circshift(obj,-cp);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           THE WORKER WASP                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [numP,denP,numO,denO] = WASPWorker(dps,obj,probe,tlX,tlY,brX,brY,recon)

% initialise worker numerator and denominator sums
numP = 0*probe; denP = 0*probe; numO = 0*obj; denO = 0*obj;

for kk = 1:recon.subIters

    % randomise the diffraction pattern order for sequential projections
    shuffleOrder = randperm(size(expt.dps,3));

    for j = shuffleOrder

        % update exit wave to conform with diffraction data
        objBox    = obj(tlY(j):brY(j),tlX(j):brX(j));
        currentEW = probe.*objBox;
        revisedEW = ifft2(dps(:,:,j).*sign(fft2(currentEW)));

        % sequential projection update of object and probe
        obj(tlY(j):brY(j),tlX(j):brX(j)) = objBox + ...
            conj(probe).*(revisedEW - currentEW)./(abs(probe).^2 + recon.alpha*mean(abs(probe).^2,'all'));

        probe = probe + conj(objBox).*(revisedEW - currentEW)./(abs(objBox).^2 + recon.beta);

        % update numerator and denominator sums
        numO(tlY(j):brY(j),tlX(j):brX(j))...
             = numO(tlY(j):brY(j),tlX(j):brX(j)) + conj(probe).*revisedEW;
        denO(tlY(j):brY(j),tlX(j):brX(j))...
             = denO(tlY(j):brY(j),tlX(j):brX(j)) + abs(probe).^2;
        numP = numP + conj(objBox).*revisedEW;
        denP = denP + abs(objBox).^2;

    end

end

end
