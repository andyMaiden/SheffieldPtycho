function [obj, probe] = ePIE(expt, recon, probe)
% version 0: 11/12/2023. 
% Please refer to the end of the code for licencing information.
%
% An implementation of the regularised Ptychographical Iterative Engine
% algorithm
%
% *** INPUTS ***
%
% expt: a structure containing the experimental parameters and data,
% having the following fields
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
% recon: a structure containing the reconstruction parameters, having the
% following fields
%
% recon.iters          - the number of iterations to carry out
% recon.gpu            - a flag indicating whether to transfer processing
%                        to a suitable CUDA-enabled graphics card
% recon.alpha          - the object step size parameter (~0.25. 1.0 for ePIE)
% recon.beta           - the probe step size parameter (~1. 1.0 for ePIE)
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
% A. M. Maiden and J.M. Rodenburg "An improved ptychographical phase      %
% retrieval algorithm for diffractive imaging,"                           %
% Ultramicroscopy 109 (10), pp. 1256-1262                                 %
%                                                                         %
% A. M. Maiden, D. Johnson and P. Li "Further improvements to the         %
% ptychographical iterative engine,"                                      %
% Optica 4 (7), pp. 736-745                                               %
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

% find suitable probe power from the brightest diffraction pattern
[~,b] = max(sum(expt.dps,[1,2]));
probePower = sum(expt.dps(:,:,b),'all');

% correct the initial probe's power
probe = probe*sqrt(probePower/(numel(probe)*sum(abs(probe(:)).^2)));

% pre-square-root and pre-fftshift the diffraction patterns (for speed)
expt.dps = fftshift(fftshift(realsqrt(expt.dps),1),2);

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

for k = 1:recon.iters

    shuffleOrder = randperm(size(expt.dps,3));

    for j = shuffleOrder

        % update exit wave to conform with diffraction data
        objBox    = obj(tlY(j):brY(j),tlX(j):brX(j));
        EW        = probe.*objBox;
        revisedEW = ifft2(expt.dps(:,:,j).*sign(fft2(EW)));

        % sequential projection update of object and probe
        obj(tlY(j):brY(j),tlX(j):brX(j)) = objBox + ...
            recon.alpha*conj(probe).*(revisedEW - EW)./max(abs(probe).^2,[],'all');

        probe = probe + ...
            recon.beta*conj(objBox).*(revisedEW - EW)./max(abs(objBox).^2,[],'all');

    end

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

% Academic License Agreement
% 
% ********************************
% Please note that Phase Focus Limited, UK, holds a portfolio of international patents regarding ptychography, which you can find listed here: https://www.phasefocus.com/patents
% 
% Implementations of ptychography included within computer software fall within the scope of patents owned by Phase Focus Limited.
% 
% If you intend to pursue ANY commercial interest in technologies making use of the patents, please contact Phase Focus Limited to discuss a commercial-use licence here: https://www.phasefocus.com/licence
% ********************************
% 
% This license agreement sets forth the terms and conditions under which the authors (hereafter "LICENSOR") grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for academic, non-commercial purposes ONLY (hereafter "LICENSE") to use this ptychography computer software program and associated documentation furnished hereunder (hereafter "PROGRAM").
% 
% Terms and Conditions of the LICENSE
%  1.	LICENSOR grants to LICENSEE a royalty-free, non-exclusive license to use the PROGRAM for academic, non-commercial purposes, upon the terms and conditions hereinafter set out and until termination of this license as set forth below.
% 
%  2.	LICENSEE acknowledges that the PROGRAM is a research tool still in the development stage. The PROGRAM is provided without any related services, improvements or warranties from LICENSOR and that the LICENSE is entered into in order to enable others to utilise the PROGRAM in their academic activities. It is the LICENSEE's responsibility to ensure its proper use and the correctness of the results.
% 
%  3.	THE PROGRAM IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT OF ANY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS. IN NO EVENT SHALL THE LICENSOR, THE AUTHORS OR THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DIRECT, INDIRECT OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY ARISING FROM, OUT OF OR IN CONNECTION WITH THE PROGRAM OR THE USE OF THE PROGRAM OR OTHER DEALINGS IN THE PROGRAM.
% 
%  4.	LICENSEE agrees that it will use the PROGRAM and any modifications, improvements, or derivatives of PROGRAM that LICENSEE may create (collectively, "IMPROVEMENTS") solely for academic, non-commercial purposes and that any copy of PROGRAM or derivatives thereof shall be distributed only under the same license as PROGRAM. The terms "academic, non-commercial", as used in this Agreement, mean academic or other scholarly research which (a) is not undertaken for profit, or (b) is not intended to produce works, services, or data for commercial use, or (c) is neither conducted, nor funded, by a person or an entity engaged in the commercial use, application or exploitation of works similar to the PROGRAM.
% 
%  5.	Except for the above-mentioned acknowledgment, LICENSEE shall not use the PROGRAM title or the names or logos of LICENSOR, nor any adaptation thereof, nor the names of any of its employees or laboratories, in any advertising, promotional or sales material without prior written consent obtained from LICENSOR in each case.
% 
%  6.	Ownership of all rights, including copyright in the PROGRAM and in any material associated therewith, shall at all times remain with LICENSOR, and LICENSEE agrees to preserve same. LICENSEE agrees not to use any portion of the PROGRAM or of any IMPROVEMENTS in any machine-readable form outside the PROGRAM, nor to make any copies except for its internal use, without prior written consent of LICENSOR. LICENSEE agrees to place this licence on any such copies.
% 
%  7.	The LICENSE shall not be construed to confer any rights upon LICENSEE by implication or otherwise except as specifically set forth herein.
% 
%  8.	This Agreement shall be governed by the material laws of ENGLAND and any dispute arising out of this Agreement or use of the PROGRAM shall be brought before the courts of England and Wales. 