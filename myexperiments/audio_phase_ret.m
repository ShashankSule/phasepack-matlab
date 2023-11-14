%% scrape from test_audio 
t = readtable('test_audioMNIST.csv'); 
alg = 'WirtFlow'; 

%% Set options for PhasePack - this is where we choose the recovery algorithm

opts = struct;                  % Create an empty struct to store options
opts.algorithm = alg;           % Use the WF method to solve the retrieval problem.  Try changing this to 'twf' for truncated Wirtinger flow.
opts.initMethod = 'optimal';    % Use the optimal spectral initializer method to generate an initial starting point for the solver  
opts.tol = 1e-6;                % The tolerance - make this smaller for more accurate solutions, or larger for faster runtimes
opts.verbose = 0;               % Print out lots of information as the solver runs (set this to 1 or 0 for less output)
opts.isComplex = false;
opts.maxIters = 1000; 
A_spar = load('A_sparse.mat'); 
A_spar = A_spar.A_spar; 
[m, n] = size(A_spar); 
%% get audio filename 
% i = 1; 
parfor i = 1:size(t,1)
    %% get audio names and filenames 
    c = strsplit(t(i,4).Var4{1}, ','); 
    filename = c{1}; 
    d = strsplit(filename, '.'); 
    audioname = d{1}; 
    %% get data subsampled 
    [x_true,Fs] = audioread(filename); 
    x_sub = resample(x_true,8000,Fs); 
    x_pad = [x_sub; zeros(8000-size(x_sub,1),1)]; 
    
    %% compute spectrograam 
    b = abs(A_spar*x_pad);
    
    %% Run the Phase retrieval Algorithm
    % fprintf('Running %s algorithm\n',opts.algorithm);
    [x, outs] = solvePhaseRetrieval(A_spar, [], b, n, opts);
    
    %% Determine the optimal phase rotation so that the recovered solution
    %  matches the true solution as well as possible.  
    alpha = (x'*x_pad)/(x'*x);
    x = alpha * x;
    
    %% save to file 
    savename = strcat("reconstructions_8K/",alg,'/', audioname, '_', alg); 
    parsave(savename,x); 

end

%% helpers 
function parsave(fname, x)
  save(fname, 'x', '-mat')
end