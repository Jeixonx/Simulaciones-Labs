function modelo = entrenarGMM(xTrainC,NumeroMezclas)

    inputDim=size(xTrainC,2);      %%%%% Numero de caracteristicas de las muestras
    mix = gmm(inputDim, NumeroMezclas, 'full');
    options = foptions; 
    options(14) = 100; % Número de iteraciones que vamos a realizar
    options(1) = 1; % Mostrar los errores
    mix = gmminit(mix, xTrainC, options);
    mix = gmmem(mix, xTrainC, options);
    modelo = mix;
end

% 'spherical' = single variance parameter for each component: stored as a vector
%  	  'diag' = diagonal matrix for each component: stored as rows of a matrix
%  	  'full' = full matrix for each component: stored as 3d array
%  	  'ppca'