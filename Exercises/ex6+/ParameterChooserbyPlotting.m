close all;
clear;
clc;
% testing all possible pairs for C & sigma
choices = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% Train the SVM
for i = 1:8
    C = choices(i);
    for j = 1:8
        sigma = choices(j);
        figure(i);
        subplot(4,2, j)
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        visualizeBoundary(X, y, model);
        title(sprintf('C = %f \n  sigma =  %f \n', C, sigma))
    end
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end