function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
C = 1; sigma = 0.03;
% You need to return the following variables correctly.
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
   C_list     = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
   sigma_list = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% =========================================================================
% Check the all possibility pairs of C and Sigma
% initializing the error vector
predition_errors  = zeros(length(C_list), length(sigma_list));
for i = 1:length(C_list)
    % go through C_list
    % assign the C accourding to the Current position
    C_test = C_list(i);
    for j = 1:length(sigma_list)
        % go through sigma_list
        % assign the sigma accourding to the position
        sigma_test = sigma_list(j);
        % traint model using  SVM on 3rd DataSet
        model = svmTrain(X, y, C_test, ...
                @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        % Initializing the pridiction vector
        predictions = zeros(length(yval), 1);
        % Compute the Predictions by trained model using svmpredict
        predictions = svmPredict(model, Xval);
        % Compute the error on C and sigma current value
        prediction_errors(i, j) = mean(double(predictions ~= yval));
    end
end
   % Finding row and col corresponding to min(prediction_error)
   [values, row_index]=min(prediction_errors);
   [~ ,col] = min(values);
   row = row_index(col);
   
   % C and sigma corresponding to min(prediction_error)
   C = C_list(row);
   sigma = sigma_list(col);

end
