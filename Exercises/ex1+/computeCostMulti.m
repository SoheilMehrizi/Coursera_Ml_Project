function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%Compute the hypothis
hypothesis = X * theta;
%Compute the SquarError
squarerror = (hypothesis - y) .^ 2;
%coputing Cost Function for each iteration
J = (1/(2*m)) * sum(squarerror);


% =========================================================================

end
