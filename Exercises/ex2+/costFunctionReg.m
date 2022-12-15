function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute the CostFunction
% Compute the z =  theta' * x
z = X * theta;

% Compute hypothesis of theta(sigmoid Function of z)
h = sigmoid(z);

% Compute the Regularization Part of the logestic regression CostFunction
theta(1, 1) = 0;


% Compute the CostFunction J of Theta for regularized logistic regression.
  
J = (1/m) * sum((-y)' * log(h) - (1 .- y)' * log(1 - h)) + ...
    (lambda / (2 * m)) * (theta' * theta);
% Compute the Partial derivatives of the CostFunction(Gradient Descent) for regularized logistic regression.


grad = (1/m) .* (X' * (h - y)) + (lambda / m) .* theta ;

% =============================================================

end
