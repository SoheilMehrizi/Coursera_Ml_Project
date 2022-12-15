function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 1/m;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Compute the CostFunction
% Compute the z =  theta' * x
z = X * theta;

% Compute h of theta(sigmoid Function of z)

h = sigmoid(z);

% Compute the CostFunction J of Theta.

J = (1/m) * sum(-y .* log(h) - (1 .- y) .* log(1 .- h));

% Compute the Partial derivatives of the CostFunction(Gradient Descent.

grad = (1/m) .* sum((h - y) .* X)';







% =============================================================

end
