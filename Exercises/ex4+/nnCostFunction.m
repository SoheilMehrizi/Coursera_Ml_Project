function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% adding the bias Column to the X Matrix, its gonna be a 5000 by input layersize dim
% Matrix

X = [ones(size(X, 1), 1), X];

% First layer Computation

z2 = X * Theta1';

% Coputation the activation Value of the layer : a2 is a inputlayersize *
% hiddenlayersize dimentional matrix
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];

% Second layer Computation
z3 = a2 * Theta2';

% Compute the activation Value/hypotheses of the output layer
% h is gonna be a inputlayer size * Output layerSize Matrix
a3 = sigmoid(z3);
h = a3;


% Compute the CostFunction of the neural network

y = y == 1:num_labels;

J = -y .* log(h) - (1 .- y) .* log(1 .- h);
 
% Sum of all the values ​​in the matrix columns : m by numlabels dim Matrix
% then Sum of all Values in the Vector : m dim Vector
J = (1/m) * sum(sum(J, 2));


% -------------------------------------------------------------
% Compute the regularization Part of the CostFunction

reg = (lambda/(2*m)) * ...
       (sum(sum(Theta1(:, 2:end).^2, 2), 1) + ...
       sum(sum(Theta2(:, 2:end).^2, 2), 1));
% Adding the Regularization Part to the CostFunction
J = J + reg;
% =========================================================================

% execute step 1-4 for each training example
for i = 1:m
    % Step1 : initializing the input layer values a_1
    % then Compute the activations(z2, a2, z3, a3) for layers 2 and 3
    a1 = X(i, :)'; z2 = Theta1 * a1; a2 = sigmoid(z2);
    a2 = [1; a2]; z3 = Theta2 * a2; a3 = sigmoid(z3);
    % Compute the error of the output layer
    delta_3 = (a3 .- y(i, :)');
    % Compute the error of the hidden layer l = 2
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2]);
    delta_2 = delta_2(2:end);
    % Accumulate the gradient from this training example 
    % Propagate error backwards
    Theta2_grad = Theta2_grad + delta_3 * a2';
    Theta1_grad = Theta1_grad + delta_2 * a1';
end
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;
% Add regularization terms
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
