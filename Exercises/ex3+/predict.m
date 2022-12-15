function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

% Theta1 is a 25 * 401 Matrix and Theta2 is a 10*26 dim Matrix

% adding bias column to the Matrix X, its gonna be a 5000 * 401 dim Matrix

X = [ones(size(X, 1), 1), X];
% calculate the z_2 the Value of the first Hidden Layer
z2 = X * Theta1';
% Calculate the a_2 The Inpute of the Second Hidden Layer ,
% a_2 is a 5000*26 dimentional matrix;
a2 = sigmoid(z2);
% adding bias column to the a2
a2 = [ones(m, 1), a2];
% Calculate the z_3 for Second Hidden Layer.
% its gonna be a 5000*26 dimentional Matrix
z3 = a2 * Theta2';
% Calculate the primary value of the output layer a_3 its a 5000*10d Matrix 
a3 = sigmoid(z3);
% Calculate the pridiction P , its gonna be a 5000*1 dimentional vector 
% p containes the Labels of predictions
% obtain the Labels of each dataset exaples Using max functionI : index of
% Max Probability for examples training dateset
% Return the prediction Vector
[probability, p]= max(a3, [], 2);


% =========================================================================


end
