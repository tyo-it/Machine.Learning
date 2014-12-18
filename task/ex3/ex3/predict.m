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
%

% Add ones to the input layer (a_1) matrix
a_1 = [ones(m, 1) X];

% computes activation of hiden layer (a_2)
z_2 = Theta1*a_1'; 
a_2 = sigmoid(z_2);

% Add ones to the hiden layer (a_2) matrix
a_2 = [ones(1, size(a_2, 2)); a_2];

% computes result of output layer (h_theta)
z_3 = Theta2*a_2;
h_theta = sigmoid(z_3);

% get index of maximum value in every column
[i, j] = max(h_theta);

% assign index of maximum value (vector j) to p
     p = j(:);
% =========================================================================

end
