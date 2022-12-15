function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;


% initializing the vactor x with dictionary indexes
x = 1:n;
% make sure that x and word_indices are vector
x = x(:);
word_indices = word_indices(:);
% remove the repetitive elements in word_indices vector
[yvalue, ~, subs] = unique(word_indices);
% check wich elements of x exist in word_indices  
[x, where] = ismember(x, yvalue);



% =========================================================================
    

end
