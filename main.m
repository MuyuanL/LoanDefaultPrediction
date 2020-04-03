% Import data
importcsv;

% Set NaN to 0 for now
X(isnan(X)) = 0;

% Remove all-0 columns
X(:, max(X) == 0) = [];

% Size of X
[n, d] = size(X);

% Normalize X to the range of [0,1]
X = X ./ repmat(max(X), n, 1);

delcol = [];
for i = 1:d
    if length(unique(X(:,i))) <= 1
        delcol = [delcol; i];
    end
end
X(:, delcol) = [];
[n, d] = size(X);


% Set initial w to be the best linear fit 
w = X \ Y;
obj = 0.5 * sum((max(X*w, 0) - Y).^2) % Compare this with next line
naive_obj = 0.5 * sum((0 - Y).^2)

% Gradient descent
step = 1e-7;  
for iter = 1:10
    grad = X' * (max(X*w, 0) - Y);          % Calculate new gradient
    w = w - step * grad;                    % Update w
    obj = 0.5 * sum((max(X*w, 0) - Y).^2)   % Print current objective value
end

cvx_begin
    variable w1(n)
    minimize( norm(max(X*w,0)-Y) )
cvx_end