%%Homework problem 1.2.3
clear all


dataHW1 = load('dataset_homework1.txt');
x = dataHW1(:,1); % one feature in this case is sq ft column
y = dataHW1(:, 3); % Housing Prices
m = length(y); % Number of training examples
q = 1; % Number of features.
theta = zeros(q+1,1); % Initialize thetas to zero.
alpha = 0.03; % Learning rate
ItrN = 10; % How long gradient descent should run for

%first lets normalize our features!
xn=x;
mu = zeros(1, size(x,q));% creates the matrix to hold mu values
stdev = zeros(1, size(x,q));% creates a matrix to hold sigma values

for i = 1: size(stdev,q)
    stdev(1,i) = std(x(:,i));
    mu(i) = mean(xn(i));
    xn(i) = ((xn(i)- mu(i))/stdev(i));
end

x = [ones(m,1) x];

Cost = zeros(ItrN, 1);
thetaL = length(theta);
j = theta;
a = alpha;

for itr=1:ItrN
    h = (x*theta - y);
    for i = 1:thetaL
        j (i,1) = sum(h.*x(:,i));
    end
    theta = theta - (a/m)*j;
    
    cost = (1/(2*m))*sum((x*theta -y).^2);
    
    Cost(itr,1) = cost;
end

theta
Cost

plot(Cost,itr)