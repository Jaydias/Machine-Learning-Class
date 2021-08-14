%Jay Dias 2/8/
%homework 1 problem 3
clear all

dataHW1 = load('dataset_homework1.txt');
x = dataHW1(:, 1:2); % our features
y = dataHW1(:, 3); % Price
theta = zeros(3,1); %theta space alocation and initilization with zeros 
a = 0.008; %alpha
iterations = 5000;

xnormal = x; %used in problem 4, this stores the values of the data set so that we have an unalterd version

%normilization of sq ft feature

mu = zeros(1, size(x,2));% creates the matrix to hold mu values
stdev = zeros(1, size(x,2));% creates a matrix to hold sigma values

for i = 1:size(mu,2)
    stdev(1,i) = std(x(:,i));
    mu(1,i) = mean(x(:,i));
    xn(:,i) = ((x(:,i)- mu(1,i))/stdev(1,i));
end

%x = xn;
x = xn
%mu;    %debug check
%stdev; %debug check

m =47; %number of training sets we are given

x = [ones(m,1)  x];

Cost = zeros(iterations, 1);
thetaL = length(theta);
j = theta;

%gradient descent

for itr=1:iterations
    h = (x*theta - y);
    for i = 1:3
        %j(1,1) = sum(h);
        %j(2,1) = sum(h.*x(:,i));
        j(i,1) = sum(h.*x(:,i));
        
    end
    theta; %debugging variable check
    theta = theta - (a/m)*j;
    
    
    Cost(itr,1) = (1/(2*m))*sum((x*theta -y).^2);
    
    %Cost(itr,1) = cost;
end
%theta %debugging variables
%Cost  %debugging variables
t1 = theta(1,1);
t2 = theta(2,1);
t3 = theta(3,1);

figure
plot(Cost)
xlabel('iterations')
ylabel('J(theta0,theta1,theta2)')
%3.2
%We now use our hypothesis to predict the price of a 1755 ft^2 house with 3
%bedrooms.
%First we need to normalize out new feature point aginst the old features
%to do this we will simply put it in the tain end of the mu and stndv data
%sets

newx = [1755 , 3];
newx = ((newx - mu)./(stdev));
newx = [ones(1,1)  newx]
newx1 = newx(1,1);
newx2 = newx(1,2);
newx3 = newx(1,3);

pricepredict = t1*newx1 + t2*newx2 + t3*newx3

%% problem 4

% we now wish to calculate the thetas from our data set using the normal
% equations which has the form $\theta = (X^{T}X)^{-1}X^{T}y$

thetanormal = zeros(3,1); %theta space alocation and initilization with zeros 
xnormal = [ones(m,1)  xnormal]; %adds the row of 1s needed for the matrix opperations and theta 0

%Since we do not have a square matrix xnormal then we must use psudo
%inverse here. The user may change if they have an invertable matrix to
%inverse().
thetanormal = pinv(xnormal' * xnormal)* xnormal' * y;

%now we store temp values for thetanormal1,2,3 so we can predict price.

tn1 = thetanormal(1,1);
tn2 = thetanormal(2,1);
tn3 = thetanormal(3,1);

nxn = [1,1755,3];

pricenormal = tn1*1 + tn2*1755 + tn3*3

%We see that our price from the normal equation is not different from our
%previous prediction. 


