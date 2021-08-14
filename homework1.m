%%Jay Dias 2/8/2018
%% Homework 1 Machine Leraning
%%problem2
% we wish to import the data set and plot it using the plot function in
% matlab.

clear all

dataHW1 = load('dataset_homework1.txt');
x = dataHW1(:,1); 
y = dataHW1(:, 3); 
m = 47; 
theta = zeros(2,1); 
a = 0.05; 
iterations = 1; 

figure
scatter(x,y)
ylabel('Price (USD)')
xlabel('ft^2')

%first lets normalize our features!
%xn = x;
mu = zeros(1, size(x,1));% creates the matrix to hold mu values
stdev = zeros(1, size(x,1));% creates a matrix to hold sigma values

for i = 1:size(mu,1)
    stdev(1,i) = std(x(:,i));
    mu(1,i) = mean(x(:,i));
    xn(:,i) = ((x(:,i)- mu(1,i))/stdev(1,i));
end

%x = xn;
x = xn;
mu;
stdev;
%to calculate the cost function we must add a column of 1s to our features
x = [ones(m,1)  x];

Cost = zeros(iterations, 1);
j = theta;

%gradient desent algorithim

for itr=1:iterations
    h = (x*theta - y);
    for i = 1:2
        j (i,1) = sum(h.*x(:,i));
    end
    theta = theta - (a/m)*j;
    
    cost = (1/(2*m))*sum((x*theta-y).^2);
    
    Cost(itr,1) = cost;
end

theta;
Cost




%%Homework problem 2.3

theta = zeros(2,1); 
a = 0.05; %alpha
iterations = 50; 



Cost = zeros(iterations, 1);
j = theta; %dumy variable for manipulations of (h(x)*x)

for itr=1:iterations
    h = (x*theta - y); %stores $(h_{\theta}*x^{i} - y^{i})$
    for i = 1:2
        %j(1,1) = sum(h);         %brute force method
        %j(2,1) = sum(h.*x(:,i)); %brute force method
        j(i,1) = sum(h.*x(:,i)); %Creats new theta values
        
    end
    %theta; %debugging var to check your thetas are correct thought out the nested loop
    thetahist1(itr,1) = theta(1,1); %keeps a history of the theta1 values thoughout the loop
    thetahist2(itr,1) = theta(2,1); %keeps a history of the theta2 values thoughout the loop
    
    theta = theta - (a/m)*j; %simultanius update for thetas
    
    
    Cost(itr,1) = (1/(2*m))*sum((x*theta -y).^2);
    
   
end
%theta  %debugging variables
%Cost   %debugging variables
t1 = theta(1,1)
t2 = theta(2,1) 
figure
plot(Cost)
xlabel('iterations')
ylabel('J(theta0,theta1)')

%2.5
%we now wish to plot our hypothesis on the same graph of the model
%paramaters to see how well we hit the points
xp = dataHW1(:,1); % one feature in this case is sq ft column

figure
scatter(xp,y)
ylabel('Price (USD)')
xlabel('ft^2')
hold on

hyp = (t1 + xn*t2);
plot(xp,hyp)
hold off

%2.5
%we now use our hypothesis to predict the price of a 1755 ft^2 house
%first we need to normalize out new feature point aginst the old features
%to do this we will simply put it in the tain end of the mu and stndv data
%sets

newx = (1755);
newx = ((newx - mu)./(stdev));
newx = [ones(1,1)  newx];
newx1 = newx(1,1);
newx2 = newx(1,2);

pricepredict = t1*newx1 + t2*newx2

%2.6
%we wish to plot a 3-D plot of the cost vs theta
Z = [thetahist1,thetahist2,Cost];
figure
surfc(Z)
ylabel('theta1')
xlabel('theta0')
zlabel('J(theta0,theta1)')
