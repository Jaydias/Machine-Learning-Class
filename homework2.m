%Jay Dias 3/2/18
%Homework 2

clear all

%% 2.1 we are asked to import data and plot the data set
dataHW2 = load('dataset_homework2.txt');

x = dataHW2(:, 1:2); % first test score, second test score
y = dataHW2(:, 3); % admission decision

m = size(y)
a = 0.05;

%we wish to plot our data set with the x or o denoting weather the student
%was admitted or not where 1 is admitted and 0 is not admitted

figure;
gscatter(x(:,1),x(:,2),y,'br','xo')
xlabel('Test 1 Score');
ylabel('Test 2 Score');

%% 2.2 we are asked to write a sigmoid function subroutine and calculate the sigmoid for a given matrix M
%logistic regression hypothesis and sigmoid function
M = [0,1 ; -200,100];

fprintf('The result of g(M) is the following :')
sigmoid(M)

%% 2.3 we are asked to write a subroutine for the cost function and gradient for logistic regression.
%cost function problem 2.3

theta  = zeros(3,1);
x = [ones(100,1)  x];
fprintf('the initial values for the cost function are the following')
[J,grad] = costFunction(theta,x,y);
fprintf('The cost function and gradient values are the following:')
J
grad


%% problem 2.4 asks us to optomize thetas for the cost function

% Set options for fminunc
theta  = zeros(3,1);
options = optimset('GradObj', 'on', 'MaxIter', 400);
initial_theta = theta;
% Run fminunc to obtain the optimal theta
% This function will return theta and the cost
[theta, cost] = ...
 fminunc(@(theta)(costFunction(theta, x, y)), initial_theta, options);

fprintf('The cost function, gradient, and optimal theta are as follow')
theta
[J,grad] = costFunction(theta, x, y);
J
grad

figure;
gscatter(x(:,2),x(:,3),y,'br','xo')
xlabel('Test 1 Score');
ylabel('Test 2 Score');
hold on

t1 = theta(1,1);
t2 = theta(2,1);
t3 = theta(3,1);
x1 = (x(:,2));
x2 = (x(:,3));

hyp = (t1 + t2.*x1)./(-t3) ; 
title('Decision Boundary')
plot(x1,hyp)
hold off

%% 2.5 
%we wish to predict the admission probability if the first exam score is 45
%and the seccond is 85.

xp = [1, 45, 85];
prob = 100*sigmoid(xp*theta);

print = ['Probability of admission for this student is ',num2str(prob),'%'];
display(print)

%% 2.6 we wish to test the accuracy of our model using the training data. 

prb = zeros(1,100);
tempn = 0;
for i=1:100
    prb(i,1) = 100*sigmoid(x(i,:)*theta);
    if prb(i,1) >= 50
        prb(i,1) = 1;
    else
        prb(i,1) = 0;
    end
    
    if prb(i,1) == y(i,1)
        tempn = tempn +1;
    end
  
end

accuracy = 100*(tempn/100);

print2 = ['The accuracy of model prediction is ', num2str(accuracy),'%'];
display(print2)
