input_layer_size=784;
hidden_layer_size1=20;
%hidden_layer_size2=20;
output_layer=10;

load('train.mat');
m=size(X,1);
y1=X(1:5000,1);
X1=X(1:5000, 2:end);

Theta1=rand(hidden_layer_size1,input_layer_size+1).*(2*.012)-.012;
Theta2=rand(output_layer,hidden_layer_size1+1).*(2*.012)-.012;
%Theta3=rand(hidden_layer_size2,hidden_layer_size1+1).*(2*.012)-.012;

lambda=0.1;
initial_nn_params=[Theta1(:);Theta2(:)];
J=costFn(initial_nn_params,input_layer_size, hidden_layer_size1,output_layer, X1,y1, 0);
fprintf('Initial Cost, lambda=0: %f\n',J);

J=costFn(initial_nn_params,input_layer_size, hidden_layer_size1,output_layer, X1,y1, 0.1);
fprintf('Initial Cost, lambda=0.1: %f\n',J);

J=costFn(initial_nn_params,input_layer_size, hidden_layer_size1,output_layer, X1,y1, 1);
fprintf('Initial Cost, lambda=1: %f\n',J);

J=costFn(initial_nn_params,input_layer_size, hidden_layer_size1,output_layer, X1,y1, 10);
fprintf('Initial Cost, lambda=10: %f\n',J);

J=costFn(initial_nn_params,input_layer_size, hidden_layer_size1,output_layer, X1,y1, 100);
fprintf('Initial Cost, lambda=100: %f\n',J);

fprintf('\nTraining Neural Network... \n')
%costFunction = @(p) costFn(p, input_layer_size,hidden_layer_size, output_layer, X1, y1, lambda);
  options = optimset('MaxIter', 50);

[nn_params, cost] = fmincg(@(p)costFn(p, input_layer_size,hidden_layer_size1, output_layer, X1, y1, lambda), initial_nn_params, options);


Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), hidden_layer_size1, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size1 * (input_layer_size + 1))):end),output_layer, (hidden_layer_size1 + 1));
pred = predict(Theta1, Theta2, X1);

J=costFn(nn_params,input_layer_size, hidden_layer_size1,output_layer, X1,y1, lambda);
fprintf(' Cost after training, lambda=%f: %f\n',lambda,J);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y1)) * 100);

