function [h_of_x]=sigmoid(z)
h_of_x=1.0./(1.0+exp(-z));
end