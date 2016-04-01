function [J,grad] = costFn(nn_params,input_layer_size, hidden_layer_size,output_layer,X,y, lambda)

m=size(X,1);
Theta1=reshape(nn_params(1:(hidden_layer_size*(input_layer_size+1))),hidden_layer_size,input_layer_size+1);
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),output_layer, (hidden_layer_size + 1));
a2=zeros(hidden_layer_size,1);
a3=zeros(output_layer,1);
z2=zeros(hidden_layer_size,1);
z3=zeros(output_layer,1);
op1=zeros(size(X,1),10);
op2=zeros(size(X,1),10);
y3=zeros(1,10);
y2=zeros(hidden_layer_size+1,1);
Y2=zeros(size(Theta2));
Y1=zeros(size(Theta1));
for l=1:size(X,1)
    if(y(l,1)~=0)
    op1(l,y(l,1))=1;
    else
        op1(l,10)=1;
    end
end

for k=1:size(X,1)
    z2= ([1 X(k,:)]*Theta1')';
    a2=sigmoid(z2);
    z3=Theta2*[1; a2];
    a3=sigmoid(z3);
    op2(k,:)=a3';
    y3=op2(k,:)-op1(k,:);
    y2=(Theta2(:,2:end)'*y3').*(a2.*(1-a2));
    Y2=y3'*[1; a2]';
    Y1=y2*[1 X(k,:)];
end
J=(-1/size(X,1))*sum(sum(op1.*log(op2)+(1-op1).*log(1-op2)));
J=J+(lambda/(2*size(X,1)))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

Y2=Y2/(size(X,1));
Y1=Y1/(size(X,1));

Y2(:,2:end)=Y2(:,2:end)+((lambda/m)*(Theta2(:,2:end)));
Y1(:,2:end)=Y1(:,2:end)+((lambda/m)*(Theta1(:,2:end)));

grad=[Y1(:); Y2(:)];

end