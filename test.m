clear all;
load('DeepNeuralNetwork.mat');
correct_Output = Testing(:,end);
c=0;
input_image = Testing(:,1:end-1);
for index=1:1000
    p=input_image(index,:);
    x = predict(p,w1,w2);
    if(x==correct_Output(index))
        c=c+1;
    end
end
acc = c/10