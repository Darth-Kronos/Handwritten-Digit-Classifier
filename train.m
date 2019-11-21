clear all;
load('dataset.mat');
data = [X y];
[m,n] = size(data) ;
P = 0.80 ;
idx = randperm(m)  ;
Training = data(idx(1:round(P*m)),:) ; 
Testing = data(idx(round(P*m)+1:end),:) ;
correct_Output = Training(:,end);
input_image = Training(:,1:end-1);
w1 = 2*rand(25,400)-1;
w2 = 2*rand(10,25)-1;

epp = [];
for epoch = 1:10
    [w1, w2, error1] = DeepLearning(w1, w2, input_image, correct_Output);
    epp = [epp, error1];
end
stem(1:epoch,epp)
save('DeepNeuralNetwork.mat')