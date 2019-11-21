function [w1, w2, error1] = DeepLearning(w1, w2, input_Image, correct_Output)
    alpha = 0.01; %to control the learning rate
    N = length(input_Image);
    
    for k = 1:N
        %reshaped_input_Image = reshape(input_Image(:,:,k), 25, 1);
        input_Image1 = input_Image(k,:);
        input_of_hidden_layer1 = w1*input_Image1';
        output_of_hidden_layer1 = ReLU(input_of_hidden_layer1);
        input_of_output_node = w2* output_of_hidden_layer1;
        final_output = Softmax(input_of_output_node);
        yd = labelToMatrix(correct_Output(k))';
        correct_Output_transpose = yd;
        
        error = correct_Output_transpose - final_output;
        delta1 = error;
        error1 = (error'*error)/10;
        error_of_hidden_layer1 = w2'*delta1;
        delta2 = (input_of_hidden_layer1>0).*error_of_hidden_layer1;

        adjustment_of_w2 = alpha*delta1*output_of_hidden_layer1';
        adjustment_of_w1 = alpha*delta2*input_Image1;

        w1 = w1+ adjustment_of_w1;
        w2 = w2+ adjustment_of_w2;
    end
end