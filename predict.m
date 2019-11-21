function x = predict(input_vector,w1,w2)
    input_of_hidden_layer1 = w1*input_vector';
    output_of_hidden_layer1 = ReLU(input_of_hidden_layer1);
    input_of_output_node = w2*output_of_hidden_layer1;
    final_output = Softmax(input_of_output_node);
    x = find(final_output == max(final_output));
end
