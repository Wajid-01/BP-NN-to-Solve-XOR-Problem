clc; clear all; close all;

%% [1] Define Parameters and Variables:

% Create XOR network inputs and initialise the network weights
net_inputs = [0 0; 0 1; 1 0; 1 1];
rand('state',sum(100*clock));
net_weights = -1 +2.*rand(3,3);

% Define learning rate, target output and layer bias
learning_rate = 0.8;
target_output = [0;1;1;0];
net_bias = [-1 -1 -1]; % bias set as -1 for H1,H2 and Y1

%% [2] Run Backpropagation Algorithm (feedforward pass, error detection and backpropagation):

for idx_learning = 1:800000 % no. of iterations (learning)
   final_output = zeros(4,1); % pre-defining output dimensions to accommodate the solution: 0 1 1 0
   
   for j = 1:length (net_inputs(:,1));
       
      % Hidden neuron layer formulation & resulting data is delivered through activation function (logsig implemented)
      H_Neuro_1 = net_bias(1,1)*net_weights(1,1) + net_inputs(j,1)*net_weights(1,2) + net_inputs(j,2)*net_weights(1,3);
      x2(1) = logsig(H_Neuro_1);
      H_Neuro_2 = net_bias(1,2)*net_weights(2,1) + net_inputs(j,1)*net_weights(2,2) + net_inputs(j,2)*net_weights(2,3);
      x2(2) = logsig(H_Neuro_2);

      % Calculate output layer and send through activation function
      x3_1 = net_bias(1,3)*net_weights(3,1) + x2(1)*net_weights(3,2)+ x2(2)*net_weights(3,3);
      final_output(j) = logsig(x3_1);
      acc_mat(idx_learning) = sum(round(final_output)==target_output) / length(net_inputs) * 100; % shows accuracy percentage across iterations
      
      % for the output layer, calculate error and consequent change required (delta) 
      dw3_1 = final_output(j)*(1-final_output(j))*(target_output(j)-final_output(j));
      
      % Initiate backpropagation - propagate calculated change backwards into hidden layer
      dw2_1 = x2(1)*(1-x2(1))*net_weights(3,2)*dw3_1;
      dw2_2 = x2(2)*(1-x2(2))*net_weights(3,3)*dw3_1;
      
      % Add the derived weight changes to initial weights and repeat across learning iterations
      
      for k_idx = 1:3
         if k_idx == 1 % Represents Bias cases
            net_weights(1,k_idx) = net_weights(1,k_idx) + learning_rate*net_bias(1,1)*dw2_1;
            net_weights(2,k_idx) = net_weights(2,k_idx) + learning_rate*net_bias(1,2)*dw2_2;
            net_weights(3,k_idx) = net_weights(3,k_idx) + learning_rate*net_bias(1,3)*dw3_1;
            
         else % In instances where k == 2 or 3, input the following cases to each neuron
            net_weights(1,k_idx) = net_weights(1,k_idx) + learning_rate*net_inputs(j,1)*dw2_1;
            net_weights(2,k_idx) = net_weights(2,k_idx) + learning_rate*net_inputs(j,2)*dw2_2;
            net_weights(3,k_idx) = net_weights(3,k_idx) + learning_rate*x2(k_idx-1)*dw3_1;
         end
      end
   end   
end

final_acc = sum(round(final_output)==target_output) / length(net_inputs) * 100 % final model accuracy percentage

%% [3] Plot Classification Results - Solved XOR Problem:

% redefine inputs & target, plot classification
figure(1);
net_inputs = [0 0 1 1; 0 1 0 1]; % rotate network inputs to plot
target_output = [0 1 1 0]; % rotate target outputs to plot
plotpv(net_inputs,target_output)
title('XOR Solved via Backpropagation'); legend('0','1')
xlabel('Network Input(1)'); ylabel('Network Input(2)')
grid on
x1 = [2 0.2]; x2 = [1.2, -0.6];
y1 = [-0.6 1.8]; y2 = [-1.4, 1];

hold on
plot(x1,y1,'k','LineWidth',2); plot(x2,y2,'k','LineWidth',2);

% adjust colours for ease of viewing
colour_change = gca;
for i = allchild(colour_change)';
    switch i.Marker
        case 'o'
            i.MarkerEdgeColor = 'r';
    end
end