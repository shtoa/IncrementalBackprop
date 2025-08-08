% Inputs for Training
Inputs = [
    [0,1];
    [1,0];
];

% Desired Output
Desired = [1,1];

% setup network using input and desired output
mlpCoursework = mlpTopology(Inputs, Desired);
mlpCoursework.learningRate = 0.2;

% add a hidden layer with 3 Hidden nodes
nHidden = 3;
mlpCoursework.addHidden(nHidden);

% weights for the coursework (W1-W9)
w = [-0.1,-0.3,-0.2,0.3,0.1,-0.1,0.2,-0.1,0.2]; % W1-W9

% Preallocate Input -> Hidden weights
WeightsToHidden = zeros(nHidden, mlpCoursework.nInputs+1);

% Weights for the Input -> Hidden
WeightsToHidden(1,2) = w(1);
WeightsToHidden(2,2) = w(3);
WeightsToHidden(2,3) = w(4);
WeightsToHidden(3,3) = w(6);
WeightsToHidden(1:3,1) = 0; % set bias to 0

% Set Weights for the Input -> Hidden inside Class
mlpCoursework.Layers.Hidden(1).Weights = WeightsToHidden;

% Preallocate Weights Hidden -> Output
WeightsToOutput = zeros(1,nHidden+1); % hidden and bias
WeightsToOutput(1,1) = 1; 

% for current network topology
WeightsToOutput(1,1) = 0; % set bias to 0
WeightsToOutput(1,2) = w(9);
WeightsToOutput(1,3) = w(8);
WeightsToOutput(1,4) = w(7); 

% weights for Hidden -> Output
mlpCoursework.Layers.Out.Weights = WeightsToOutput;

% Preallocate Weights Input -> Output
IO_Weights = zeros(1, mlpCoursework.nInputs);

IO_Weights(1,1) = w(2);
IO_Weights(1,2) = w(5);

mlpCoursework.Layers.IO.Weights = IO_Weights;

% Train Model Epoch 
mlpCoursework.trainOneEpoch();


