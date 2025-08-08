classdef mlpTopology < handle
    properties
        Data = struct('Input',[], 'Desired', []);
        Layers = struct('Hidden', [], 'Out', [], 'IO', []); % call this Connections
        learningRate = 0.2;
        nInputs = 0;
        nPatterns = 0;
        nHiddenLayers = 0;
    end
    
    methods
       function obj = mlpTopology(Input, Desired) 
          
           obj.Data.Input = Input;
           obj.Data.Desired = Desired;
           [obj.nPatterns,obj.nInputs] = size(Input);
           
       end
       
       function addHidden(this, nHidden) 
          
           % layers are structured, (rows, columns) -> (output, input);
           
           if isempty(this.Layers.Hidden)
               
               this.Layers.Hidden(1).Weights = zeros(nHidden,size(this.Data.Input,2)+1); % initialize to zeros
               this.Layers.Hidden(1).Weights = randn(size(this.Layers.Hidden(1).Weights)); % set to random weights
               
               this.Layers.Hidden(1).nHidden = nHidden;
               
           else
                this.Layers.Hidden(end+1).Weights = zeros(nHidden,this.Layers.Hidden(end).nHidden+1); % initialize to zeros
                this.Layers.Hidden(end).Weights = randn(size(this.Layers.Hidden(end).Weights)); % set to random weights
                
                this.Layers.Hidden(end).nHidden = nHidden;

           end
           
           this.Layers.Out.Weights = randn(1,this.Layers.Hidden(end).nHidden+1);
           this.nHiddenLayers = this.nHiddenLayers + 1;
           
       end
       
       
       function trainOneEpoch(this)

            % incremental training loop over patterns
            for i = 1:this.nPatterns
                
                % get current pattern
                in = this.Data.Input(i,:);
                
                
                % -------------------
                %  Forward Propagate
                % -------------------

                % forward propagate through hidden layers
                for j = 1:this.nHiddenLayers
                    if j == 1 % if is input layer use input
                        
                        out = this.feedforward([1,in], this.Layers.Hidden(j).Weights, true); % feedforward to hidden layer
                        this.Layers.Hidden(j).Out = out; 
                        
                    else % if non input use the output of the previous layer 
                        
                        out = this.feedforward([1,out], this.Layers.Hidden(j).Weights, true); % feedforward to hidden layer
                        this.Layers.Hidden(j).Out = out;
                        
                    end
                    
                end

                y_hat = this.feedforward([1,out], this.Layers.Out.Weights, false);

                if ~isempty(this.Layers.IO) % if there are input output connections propagate
                     y_hat = y_hat + this.feedforward(in, this.Layers.IO.Weights, false);
                end 

                % ------------------
                %   Back Propagate
                % ------------------

                err = this.Data.Desired(i)-y_hat; % error on the output
                this.Layers.Out.Error = err; % set error on the final layer

                for j = this.nHiddenLayers:-1:1 % loop backwards over hidden to backpropagate

                     if j == this.nHiddenLayers % if last hidden use error
                        this.Layers.Hidden(j).Bperrs = this.backprop_error(this.Layers.Hidden(j).Out, this.Layers.Out.Weights(:,2:end), err); % backpropagated errors
                     else % use next layers backpropageted error
                        this.Layers.Hidden(j).Bperrs = this.backprop_error(this.Layers.Hidden(j).Out, this.Layers.Hidden(j+1).Weights(:,2:end), this.Layers.Hidden(j+1).Bperrs); % backpropagated errors
                     end
                     
                end 

                % ------------------
                %  Calculate Deltas
                % ------------------

                % update deltas

                 for j = this.nHiddenLayers:-1:1 % loop backwards to update deltas using inputs from the previousLayer
                    
                    if j == this.nHiddenLayers % calculcate delta for the weights to the output
                        this.Layers.Out.Delta = this.backprop_delta([1,this.Layers.Hidden(j).Out], err, this.learningRate);   
                    else
                        this.Layers.Hidden(j+1).Delta = this.backprop_delta([1,this.Layers.Hidden(j).Out], this.Layers.Hidden(j+1).Bperrs, this.learningRate); % delta for non input, non output connections
                    end

                 end

                this.Layers.Hidden(1).Delta = this.backprop_delta([1,in], this.Layers.Hidden(1).Bperrs, this.learningRate); % update delta for input to hidden 1 weights
                
                if ~isempty(this.Layers.IO)
                    this.Layers.IO.Delta = this.backprop_delta(in, err, this.learningRate); % update delta for Input Output Connections
                end 
                 
                 

                 % comment this out for non coursework

                 % set non existant connections to 0
                 this.Layers.Hidden(1).Delta(3,2) = 0;
                 this.Layers.Hidden(1).Delta(1,3) = 0;

                 % set bias deltas to 0
                 this.Layers.Hidden(1).Delta(1:end,1) = 0;

                 this.Layers.Out.Delta(1) = 0;


                % -----------------------
                %  Weight Updates Deltas
                % -----------------------

                % update the to -> hiddenWeights using their Delta
                
                for j = 1:(this.nHiddenLayers)
                    this.Layers.Hidden(j).Weights = this.update_weights_online(this.Layers.Hidden(j).Weights, this.Layers.Hidden(j).Delta);
                end
                
                % for the last summation layer
                this.Layers.Out.Weights = this.update_weights_online(this.Layers.Out.Weights, this.Layers.Out.Delta); % update to -> Output weights using deltas
   
                if ~isempty(this.Layers.IO)
                     this.Layers.IO.Weights = this.update_weights_online(this.Layers.IO.Weights, this.Layers.IO.Delta); % update Input -> Output weights using deltas
                end
                
                this.showUpdatedWeights() % show updated weights after each pattern
                
            end
       end
    
    % -------------------------------------
    %   function to Print Results
    % -------------------------------------
    function showUpdatedWeights(this)
        fprintf("-------- Updated Weights --------");
        for i = 1:this.nHiddenLayers
           if i == 1
               fprintf("\n - Weights from Input to Hidden %d - \n\n", i);
               disp(this.Layers.Hidden(i).Weights)
               disp(" ..................")
           else
               fprintf("\n - Weights from Hidden %d to Hidden %d - \n\n", i-1, i);
               disp(this.Layers.Hidden(i).Weights)
               disp(" ..................")
           end
               
        end
        
        
        fprintf("\n - Weights From Hidden %d to Output - \n\n", this.nHiddenLayers);
        disp(this.Layers.Out.Weights)
        disp(" ..................")
        
        if ~isempty(this.Layers.IO)
            fprintf("\n - Weights Input/Output - \n\n")
            disp(this.Layers.IO.Weights)
            disp(" ..................")
        end 
        disp("---------------------------------");
        
        
    end 

    % -------------------------------------
    %   function to Feedforward *one layer
    % -------------------------------------

    function outputs = feedforward(this, inputs, weights, activation)
        
        nOutputs = size(weights,1); % rows of weights represent number of outputs (based on nHidden)
        nInputs = size(inputs,2); % columns of weights / length of inputs represent number of inputs to layer 

        outputs = zeros(1,nOutputs);

        for i = 1:nOutputs
            for j = 1:nInputs
                outputs(i) = outputs(i) + inputs(j)*weights(i,j);
            end

            % check if activation function is needed or if is summed output
            if activation == true
                outputs(i) = this.sigmoid(outputs(i));
            end
        end

    end

    % -----------------------------------------
    %       backpropagate error one layer
    % -----------------------------------------

    function bperr = backprop_error(this, thisLayerOutput, nextLayerWeights, nextLayerError)
        
        % using the error on the next layer and its weights computer the
        % error on the inputs of this layer
        
        nOutputs = size(thisLayerOutput,2); % length of this layer output (nHidden in this layer) / number of inputs to the weights
        nInputs = size(nextLayerWeights,1); % number of outputs of the weights
   
        bperr = zeros(1,nOutputs);

        for i=1:nInputs
            for j=1:nOutputs
                bperr(1,j) = this.dsigmoid(thisLayerOutput(j))*nextLayerError(i)*nextLayerWeights(i,j);
            end
        end
    end

    % -----------------------------------------
    %    backpropagation of deltas one layer
    % -----------------------------------------

    function deltas = backprop_delta(this, inputsToThisLayer, errorOnNextLayer, learningRate)
        
        % rows of weight matrix (nOutputs)
        nOutputs = size(errorOnNextLayer,2); % length of bpper on next layer (number of 
        
        % columns of weight matrix (nInputs)
        nInputs = size(inputsToThisLayer,2); % the outputs of the previous layer / inputs to this layer

        deltas = zeros(nOutputs,nInputs); % rows represent outputs, columns inputs

        for i = 1:nOutputs
            for j = 1:nInputs
                deltas(i,j) = inputsToThisLayer(1,j)*errorOnNextLayer(1,i)*learningRate;
            end
        end

    end

    % -----------------------------------------
    %              Weight Updating
    % -----------------------------------------

    function updated_weights = update_weights_online(this, weights, deltas)
        
        updated_weights = zeros(size(weights));

        for i = 1:size(weights,1) 
            for j = 1:size(weights,2)

                updated_weights(i,j) = deltas(i,j)+weights(i,j);
            
            end

        end


    end

    % -----------------------------------------
    %       Sigmoid Activation Function
    % -----------------------------------------

    function out = sigmoid(this, in)
        out = 1.0 ./( 1.0 + exp( -in ));
    end

    % -----------------------------------------
    %  Sigmoid Activation Function Derivative 
    % -----------------------------------------

    function out = dsigmoid(this, in)
        out = in * (1-in);
    end

       
    end
end

