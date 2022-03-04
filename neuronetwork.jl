## Reading Data
## File Translation into Array ##
# These Functions are used to Construct the Array from the given File.
# The Array is constructed as an Array of Arrays and which then has
# its values normalized to better work with the Neural Network. This
# Normalization keeps inputs between 1 and 0 to feed to the Network.
function constructarray(file)
	dataset = []
    open(expanduser("~")*file) do f
        for (index, line) in enumerate(eachline(f))
			data = split(line,",")
			data = parse.(Float64, data)
			push!(dataset,data)
		end
	end
	return dataset
end
function normalizearray(dataset)
	for data in dataset
		for i in 1:length(data)-1
			column = [dataset[j][i] for j in 1:length(dataset)]
			data[i] = (data[i] - findmin(column)[1]) / (findmax(column)[1] - findmin(column)[1])
		end
	end
	return dataset
end
## Neural Network
## Initialize Network for Training ##
# Initializing the Network is necessary for Network Training.
# The Network is created through set Input Outputs and Hidden Nodes.
# Since the Input Layer already exist, the Hidden Layer is connected
# along with the Output Layer which forms the Network.
# This Function uses Conditionals to link Connections between the Input
# Hidden and Output Layers. This Network will only have One Hidden Layer.
function initialize_network(inputs, hidden, output)
    network = Vector()
    hidden_layer = [Dict("weight" => [rand(Float64) for i in 1:inputs+1]) for i in 1:hidden]
    push!(network,hidden_layer)
    output_layer = [Dict("weight" => [rand(Float64) for i in 1:hidden+1]) for i in 1:output]
    push!(network,output_layer)
    return network
end
## Forward Propagation through Network ##
# The Output of a Neural Network can be calculated through Propagating
# an Input Signal. Foward Propagation is used to both Train the Network
# and Generate Predictions on a different Data Set. Forward Propagation
# goes through each Layer of each Node and Calculates the Node's Activation
# based on the Initial Inputs, these Inputs are then fed into
# the Next Layer. The Sigmoid Functions keeps the values between 1 and 0.
function activate(weights, inputs)
	# Activation is the Sum of the Weight and Input plus Bias
    activation = weights[end] # Activation is set to Bias
    for i in 1:length(weights)-1
        activation += weights[i] * inputs[i] # Sum of Weight and Input Added to Bias
    end
    return activation
end
function sigmoids(activation)
	# Sigmoid Function is 1 / 1 + e^-x
	return 1.0 / (1.0 + exp(-activation))
end
function forward_propagation(network, data)
	inputs = copy(data) # Initial Inputs
    for layer in network
		update = Vector()
		for node in layer
			# Inputs are Calculated and Stored
			activation = activate(node["weight"], inputs)
			node["output"] = [sigmoids(activation)]
			push!(update,node["output"][1])
		end
		inputs = copy(update) # Set Inputs for Next Layer
	end
	return inputs
end
## Back Propagation through Network ##
# The Error of a Neural Network can be calculated through Propagating the
# Error between the Given Outputs and Expected Output backwards through the
# Network. The Network goes from the Output Layer to the Hidden Layer while
# assigning blame. Error is calculated as the product of the Sigmoid Derivative
# of the Output and the Output subtracted by the Expected. Error Signals are
# Stored which are used in the next layer as the layers are iterated in reverse.
function transfer(output)
	# Derivative of the Sigmoid Function called here as the Sigmoid Transfer Function
	return output * (1.0 - output)
end
function backward_propagation(network, expected)
	for i in Iterators.reverse(1:length(network)) # Propagate Backwards
		layers = network[i]
        errors = Vector()
		if i != length(network)
			for j in 1:length(layers)
                error = 0.0
                for node in network[i + 1] # Relies on Previous Layer having Error Values
					error += (node["weight"][j] * node["change"][1]) # Calculate Error
                end
                push!(errors,error)
            end
        else
			for j in 1:length(layers) # Layer is First in Reverse Order
				node = layers[j]
                push!(errors,node["output"][1] - expected[j]) # Calculate Error
            end
        end
		for j in 1:length(layers) # Add Error to Node in Layer
			node = layers[j]
			node["change"] = [errors[j] * transfer(node["output"][1])] # Sigmoid Derivative
		end
	end
end
## Train Network
## Form Predictions from Dataset ##
# Predictions are made through Foward Propagation and Expected Inputs.
# The Network should already be trained at this point which just makes
# forming predictions easy. The Network is not further trained during
# Predictions.
function predict(network, inputs)
	outputs = forward_propagation(network, inputs)
	return findmax(outputs)[2]
end
## Train Networkfrom Dataset ##
# The network is trained using Stochastic Gradient Descent. Gradient Descent
# optimizes the algorithm by finding the local minimum, used to find the
# values which minimize the cost function. This is done through updating weights
# by the product of the rate error and input subtracted by the weight.
function update_network(network, data, learn_rate)
	for i in 1:length(network)
		inputs = copy(data) # Initial Data
		pop!(inputs)
		if i != 1 # Grabs Outputs from Previous Layer to be used to Calculate
			inputs = [node["output"][1] for node in network[i - 1]]
		end
		for node in network[i] # Each Node in the Layer
			for j in 1:length(inputs) # Calculates each Weight in the Node
				node["weight"][j] -= learn_rate * node["change"][1] * inputs[j]
			end # Calculate the Bias in the Node
			node["weight"][end] -= learn_rate * node["change"][1]
		end
	end
end
function train_network(network, dataset, learn_rate, total_epoch, total_outputs)
	for epoch in 1:total_epoch # For Each Epoch
		sum_error = 0
		for data in dataset
			outputs = forward_propagation(network, data) # Propagate Data in Network
			expects = [0 for i = 1:total_outputs]
			expects[trunc(Int, data[end])] = 1 # Expected Values from Data to Compare
			sum_error += sum([(expects[i]-outputs[i])^2 for i in 1:length(expects)]) # For Debug
			backward_propagation(network, expects) # Back Propagate with Expected Values
			update_network(network, data, learn_rate) # Updates Network in Training it on the Data
		end
		if epoch == 1
			println("") # Because Atom gives me issues and Eats Up Print Statements
		end
		println("epoch = ", epoch, " error = ", sum_error)
	end
end
## Test Network
## Cross Validation Data Partition ##
# Cross Validation is the statistical method used to estimate the skill
# of machine learning models. Various ways to handle this though here
# Data will be partitioned into certain amount of parts which can then
# be tested individually and against eachother to train and form predictions.
function cross_validation_split(dataset, amount)
	dataset_fold = []
	dataset_part = trunc(Int,length(dataset)/amount)
	for i in 1:amount
		fold = []
		while length(fold) < dataset_part
			position = rand(1:length(dataset))
			push!(fold,dataset[position])
			deleteat!(dataset,position:position)
		end
		push!(dataset_fold,fold)
	end
	return dataset_fold
end
## Evaluate Network ##
# Code which is used to Evaluate the Network. The Code here is split into
# two parts which the First Part organizes the data set and feeds it into
# the second part to collect results. The Second part Creates the Networks
# and trains them on a portion of the data before evaluating thier accuracy
function evaluate_algorithm(dataset, rate, epoch, hidden)
	folds = cross_validation_split(dataset, 4)
	score = []
	for fold in folds
		train_set = Vector()
		for i in 1:length(folds)
			if folds[i] != fold
				train_set = union!(train_set, folds[i])
			end
		end
		tests_set = deepcopy(fold)
		for data in tests_set
			data[end] = NaN
		end
		predicted = run_algorithm(train_set, tests_set, rate, epoch, hidden)
		projected = [trunc(Int, data[end]) for data in fold]
		correct = 0
		for i in 1:length(projected)
			if projected[i] == predicted[i]
				correct += 1
			end
		end
		push!(score, (correct / length(projected)) * 100.0)
	end
	return score
end
function run_algorithm(train_set, tests_set, rate, epoch, hidden)
	inputs = length(train_set[1]) -  1
	outputs = length(unique([data[end] for data in train_set]))
	network = initialize_network(inputs, hidden, outputs)
	train_network(network, train_set, rate, epoch, outputs)
	predictions = Vector()
	for data in tests_set
		prediction = predict(network, data)
		push!(predictions,prediction)
	end
	return predictions
end
##
## Neural Network Driver Code ##
# This Driver Code constructs a DataSet based on the given File.
# Along with that is also declares variables which will be used
# to evaluate the Network. The Network will be scored through
# this evaluation and will display the average accuracy.
dataset = constructarray("/Desktop/Computer Science/seeds.txt")
dataset = normalizearray(dataset)
total_hidden = 5
total_epoch = 2000
learn_rate = 0.25
score = evaluate_algorithm(dataset, learn_rate, total_epoch, total_hidden)
println("Scores: ", score)
println("Average: ", sum(score)/(length(score)))
##
## Finished 3/3/2022 at 9:40pm
# https://www.ibm.com/cloud/learn/neural-networks
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
