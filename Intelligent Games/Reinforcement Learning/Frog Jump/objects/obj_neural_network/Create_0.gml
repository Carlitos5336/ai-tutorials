net = {
	weights: [],
	biases: [],
	activations: []  
};

function init_network(_layer_sizes) {

	for (var _i = 1; _i < array_length(_layer_sizes); _i++) {
		var _w = create_random_matrix(_layer_sizes[_i], _layer_sizes[_i - 1]);
		var _b = create_random_matrix(_layer_sizes[_i], 1);
		array_push(net.weights, _w);
		array_push(net.biases, _b);
	}
	
}

function evaluate_network(_input_vec, _activation_fn) {
	
	if(n_inputs == 0) return [[0]]
	var _a = _input_vec;
	net.activations = [a];  // Store activations

	for (var _i = 0; _i < array_length(net.weights); _i++) {
		var _z = _matrix_sum(_matrix_multiply(net.weights[_i], _a), net.biases[_i]);
		_a = _activation_fn(z);
		array_push(net.activations, _a);
	}
	
	return _a;
}

function relu(_x_matrix) {
	return _matrix_map(_x_matrix, function(_v) { return max(0, _v); });
}

function sigmoid(_x_matrix) {
	return _matrix_map(_x_matrix, function(_v) { return 1 / (1 + exp(-_v)); });
}

function evaluate_network_2(_input_vec, _hidden_act_fn=relu, _output_act_fn=sigmoid) {
	if(n_inputs == 0) return [[0]]
	var _a = _input_vec;
	for (var _i = 0; _i < array_length(net.weights); _i++) {
		var _z = _matrix_sum(_matrix_multiply(net.weights[_i], _a), net.biases[_i]);
		_a = (_i == array_length(net.weights) - 1) ? _output_act_fn(_z) : _hidden_act_fn(_z);
	}
	return _a;
}

n_inputs = 5
n_outputs = 1

var _global_hidden = [
	global.nn_config[$ "h1"],
	global.nn_config[$ "h2"],
	global.nn_config[$ "h3"],
	global.nn_config[$ "h4"]
]

mask_x = [
	global.nn_config[$ "x1"],
	global.nn_config[$ "x2"],
	global.nn_config[$ "x3"],
	global.nn_config[$ "x4"],
	global.nn_config[$ "x5"]
]
n_inputs = array_length(array_filter_by_mask(mask_x, mask_x))

hidden_sizes = array_filter_by_mask(_global_hidden, _global_hidden)
layers = array_concat([n_inputs], hidden_sizes, [n_outputs])

init_network(layers)