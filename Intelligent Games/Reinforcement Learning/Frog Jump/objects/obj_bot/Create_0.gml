// Inherit the parent event
event_inherited();

neural_network = instance_create_layer(x, y, "Instances", obj_neural_network)
//neural_network.b = [[-0.9]]
//neural_network.w[0][0] = 1
//neural_network.w[0][1] = -1
//neural_network.w[0][2] = 1

function get_sensor_matrix(){
	
	var _sensor_array = [
		1-(min(hdist_to_obstacle, horizontal_detection_range)/horizontal_detection_range),
		1-(min(vdist_to_obstacle, vertical_detection_range)/vertical_detection_range),
		1-(min(hdist_to_abyss, horizontal_detection_range)/horizontal_detection_range),
		1-(min(hdist_to_trampoline, horizontal_detection_range)/horizontal_detection_range),
		on_balloon // 0 a 1
	]
	
	// Filter only usable xs
	_sensor_array = array_filter_by_mask(_sensor_array, neural_network.mask_x)
	var _nx = array_length(_sensor_array)
	
	var _matrix = create_random_matrix(_nx, 1)
	for(var _i = 0; _i < _nx; _i++){
		_matrix[_i][0] = _sensor_array[_i]
	}
	
	return _matrix
}

function get_player_input(){
	var _x = get_sensor_matrix()
	var _y = neural_network.evaluate_network_2(_x)
	if(_y[0][0] > 0.5){
		return true	
	}
	return false
}
function kill_player(){
	instance_destroy()	
}

function get_debug_text_mask(){
	return [
		true,
		global.nn_config[$ "x1"],
		global.nn_config[$ "x2"],
		global.nn_config[$ "x3"],
		global.nn_config[$ "x4"],
		true,
		global.nn_config[$ "x5"],
	];	
}

function get_debug_lines_mask(){
	return [
		global.nn_config[$ "x1"],
		global.nn_config[$ "x2"],
		global.nn_config[$ "x3"],
		global.nn_config[$ "x4"],
	];	
}