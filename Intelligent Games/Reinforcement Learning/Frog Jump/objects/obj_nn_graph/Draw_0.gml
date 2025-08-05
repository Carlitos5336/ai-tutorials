if(no_input) exit;

neurons_pos = array_create(0)

array_push(neurons_pos, input_layer_pos)

for(var _i = 1; _i <= 5; _i++){
	var _hidden_name = "h"+string(_i)
	var _n_neurons = global.nn_config[$ _hidden_name]
	var _pos = draw_neurons(hidden_x[$ _hidden_name], miny, maxy, _n_neurons, 100, 30, true)
	array_push(neurons_pos, _pos)
}

for (var _i = 0; _i < array_length(neurons_pos) - 1; _i++)
{
    var _layer1 = neurons_pos[_i];
    if (array_length(_layer1) == 0) continue;

    // Search for next non-empty layer
    var _next_index = -1;
    for (var _j = _i + 1; _j < array_length(neurons_pos); _j++)
    {
        if (array_length(neurons_pos[_j]) > 0)
        {
            _next_index = _j;
            break;
        }
    }

    if (_next_index == -1) continue; // No forward layer found

    var _x1 = hidden_x[$ ("h" + string(_i))];
    var _x2 = hidden_x[$ ("h" + string(_next_index))];
    var _layer2 = neurons_pos[_next_index];
	
	var _s2 = array_length(_layer2)

    for (var _j = 0; _j < array_length(_layer1); _j++)
    {
        if (_i == 0 && !global.nn_config[$ ("x" + string(_j + 1))]) continue;
	
        for (var _k = 0; _k < _s2; _k++)
        {
            draw_line_width_color(_x1, _layer1[_j], _x2, _layer2[_k], max(3, 5/_s2), c_white, c_white);
        }
    }
}
