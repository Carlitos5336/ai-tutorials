n_bots = global.nn_config[$ "n"]
mutation_prob = global.nn_config[$ "mut"]
n_generations = 0

best_gene = noone
best_reward = 0

bots = ds_list_create()
genes = ds_list_create()

#region Genetic Algorithm Optimization

function create_bot(_hue=noone){
	var _x_offset = irandom_range(-128, 0)
	var _y_offset = irandom_range(-64, 0)
	var _bot = instance_create_layer(x+_x_offset, y+_y_offset, "Instances", obj_bot)
	if(_hue == noone){
		_bot.hue_shift = random_range(0, 360)
	}
	else{
		_bot.hue_shift = _hue
	}
	_bot.log_stats = false
	return _bot
}

function init_gen(_n){
	repeat(_n){
		var _bot = create_bot()
		ds_list_add(bots, _bot)
	}
}

function calculate_reward(_gene){
	return _gene.best_dist
}

function update_best_gene(){
	for(var _i = 0; _i < ds_list_size(genes); _i++){
		var _gene = genes[|_i]
		var _gene_reward = calculate_reward(_gene)
		if(_gene_reward-10 > best_reward){
			best_gene = _gene
			best_reward = _gene_reward
		}
	}
}

function next_gen(){
	
	ds_list_clear(bots)
	
	// Create best player again
	update_best_gene()
	var _bestbot = create_bot(best_gene.hue)
	_bestbot.neural_network.net.weights = best_gene.weights
	_bestbot.neural_network.net.biases = best_gene.biases
	_bestbot.x = x+4 // Place it at front
	_bestbot.crown.visible = true
	_bestbot.log_stats = true
	
	init_gen(n_bots-1)
	
	for (var _i = 0; _i < n_bots-1; _i++) {
		var _pob = bots[|_i];
	
		var _new_weights = [];
		var _new_biases  = [];

		for (var _l = 0; _l < array_length(best_gene.weights); _l++) {
			var _w = best_gene.weights[_l];
			var _b = best_gene.biases[_l];

			// Mutate each layer's weights and biases
			var _mutated_w = mutate_matrix(_w, mutation_prob, -0.5, 0.5, -2, 2);
			var _mutated_b = mutate_matrix(_b, mutation_prob, -0.5, 0.5, -3, 3);

			array_push(_new_weights, _mutated_w);
			array_push(_new_biases, _mutated_b);
		}

		_pob.neural_network.net.weights = _new_weights;
		_pob.neural_network.net.biases  = _new_biases;
	}
	
	bots[|n_bots-1] = _bestbot
	
	ds_list_clear(genes)
	
	n_generations += 1
	//show_debug_message("Best W of Gen:" + string(best_gene.weights))
	//show_debug_message("Best B of Gen:" + string(best_gene.biases))
	//show_debug_message("Best Reward:" + string(best_reward))
}

init_gen(n_bots)
bots[|n_bots-1].log_stats = true

#endregion