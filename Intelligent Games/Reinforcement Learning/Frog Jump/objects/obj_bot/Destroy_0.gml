
instance_destroy(crown)

var _gene = {
	"weights": neural_network.net.weights,
	"biases": neural_network.net.biases,
	"best_dist": x,
	"pineapples": pineapples,
	"hue": hue_shift
}

ds_list_add(obj_genetic_optimizer.genes, _gene)

instance_destroy(neural_network)

if(instance_number(obj_bot) == 1){
	obj_game.reset_sim()
}