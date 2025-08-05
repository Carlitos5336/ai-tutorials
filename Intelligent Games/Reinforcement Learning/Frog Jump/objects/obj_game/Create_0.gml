function init_human_game(){
	player = instance_create_layer(x, y, "Instances", obj_human)
}
function init_simulation(){
	ga = instance_create_layer(x, y, "Instances", obj_genetic_optimizer)
}

function init_level(){
	if(global.mode == "human"){
		exit_button.target_room = rm_menu_human
		init_human_game()
	}
	else{
		exit_button.target_room = rm_menu_ai
		init_simulation()	
	}
}

if(!audio_is_playing(snd_level)){
	audio_play_sound(snd_level, 0, 1)
}

function reset_level(){
	with(obj_pineapple){
		visible = true	
	}
}

function reset_sim(){
	obj_genetic_optimizer.next_gen()
	reset_level()
}

camera = instance_create_layer(x, y, "Instances", obj_camera)
particles = instance_create_layer(x, y, "Particles", obj_particles)

// Create exit button
exit_button = instance_create_layer(x, y, "Overlay", obj_lvl_button)
exit_button.target = camera
exit_button.button_text = "X"
exit_button.type_ = "room_change"
exit_button.target_xoff = camera.cam_w - 60
exit_button.target_yoff = camera.cam_h - 60
exit_button.text_size = 0.6

init_level()

run_sfx_id = -1