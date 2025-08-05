image_speed = 0
image_index = 1

select_anim = instance_create_layer(x, y, "Overlay", obj_animation)
select_anim.sprite_index = spr_select
select_anim.visible = false
select_anim.image_xscale = image_xscale+1
select_anim.image_yscale = image_yscale+1

text_yoffset = 0
text_size = 0.8
hover_text_color = c_black
button_text_color = c_black

is_hovering = false
is_clicking = false

function get_mouse_position(){
	return [mouse_x*global.aspect_ratio, mouse_y*global.aspect_ratio,]
}

function start_hover(){
	audio_play_sound(sfx_blob, 0, 0)
	select_anim.visible = true
}

function hover(){
	
}

function end_hover(){
	select_anim.visible = false
}

function start_click(){
	audio_play_sound(sfx_keyboard, 0, 0)
	press_()
}

function click(){
	
}

function press_(){
	image_index = 0
	text_yoffset = 10
}

function release_(){
	image_index = 1
	text_yoffset = 0
}

function end_click(){
	release_()
	if(target_room != rm_menu_ai and target_room != rm_menu_human and
	target_room != rm_config and target_room != rm_credits
	and target_room != noone){
		audio_stop_sound(snd_menu)	
	}
	do_action()
}

function update_debug_button(){
	if(!global.debug){
		sprite = spr_debug_disabled	
	}
	else{
		sprite = spr_debug	
	}
}

function update_neuron_button(){
	if(!global.nn_config[$ button_text]){
		image_blend = c_gray
		release_()
		button_text_color = c_dkgray
	}
	else{
		image_blend = c_white
		press_()
		button_text_color = c_black
	}
}

function up_and_clamp(_var, _min, _max, _step){
	_var += _step
	if(_var>_max) _var = _min
	return _var
}

function do_action(){
	if(type_ == "room_change"){
		room_goto(target_room)	
	}
	if(type_ == "debug"){
		global.debug = !global.debug
		update_debug_button()
	}
	if(type_ == "neuron_input"){
		global.nn_config[$ button_text] = !global.nn_config[$ button_text]
		update_neuron_button()
	}
	if(type_ == "hidden_layer"){
		global.nn_config[$ button_text] = up_and_clamp(global.nn_config[$ button_text], 0, 6, 1)
	}
	if(type_ == "param_change"){
		switch(button_text){
			case "n":
				global.nn_config[$ button_text] = up_and_clamp(global.nn_config[$ button_text], 2, 50, 2)
				break;
			case "mut":
				global.nn_config[$ button_text] = up_and_clamp(global.nn_config[$ button_text], 0, 100, 5)
				break;
			case "hdet":
				global.nn_config[$ button_text] = up_and_clamp(global.nn_config[$ button_text], 100, 1000, 50)
				break;
			case "vdet":
				global.nn_config[$ button_text] = up_and_clamp(global.nn_config[$ button_text], 100, 1000, 50)
				break;
		}
	}
	if(type_ == "exit_game"){
		game_end()
	}

}


// Initing

if(room == rm_menu_ai){
	image_blend = make_color_rgb(255, 188, 43)
}
if(room == rm_menu_human){
	hover_text_color = c_white
}

if(type_ == "debug"){
	update_debug_button()
}
if(type_ == "neuron_input" or type_ == "hidden_layer"){
	text_size = 0.5
}
if(type_ == "param_change"){
	text_size = 0.3	
}
if(type_ == "neuron_input"){
	update_neuron_button()
}