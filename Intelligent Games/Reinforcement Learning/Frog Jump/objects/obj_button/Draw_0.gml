draw_set_font(fnt_menu)

draw_self()
draw_set_color(button_text_color)
draw_set_halign(fa_center)
draw_set_valign(fa_middle)

if(button_text != ""){
	draw_text_transformed(x, y+text_yoffset, button_text, text_size, text_size, 0)
}
else if(sprite != noone){
	draw_sprite(sprite, 0, x, y+text_yoffset)
}

draw_set_color(hover_text_color)

var _activate_subtitle = is_hovering
if(type_ == "neuron_input") _activate_subtitle = global.nn_config[$ button_text]
if(type_ == "hidden_layer" or type_ == "param_change") _activate_subtitle = true

if(_activate_subtitle){
	if(type_ == "neuron_input"){
		draw_set_halign(fa_right)
		draw_text_ext_transformed(x-50, y, subtitle, 60, 500, 0.3, 0.3, 0)
	}
	else if(type_ == "hidden_layer"){
		draw_text_transformed(x+50, y, global.nn_config[$ button_text], 0.5, 0.5, 0)
	}
	else if(type_ == "param_change"){
		draw_set_halign(fa_left)
		draw_text_ext_transformed(x+50, y, subtitle, 60, 500, 0.3, 0.3, 0)
		draw_set_halign(fa_right)
		draw_text_transformed(x-50, y, global.nn_config[$ button_text], 0.5, 0.5, 0)
	}
	else if(type_ == "exit_game"){
		draw_text_transformed(x, y-80, subtitle, 0.5, 0.5, 0)
	}
	else{
		draw_text_transformed(x, y+80, subtitle, 0.5, 0.5, 0)
	}
}

draw_set_halign(fa_left)
draw_set_valign(fa_top)
draw_set_color(c_white)

