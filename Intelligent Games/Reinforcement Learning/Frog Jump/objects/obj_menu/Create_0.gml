if(!audio_is_playing(snd_menu)){
	audio_play_sound(snd_menu, 0, 1)
}

text = "FROG JUMP"

if(room == rm_menu_human){
	global.mode = "human"
}
else{
	text = "AI FROG JUMP"
	global.mode = "sim"	
}

font_enable_effects(fnt_title, true, {
    outlineEnable: true,
    outlineDistance: 9,
    outlineColour: c_black
});

if(os_browser == browser_not_a_browser){
	// Create exit button
	exit_button = instance_create_layer(room_width - 123, room_height - 100, "Instances", obj_button)
	exit_button.image_xscale = 3
	exit_button.image_yscale = 3
	exit_button.button_text = "X"
	exit_button.subtitle = "exit"
	exit_button.type_ = "exit_game"
	exit_button.select_anim.image_xscale = exit_button.image_xscale+1
	exit_button.select_anim.image_yscale = exit_button.image_yscale+1
}