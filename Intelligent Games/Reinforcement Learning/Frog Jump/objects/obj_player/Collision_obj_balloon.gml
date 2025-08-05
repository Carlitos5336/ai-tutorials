if(jump){
	if(!audio_is_playing(sfx_balloon) or global.mode == "human"){
		audio_play_sound(sfx_balloon, 0, 0)
	}
	
	other.image_xscale = 2*sign(other.image_xscale)
	other.image_yscale = 2*sign(other.image_yscale)

	vspd = other.jump_spd
	move_and_collide(0, other.jump_spd, obj_wall, n_iters_to_resolve_collision)
}