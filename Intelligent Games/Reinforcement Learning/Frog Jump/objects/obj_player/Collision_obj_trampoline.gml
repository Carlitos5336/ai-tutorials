if(!audio_is_playing(sfx_spring) or global.mode == "human"){
	audio_play_sound(sfx_spring, 0, 0)
}

vspd = other.jump_spd
move_and_collide(0, other.jump_spd, obj_wall, n_iters_to_resolve_collision)
