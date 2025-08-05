// Controlling running sfx

var _any_running = false;

with(obj_player) {
    if (state == "running") {
        _any_running = true;
        break; // No need to check others
    }
}

// Start sound if any bot is running and it's not playing
if (_any_running && !audio_is_playing(run_sfx_id)) {
    run_sfx_id = audio_play_sound(sfx_run, 0, 1); // looped sound
}

// Stop sound if no bots are running and sound is playing
if (!_any_running && audio_is_playing(run_sfx_id)) {
    audio_stop_sound(run_sfx_id);
    run_sfx_id = -1;
}

if(keyboard_check_pressed(vk_escape) or keyboard_check_pressed(ord("M"))){
	if(global.mode == "human"){
		room_goto(rm_menu_human)	
	}
	else{
		room_goto(rm_menu_ai)	
	}
}

