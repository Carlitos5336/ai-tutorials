
on_ground = place_meeting(x, y+1, obj_wall)
on_ceiling = place_meeting(x, y-1, obj_wall)
wall_in_front = place_meeting(x+1, y, obj_wall)

on_balloon = place_meeting(x, y, obj_balloon)

//move_x = keyboard_check(vk_right) - keyboard_check(vk_left)
jump = get_player_input() and control

hspd = move_x * run_spd

vspd += grav
vspd = min(vspd, max_vspd)

if(on_ground){
	vspd = 0	
}
if(on_ceiling){
	vspd = max(0, vspd)	
}

if(jump and on_ground){
	if(!audio_is_playing(sfx_jump)){
		audio_play_sound(sfx_jump, 0, 0)
	}
	vspd = jump_spd	
}

if(move_x != 0){
	image_xscale = abs(image_xscale)*move_x
}

move_and_collide(hspd, vspd, obj_wall, n_iters_to_resolve_collision)

if(control){
	if(y > room_height+sprite_height or wall_in_front or on_ceiling or place_meeting(x, y, obj_saw)){
		if(!audio_is_playing(sfx_death)){
			audio_play_sound(sfx_death, 0, 0)
		}
		state="dying"
		move_x = 0
		control = false
		image_index = 0
	}
	if(place_meeting(x,y, obj_goal)){
		state="win"	
		move_x = 0
		control = false
		sprite_index = spr_player_idle	
		if(!audio_is_playing(sfx_victory)){
			audio_play_sound(sfx_victory, 0, 0)
		}
	}
}
	

// FSM

switch(state){

	case "idle":
		sprite_index = spr_player_idle
		if(vspd != 0){
			state="on_air"
		}
		else if(hspd != 0){
			state="running"	
		}
		break;
		
	case "running":
		sprite_index = spr_player_run
		if(irandom(4) == 4){
			part_particles_create(obj_particles.part_sys, x, y, obj_particles.part_smoke, 1);
		}
		if(hspd == 0 or not on_ground){
			state="idle"	
		}
		break;
		
	case "on_air":
		if(vspd > 0) sprite_index = spr_player_fall
		else if(vspd < 0) sprite_index = spr_player_jump
		else{
			state="idle"	
		}
		break;
		
	case "dying":
		sprite_index = spr_player_hurt
		if(image_index >= image_number-1){
			sprite_index = spr_player_dead
			state = "dead"
			kill_player()
		}
		break;
		
}

update_sensors_data()

// crown
if(instance_exists(crown)){
	crown.x = x
	crown.y = bbox_top - 32
}