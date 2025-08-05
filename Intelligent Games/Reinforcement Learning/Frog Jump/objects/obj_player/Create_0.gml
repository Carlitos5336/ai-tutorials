move_x = 1
jump = false
jump_pressed = false
run_spd = 5

hspd = 0
vspd = 0
max_vspd = 10

jump_spd = -12

grav = 0.5

on_ground = false
on_ceiling = false
wall_in_front = false

on_balloon = false

state = "idle"
control = true

n_iters_to_resolve_collision = max_vspd
pineapples = 0

debug_text_padding = 25
init_debug_text_pos = [20, 20]

image_xscale = 3
image_yscale = 3

crown = instance_create_layer(x, y, "Instances", obj_animation)
crown.visible = false
crown.sprite_index = spr_crown
crown.image_xscale = 2
crown.image_yscale = 2

horizontal_detection_range = global.nn_config[$ "hdet"]
hdist_to_obstacle = horizontal_detection_range
hdist_to_trampoline = horizontal_detection_range

vertical_detection_range = global.nn_config[$ "vdet"]
vdist_to_obstacle = vertical_detection_range

abyss_vert_detection_range = 1000
abyss_hstep = 32
hdist_to_abyss = horizontal_detection_range

hue_shift = 0

log_stats = true

function find_nearest_object(_obj, _hdetect=0, _vdetect=0, _xoffset=0, _yoffset=0){
	var _list = ds_list_create()
	var _n_collisions = collision_line_list(x+_xoffset, bbox_bottom - 10-_yoffset,
										x+_hdetect, bbox_bottom - 10 - _vdetect,
										_obj, false, true, _list, true)
	var _res_obj = noone
	if(_n_collisions != 0){
		_res_obj = _list[|0]
	}
	ds_list_destroy(_list)
	return _res_obj
}

function update_sensors_data(){
	
	// Set default values to sensors
	hdist_to_obstacle = horizontal_detection_range
	vdist_to_obstacle = vertical_detection_range
	
	// Horizontal obstacle sensor
	var _hnearest_obstacle = find_nearest_object(obj_obstacle, horizontal_detection_range, 0)
	var _hnearest_wall = find_nearest_object(obj_wall, horizontal_detection_range, 0)
	
	var _hdist_to_wall = horizontal_detection_range
	var _hdist_to_obstacle = horizontal_detection_range
	if(_hnearest_obstacle != noone){
		_hdist_to_obstacle = point_distance(x, 0, _hnearest_obstacle.x, 0)
	}
	if(_hnearest_wall != noone){
		_hdist_to_wall = point_distance(x, 0, _hnearest_wall.x, 0)
	}
	
	hdist_to_obstacle = min(
		_hdist_to_wall, _hdist_to_obstacle
	)
	
	// Vertical obstacle sensor
	var _vnearest_obstacle = find_nearest_object(obj_obstacle, 0, vertical_detection_range)
	var _vnearest_wall = find_nearest_object(obj_wall, 0, vertical_detection_range)
	
	var _vdist_to_wall = vertical_detection_range
	var _vdist_to_obstacle = vertical_detection_range
	if(_vnearest_obstacle != noone){
		_vdist_to_obstacle = point_distance(0, y, 0, _vnearest_obstacle.y)
	}
	if(_vnearest_wall != noone){
		_vdist_to_wall = point_distance(0, y, 0, _vnearest_wall.y)
	}
	
	vdist_to_obstacle = min(
		_vdist_to_wall, _vdist_to_obstacle
	)
	
	// Simple abyss detection
	hdist_to_abyss = horizontal_detection_range
	var _nearest_abyss = noone
	for(var _i=abyss_hstep; _i < horizontal_detection_range; _i+=abyss_hstep){
		_nearest_abyss = find_nearest_object(obj_wall, 0, -abyss_vert_detection_range, _i)
		if(_nearest_abyss == noone){
			hdist_to_abyss = _i
			break;
		}
	}
	
	// Vertical obstacle sensor
	var _vnearest_trampoline = find_nearest_object(obj_trampoline, horizontal_detection_range, 0)
	
	hdist_to_trampoline = horizontal_detection_range
	if(_vnearest_trampoline != noone){
		hdist_to_trampoline = point_distance(x, 0, _vnearest_trampoline.x, 0)
	}
	
}

function get_player_input(){
	return keyboard_check(vk_space)	or mouse_check_button(mb_left)
}


function kill_player(){
	alarm_set(0, 1 * room_speed)	
}

function get_debug_text_mask(){
	return [
		true,
		true,
		true,
		true,
		true,
		true,
		true
	];	
}

function get_debug_lines_mask(){
	return [
		true,
		true,
		true,
		true
	];	
}

function draw_stats(){

	var _hdist_obs_str = string(ceil(hdist_to_obstacle))
	if(_hdist_obs_str == string(horizontal_detection_range)) _hdist_obs_str = "MAX"
	
	var _vdist_obs_str = string(ceil(vdist_to_obstacle))
	if(_vdist_obs_str == string(vertical_detection_range)) _vdist_obs_str = "MAX"
	
	var _hdist_aby_str = string(ceil(hdist_to_abyss))
	if(_hdist_aby_str == string(horizontal_detection_range)) _hdist_aby_str = "MAX"

	var _hdist_tra_str = string(ceil(hdist_to_trampoline))
	if(_hdist_tra_str == string(horizontal_detection_range)) _hdist_tra_str = "MAX"
	
	var _ground_txt_color = c_red
	if(on_ground) _ground_txt_color = c_lime

	var _balloon_txt_color = c_red
	if(on_balloon) _balloon_txt_color = c_lime
	
		
	var _mask = get_debug_text_mask()
	
	var _texts = [
		"spd = " + string(hspd) + ", " + string(ceil(vspd)),
	    "hdist_obs = " + _hdist_obs_str,
	    "vdist_obs = " + _vdist_obs_str,
	    "hdist_abyss = " + _hdist_aby_str,
	    "hdist_tramp = " + _hdist_tra_str,
		"on_ground",
		"on_balloon"
	];
	var _colors = [
		c_white,
	    c_white,
		c_white,
		c_white,
		c_white,
		_ground_txt_color,
		_balloon_txt_color
	];
	
	var _n_texts = array_length(array_filter_by_mask(_mask, _mask))
	var _gui_aspect = 1/global.aspect_ratio
	
	// Draw base rect
	draw_set_color(c_black)
	draw_set_alpha(0.3)
	draw_rectangle(0, 0, 300*_gui_aspect, (init_debug_text_pos[1]+debug_text_padding*(_n_texts+1))*_gui_aspect, false)
	draw_set_color(c_white)
	draw_set_alpha(1)
	
	// Draw texts
	var _y_offset = 0;
	for (var _i = 0; _i < array_length(_mask); _i++) {
	    if (_mask[_i]) {
	        draw_text_transformed_color(
	            init_debug_text_pos[0]*_gui_aspect,
	            (init_debug_text_pos[1] + debug_text_padding * _y_offset)*_gui_aspect,
	            _texts[_i],
	            0.8*_gui_aspect, 0.8*_gui_aspect, 0,
				_colors[_i], _colors[_i], _colors[_i], _colors[_i], 1
	        )
	        _y_offset += 1
	    }
	}


}