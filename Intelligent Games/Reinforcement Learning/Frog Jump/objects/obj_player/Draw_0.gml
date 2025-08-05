
if(sprite_exists(sprite_index)){
	
	shader_set(sh_hueshift);

	var _hue_uniform = shader_get_uniform(sh_hueshift, "u_hue_shift");
	shader_set_uniform_f(_hue_uniform, hue_shift);

	// Draw sprite
	draw_self()

	shader_reset();
}

if(global.debug){
	
	// Draw collision mask
	draw_set_alpha(0.5)
	draw_rectangle_color(bbox_left, bbox_top, bbox_right, bbox_bottom, c_lime, c_lime, c_lime, c_lime, false)
	draw_set_alpha(1)
	
	var _mask = get_debug_lines_mask()
	
	var _show_hdet = _mask[0]
	var _show_vdet = _mask[1]
	var _show_haby = _mask[2]
	var _show_htra = _mask[3]
	
	if(hdist_to_obstacle < horizontal_detection_range and _show_hdet){
		draw_line_width_color(x, bbox_bottom - 10, x+hdist_to_obstacle, bbox_bottom - 10, 4, c_red, c_red)
	}
	if(vdist_to_obstacle < vertical_detection_range and _show_vdet){
		draw_line_width_color(x, bbox_bottom - 10, x, bbox_bottom -vdist_to_obstacle, 4, c_red, c_red)
	}
	if(hdist_to_abyss < horizontal_detection_range and _show_haby){
		draw_line_width_color(x, bbox_bottom - 30, x+hdist_to_abyss, bbox_bottom - 30, 4, c_blue, c_blue)
		draw_line_width_color(x+hdist_to_abyss, bbox_bottom - 30, x+hdist_to_abyss, room_height, 4, c_blue, c_blue)
	}
	if(hdist_to_trampoline < horizontal_detection_range and _show_htra){
		draw_line_width_color(x, bbox_bottom - 10, x+hdist_to_trampoline, bbox_bottom - 10, 4, c_lime, c_lime)
	}
	
}