draw_self()
if(global.debug){
	
	// Draw collision mask
	draw_set_alpha(0.5)
	draw_rectangle_color(bbox_left, bbox_top, bbox_right, bbox_bottom, c_red, c_red, c_red, c_red, false)
	draw_set_alpha(1)
}