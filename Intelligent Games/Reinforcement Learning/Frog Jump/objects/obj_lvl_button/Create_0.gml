target = noone
target_xoff = 0
target_yoff = 0

image_xscale = 2
image_yscale = 2

// Inherit the parent event
event_inherited();

select_anim.depth = depth-1

function get_mouse_position(){
	return [mouse_x, mouse_y]
}