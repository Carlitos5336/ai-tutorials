// Inherit the parent event
event_inherited();

if(instance_exists(target)){
	x = target.x + target_xoff
	y = target.y + target_yoff
	select_anim.x = x
	select_anim.y = y
}