if(not instance_exists(target)){
	exit;	
}

var _tx = obj_player.x - cam_w/2
var _ty = obj_player.y - cam_h/2

x = clamp(_tx, 0, room_width-cam_w)
y = clamp(_ty, 0, room_height-cam_h)

camera_set_view_pos(cam, x, y)


// Parallax scrolling

var _lay_id = layer_get_id("Background")
layer_x(_lay_id, x)

_lay_id = layer_get_id("Backgrounds_1")
layer_x(_lay_id, x/1.1)

_lay_id = layer_get_id("Backgrounds_2")
layer_x(_lay_id, x/1.3)

_lay_id = layer_get_id("Backgrounds_3")
layer_x(_lay_id, x/1.7)