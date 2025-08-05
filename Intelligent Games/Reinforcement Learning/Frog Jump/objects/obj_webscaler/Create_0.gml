
function scale_canvas(){
	
	var _bw = base_w;
	var _bh = base_h;
	var _cw = width;
	var _ch = height;
	var _center = true;
	var _aspect = (_bw / _bh);

	if((_cw / _aspect) > _ch){
	    window_set_size((_ch *_aspect), _ch);
	}
	else{
	    window_set_size(_cw, (_cw / _aspect));
	}
	if(_center) {
	    window_center();
	}
	
	global.aspect_ratio = _bw/window_get_width()
	
	display_set_gui_maximize(1/aspect, 1/aspect, 0, 0)

	//view_wport[0] = min(window_get_width(), _bw);
	//view_hport[0] = min(window_get_height(), _bh)
	//surface_resize(application_surface, view_wport[0], view_hport[0]);
	
}


aspect = 1

base_w = 1366
base_h = 768
width = base_w;
height = base_h;