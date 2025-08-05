
var _mpos = get_mouse_position()

if(position_meeting(_mpos[0], _mpos[1], self)){
	if(!is_hovering){
		is_hovering = true
		start_hover()
	}
	hover()
	if(mouse_check_button(mb_left)){
		if(!is_clicking){
			is_clicking = true
			start_click()
		}
		click()
	}
	else{
		if(is_clicking){
			is_clicking = false
			end_click()
		}
	}
}
else{
	if(is_hovering){
		is_hovering = false
		end_hover()
	}
	if(is_clicking){
		is_clicking = false
		end_click()
	}
}