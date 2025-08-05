
draw_set_font(fnt_game)

if(global.debug and log_stats){
	draw_stats()
}

var _gui_aspect = 1/global.aspect_ratio

draw_set_valign(fa_middle)
draw_sprite_ext(spr_pineapple, 0, display_get_gui_width() - 60*_gui_aspect, 60*_gui_aspect, 1.5*_gui_aspect, 1.5*_gui_aspect, 0, c_white, 1)
draw_text_transformed(display_get_gui_width() - 120*_gui_aspect, 60*_gui_aspect, string(pineapples), _gui_aspect, _gui_aspect, 0)
draw_set_valign(fa_top)

if(state == "win"){
	draw_set_halign(fa_center)
	draw_text_transformed(display_get_gui_width()/2, 160*_gui_aspect, "You won!", _gui_aspect, _gui_aspect, 0)
	draw_text_transformed(display_get_gui_width()/2, 200*_gui_aspect, "Press Escape or M to end level", _gui_aspect, _gui_aspect, 0)
	draw_set_halign(fa_left)
}