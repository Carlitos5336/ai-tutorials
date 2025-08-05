draw_set_font(fnt_game)
var _gui_w = display_get_gui_width()
var _gui_h = display_get_gui_height()

var _gui_aspect = 1/global.aspect_ratio

draw_set_halign(fa_center)
draw_text_transformed(_gui_w/2, 20*_gui_aspect, "Generation "+ string(n_generations), _gui_aspect, _gui_aspect, 0)
draw_text_transformed(_gui_w/2, 50*_gui_aspect, "Best R = "+ string((best_reward/obj_goal.x)*100) + "%", _gui_aspect, _gui_aspect, 0)
draw_set_halign(fa_left)