part_sys = part_system_create();
var _layer_id = layer_get_id("Particles");
part_system_layer(part_sys, _layer_id);

part_smoke = part_type_create();
part_type_shape(part_smoke, pt_shape_pixel);
part_type_size(part_smoke, 5, 8, 0, 0);
part_type_alpha3(part_smoke, 1, 0.5, 0);
part_type_color1(part_smoke, make_color_rgb(160, 82, 45));
part_type_speed(part_smoke, 1, 2, 0, 0);
part_type_direction(part_smoke, 85, 95, 0, 0);
part_type_gravity(part_smoke, 0.05, 270)
part_type_life(part_smoke, 1 * game_get_speed(gamespeed_fps), 2* game_get_speed(gamespeed_fps));

part_trampolin = part_type_create();
part_type_shape(part_trampolin, pt_shape_pixel);
part_type_size(part_trampolin, 5, 8, 0, 0);
part_type_alpha3(part_trampolin, 1, 0.5, 0);
part_type_color1(part_trampolin, c_white);
part_type_speed(part_trampolin, 1, 2, 0, 0);
part_type_direction(part_trampolin, 85, 95, 0, 0);
part_type_life(part_trampolin, 1 * game_get_speed(gamespeed_fps), 2* game_get_speed(gamespeed_fps));
