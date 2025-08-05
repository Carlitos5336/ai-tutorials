y = pivot_y + sin(current_time*swing_speed+swing_offset)*swing_amp

image_xscale = lerp(abs(image_xscale), 1, 0.1)*sign(image_xscale)
image_yscale = lerp(abs(image_yscale), 1, 0.1)*sign(image_yscale)