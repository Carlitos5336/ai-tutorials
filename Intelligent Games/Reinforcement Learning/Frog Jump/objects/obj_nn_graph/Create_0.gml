function draw_neurons(_x, _y1, _y2, _n, _d_max, _radius, _filled)
{
    if (_n <= 0) return;

    var _dir = sign(_y2 - _y1);
    var _total_height = abs(_y2 - _y1);
    var _spacing = 0;
	
	var _circles_pos = array_create(0)

    if (_n == 1)
    {
        // Single circle exactly centered
        var _y = (_y1 + _y2) * 0.5;
        draw_circle(_x, _y, _radius, !_filled);
		array_push(_circles_pos, _y)
        return _circles_pos;
    }

    // For multiple circles
    _spacing = _total_height / (_n - 1);

    // If spacing would be too large, reduce height and center
    if (_spacing > _d_max)
    {
        _spacing = _d_max;
        _total_height = _spacing * (_n - 1);
    }

    var _start_y = (_y1 + _y2) * 0.5 - _dir * (_total_height * 0.5);

    for (var _i = 0; _i < _n; _i++)
    {
        var _y = _start_y + _dir * _i * _spacing;
        draw_circle(_x, _y, _radius, !_filled);
		array_push(_circles_pos, _y)
    }
	return _circles_pos;
}

neurons_pos = array_create(0)

no_input = false

miny = 258
maxy = 650
minx = 347
maxx = 862

input_layer_pos = [
	258,
	351,
	450,
	546,
	650
]

hidden_x = {
	"h0": 214,
	"h1": 347,
	"h2": 477,
	"h3": 611,
	"h4": 729,
	"h5": 862
}