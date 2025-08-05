function _matrix_map(_mat, _func) {
	var _out = [];
	for (var _i = 0; _i < array_length(_mat); _i++) {
		_out[_i] = [];
		for (var _j = 0; _j < array_length(_mat[_i]); _j++) {
			_out[_i][_j] = _func(_mat[_i][_j]);
		}
	}
	return _out;
}