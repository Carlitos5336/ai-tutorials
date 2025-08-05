function mutate_matrix(_matrix, _prob, _minv, _maxv, _minclip, _maxclip){
	
	var _n_rows = array_length(_matrix)
	var _n_cols = array_length(_matrix[0])
	
	_matrix = matrix_copy(_matrix)
	
	for(var _i = 0; _i < _n_rows; _i++){
		for(var _j = 0; _j < _n_cols; _j++){
			if(random(100) < _prob){
				_matrix[_i][_j] = clamp(_matrix[_i][_j] + random_range(_minv, _maxv), _minclip, _maxclip)
			}
		}
	}
	return _matrix
}