function create_random_matrix(n_rows, n_cols){
	var _matrix = array_create(n_rows)
	for(var _i = 0; _i < n_rows; _i++){
		var _row = array_create(n_cols)
		for(var _j = 0; _j < n_cols; _j++){
			_row[_j] = random_range(-1, 1)
		}
		_matrix[_i] = _row
	}
	return _matrix
}