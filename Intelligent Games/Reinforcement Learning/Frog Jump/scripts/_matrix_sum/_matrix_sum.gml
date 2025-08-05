function _matrix_sum(a, b){
	
	// Get dimensions of arrays
	var am, an, bm, bn; // rows/columns in array a/b
	am = array_length(a);
	an = array_length(a[0]);
	bm = array_length(b);
	bn = array_length(b[0]);
	
	// Verify that dimensions are compatible
	if (am != bm or an != bn){
		return undefined;
	}
	
	var _matrix = array_create(am)
	for(var _i = 0; _i < am; _i++){
		var _row = array_create(an)
		for(var _j = 0; _j < an; _j++){
			_row[_j] = a[_i][_j] + b[_i][_j]
		}
		_matrix[_i] = _row
	}
	
	return _matrix
}