function array_copy_custom(_matrix){
    var _len = array_length(_matrix);
	var _new_matrix = array_create(_len)
    array_copy(_new_matrix, 0, _matrix, 0, _len);
	return _new_matrix
}

function matrix_copy(_m){
	var _copy = [];
	for (var _i = 0; _i < array_length(_m); _i++){
		_copy[_i] = array_copy_custom(_m[_i]); // shallow copy of row
	}
	return _copy;
}

