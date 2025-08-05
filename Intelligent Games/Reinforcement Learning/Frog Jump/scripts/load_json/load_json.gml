function load_json(_filename){
	
	var _buffer = buffer_load(_filename);
	var _string = buffer_read(_buffer, buffer_string);
	buffer_delete(_buffer);
	
	var _json = json_parse(_string)
	
	return _json
}

// To read json data just use gml syntax for maps
// Example
// value = _json[$ "key"]