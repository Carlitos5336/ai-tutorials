function read_txt(_filename){
	
	var _file = file_text_open_read(working_directory + _filename)
	var _string = file_text_read_string(_file)
	file_text_close(_file)
	
	return _string
	
}