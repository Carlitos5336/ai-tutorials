function write_txt(_string, _filename){
	
	var _file = file_text_open_write(working_directory + _filename)
	file_text_write_string(_file, _string)
	file_text_close(_file)
	
}