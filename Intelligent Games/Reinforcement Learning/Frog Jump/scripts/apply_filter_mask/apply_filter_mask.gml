function array_filter_by_mask(_data, _mask) {
    var _result = [];
    var _len = array_length(_data);
    for (var _i = 0; _i < _len; _i++) {
        if(_mask[_i]){
            array_push(_result, _data[_i]);
        }
    }
    return _result;
}
