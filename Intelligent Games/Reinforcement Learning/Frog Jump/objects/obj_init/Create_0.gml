global.mode = "human" //human or sim
global.debug = false // use debug only for human tests
global.nn_config = {
	"x1": true,	
	"x2": true,	
	"x3": true,	
	"x4": true,	
	"x5": true,	
	"h1": 0,
	"h2": 3,
	"h3": 0,
	"h4": 0,
	"h5": 1,
	"n": 50,
	"hdet": 500,
	"vdet": 500,
	"mut": 25
}

global.aspect_ratio = 1

//audio_master_gain(0)

alarm_set(0, 5 * room_speed)

instance_create_layer(room_width/2, 450, "Instances", obj_logo)

if(os_browser != browser_not_a_browser){
	instance_create_layer(x, y, "Instances", obj_webscaler)
}
