//
// Simple passthrough fragment shader
//

precision mediump float;

varying vec2 v_vTexcoord;
varying vec4 v_vColour;
uniform float u_hue_shift; 

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz),
                 vec4(c.gb, K.xy),
                 step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r),
                 vec4(c.r, p.yzx),
                 step(p.x, c.r));
    
    float d = q.x - min(q.w, q.y);
    float e = 1e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)),
                d / (q.x + e),
                q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec3 rgb = clamp( abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0),
                              6.0) - 3.0) - 1.0,
                      0.0,
                      1.0 );
    return c.z * mix(vec3(1.0), rgb, c.y);
}

void main() {
    vec4 tex = texture2D(gm_BaseTexture, v_vTexcoord) * v_vColour;
    vec3 hsv = rgb2hsv(tex.rgb);

    // Shift hue (convert degrees to 0–1)
    float hue_shift = u_hue_shift / 360.0;
    hsv.x = mod(hsv.x + hue_shift, 1.0);

    vec3 shifted_rgb = hsv2rgb(hsv);
    gl_FragColor = vec4(shifted_rgb, tex.a);
}
