#version 450

layout(location = 0) in vec2 in_pos;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(in_pos.x, in_pos.y, 1.0 - in_pos.x, 1.0);
}
