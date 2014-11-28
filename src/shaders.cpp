/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define STRINGIFY(A) #A
#define GLSL(version,shader) "#version " #version "\n" #shader
#define GLSL_GEOM(version,shader) "#version " #version "\n#extension GL_EXT_geometry_shader4 : enable\n" #shader


// vertex shader
const char *vertexShader = STRINGIFY(
                               uniform float pointRadius;  // point size in world space
                               uniform float pointScale;   // scale to calculate size in pixels
                               uniform float densityScale;
                               uniform float densityOffset;
                               void main()
{
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

    gl_FrontColor = gl_Color;
    //gl_FrontColor = vec4(0,1,0,1);
}
                           );

// pixel shader for rendering points as shaded spheres
const char *spherePixelShader = STRINGIFY(
                                    void main()
{
    //const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
//
    if (mag > 1.0) discard;   // kill pixels outside circle
//
    N.z = sqrt(1.0-mag);
//
//    // calculate lighting
    //float diffuse = max(0.0, dot(lightDir, N));
//    float diffuse=1;

    gl_FragColor = gl_Color;//*diffuse;//diffuse=1
    //gl_FragColor = vec4(0,1,0,1);

}
);
const char *vertexArrowShader=STRINGIFY(
	void main()
	{
	gl_FrontColor = gl_Color;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

	}
);
const char *fragmentArrowShader=GLSL(120,
	void main()
	{

	gl_FragColor = gl_Color;


	//gl_FragColor=vec4(0,1,0,1);
	}
);

const char *geometryArrowShader=GLSL_GEOM(150,

layout(lines) in;
layout(line_strip, max_vertices=6) out;

	void main()
	{
		for(int i=0; i<2; i++)
		  {
		    gl_Position = gl_in[i].gl_Position;

		    gl_FrontColor = gl_FrontColorIn[i];


		    EmitVertex();

		  }

		  EndPrimitive();



		  vec3 a=gl_in[0].gl_Position.xyz;
		  vec3 b=gl_in[1].gl_Position.xyz;
		  vec3 t=normalize(cross(a,b))*0.5;
		  vec3 diff=gl_in[0].gl_Position.xyz - gl_in[1].gl_Position.xyz;
		  vec3 d=normalize(diff);
		  vec3 f=normalize(d+t)*0.2*length(diff);
		  vec3 final=f+b;
		  {//paint arrowhead  part 1
		  /*next: inicio es b (=gl_in[1]), final=final*/
		  gl_Position = gl_in[1].gl_Position;
		  gl_FrontColor = gl_FrontColorIn[1];
		  EmitVertex();
		  gl_Position.xyz = final;
		  gl_FrontColor = gl_FrontColorIn[1];
		  EmitVertex();

		  EndPrimitive();
		  }
		  {//paint arrowhead  part 2
		  		  /*next: inicio es b (=gl_in[1]), final=final*/

			  f=normalize(d-t)*0.2*length(diff);
			  		  final=f+b;
		  		  gl_Position = gl_in[1].gl_Position;
		  		  gl_FrontColor = gl_FrontColorIn[1];
		  		  EmitVertex();
		  		  gl_Position.xyz = final;
		  		  gl_FrontColor = gl_FrontColorIn[1];
		  		  EmitVertex();

		  		  EndPrimitive();
		  		  }


	}
);
