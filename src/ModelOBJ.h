/*
 * ModelOBJ.h
 *
 *  Created on: Sep 12, 2014
 *      Author: carol
 */

#ifndef MODELOBJ_H_
#define MODELOBJ_H_
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

/*
    Particle system example with collisions using uniform grid

    CUDA 2.1 SDK release 12/2008
    - removed atomic grid method, some optimization, added demo mode.

    CUDA 2.2 release 3/2009
    - replaced sort function with latest radix sort, now disables v-sync.
    - added support for automated testing and comparison to a reference value.
*/

// OpenGL Graphics includes

/*
// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h

// Includes


*/

/***************************************************************************
 OBJ Loading
 ***************************************************************************/
#include <cstdio>
#include <cstdlib>

class Model_OBJ {
public:
	Model_OBJ();
	float* calculateNormal(float* coord1, float* coord2, float* coord3);
	int Load(char* filename); // Loads the model
	void Draw(); // Draws the model on the screen
	//void DrawMode(GLenum mode, float alpha);
	void Release(); // Release the model
	void DrawMode(unsigned int mode, float alpha); // Release the model

	float* normals; // Stores the normals
	float* Faces_Triangles; // Stores the triangles
	float* vertexBuffer; // Stores the points which make the object
	long TotalConnectedPoints; // Stores the total number of connected verteces
	long TotalConnectedTriangles; // Stores the total number of connected triangles

};

#define POINTS_PER_VERTEX 3
#define TOTAL_FLOATS_IN_TRIANGLE 9
using namespace std;


#endif /* MODELOBJ_H_ */
