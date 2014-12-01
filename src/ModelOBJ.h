/*
 * ModelOBJ.h
 *
 *  Created on: Sep 12, 2014
 *      Author: carol
 */

#ifndef MODELOBJ_H_
#define MODELOBJ_H_
#include <cstdio>
#include <cstdlib>

class Model_OBJ {
public:
	Model_OBJ();
	float* calculateNormal(float* coord1, float* coord2, float* coord3);
	int Load(char* filename,float maxsize); // Loads the model
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
