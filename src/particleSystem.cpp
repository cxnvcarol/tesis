/*
 // * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "particleSystem.h"
#include "particleSystem.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#include "nvcc_custom_operators.h"

//custom
GLenum modeVolume = GL_TRIANGLES_ADJACENCY;

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

void ParticleSystem::advanceCutter()
{
	//1. check if i'm in limit, if yes then invert direction
	//2.then according to currentcutter and direction forward or advance
	switch (currentCutter) {
		case 0:
			if(cutterX.pos.x>1||cutterX.pos.x<-1) forwardDirectionCutter=!forwardDirectionCutter;
			if(forwardDirectionCutter) forwardCutterX();
			else rewindCutterX();

			break;
		case 1:
					if(cutterY.pos.y>1||cutterY.pos.y<-1) forwardDirectionCutter=!forwardDirectionCutter;
					if(forwardDirectionCutter) forwardCutterY();
					else rewindCutterY();

					break;
		case 2:
					if(cutterZ.pos.z>1||cutterZ.pos.z<-1) forwardDirectionCutter=!forwardDirectionCutter;
					if(forwardDirectionCutter) forwardCutterZ();
					else rewindCutterZ();
		default:
			break;
	}
	updateColor();
}
	void ParticleSystem::forwardCutterX()
		{
			cutterX.pos=cutterX.pos-make_float3(0.05,0,0);

		}
		void ParticleSystem::rewindCutterX()
		{
			cutterX.pos=cutterX.pos+make_float3(0.05,0,0);

		}
		void ParticleSystem::forwardCutterY()
			{
				cutterY.pos=cutterY.pos-make_float3(0,0.05,0);

			}
			void ParticleSystem::rewindCutterY()
			{
				cutterY.pos=cutterY.pos+make_float3(0,0.05,0);

			}
			void ParticleSystem::forwardCutterZ()
				{
					cutterZ.pos=cutterZ.pos-make_float3(0,0,0.05);

				}
				void ParticleSystem::rewindCutterZ()
				{
					cutterZ.pos=cutterZ.pos+make_float3(0,0,0.05);

				}
void ParticleSystem::setFileSource(string filePath) {
	m_bInitialized = false;
	loadSimulationData(filePath);
	m_numParticles = tamMax;
	_initialize(m_numParticles);
}

ParticleSystem::ParticleSystem(uint3 gridSize, bool bUseOpenGL) :
				m_bInitialized(false), m_bUseOpenGL(bUseOpenGL), m_numParticles(0), m_hPos(
						0), m_hVel(0), m_dPos(0), m_gridSize(gridSize), m_timer(
								NULL), m_solverIterations(1), alpha(1), clipped(false), currentVariable(
										0), m_numberHistogramIntervals(MAX_HISTOGRAM_INTERVALS), m_histogram(
												0) {
	colorRangeMode = COLOR_GRADIENT;

	gradientInitialColor = new float[3] { 1, 1, 0 }; //{1,1,0};//yellow default
	gradientFinalColor = new float[3] { 1, 0, 0 }; //{1,0,0};//red default

	m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;

	//TODO set radius smarter
	particleRadius = 1.0f / 640.0f * 3;
	cutterBox.pos = make_float3(-1.2f, -0.8f, 0.8f);
	cutterBox.size = make_float3(0.4f, 0.4f, 0.4f);

	initCutters2();
}

ParticleSystem::~ParticleSystem() {
	_finalize();
	m_numParticles = 0;
}

uint ParticleSystem::createVBO(uint size) {
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

inline float lerp(float a, float b, float t) {
	return a + t * (b - a);
}


void ParticleSystem::initCutters()
{
	cutterX.pos=make_float3(1,0,0);
	cutterX.size=make_float3(0.05,2,2);
	cutterY.pos=make_float3(0,1,0);
	cutterY.size=make_float3(2,0.05,2);
	cutterZ.pos=make_float3(0,0,1);
	cutterZ.size=make_float3(2,2,0.05);

	switch(currentCutter)
	{
	case 0:
		cutterX.pos=make_float3(0,0,0);
		break;
	case 1:
		cutterY.pos=make_float3(0,0,0);
		break;
	case 2:
		cutterZ.pos=make_float3(0,0,0);
		break;
	}
	enableCutting=true;
}
void ParticleSystem::initCutters2()
	{
		cutterX.pos=make_float3(1,0,0);
		cutterX.size=make_float3(0.05,2,2);
		cutterY.pos=make_float3(0,1,0);
		cutterY.size=make_float3(2,0.05,2);
		cutterZ.pos=make_float3(0,0,1);
		cutterZ.size=make_float3(2,2,0.05);
	}

int ParticleSystem::colorVar(int index, float *r) {

	//if(colorRangeMode==COLOR_GRADIENT)//{do this... } else {.. too (for now)} //TODO!
	float *cini;// = gradientInitialColor;
	float *cfin;// = gradientFinalColor;



	float varNorm = 0; //in [0,1]
	int range=1;

	switch (currentVariable) {
	case VAR_TEMPERATURE:
		if(temp[index]>=n_tmin&&temp[index]<=n_tmax)
			{
			varNorm = (temp[index] - n_tmin) / (n_tmax - n_tmin);
			range=1;
			}
		else if(temp[index]<n_tmin)
			{
			varNorm = (temp[index] - tmin) / (n_tmin - tmin);
			range=0;
			}
		else if(temp[index]>n_tmax)
			{
			varNorm = (temp[index] - n_tmax) / (tmax - n_tmax);
			range=2;
			}
		break;
	case VAR_PRESSURE:
		//varNorm = (pressureArray[index] - pmin) / (pmax - pmin);
		if(pressureArray[index]>=n_pmin&&pressureArray[index]<=n_pmax)
		{
			varNorm = (pressureArray[index] - n_pmin) / (n_pmax - n_pmin);
			range=1;
		}
		else if(pressureArray[index]<n_pmin)
		{
			varNorm = (pressureArray[index] - pmin) / (n_pmin - pmin);
			range=0;
		}
		else if(pressureArray[index]>n_pmax)
		{
			varNorm = (pressureArray[index] - n_pmax) / (pmax - n_pmax);
			range=2;
		}

		break;
	case VAR_VELOCITY:
		//varNorm=velArray[index].magnitude/vmax;
		if(velArray[index].magnitude>=n_vmin&&velArray[index].magnitude<=n_vmax)
		{
			varNorm = (velArray[index].magnitude - n_vmin) / (n_vmax - n_vmin);
			range=1;
		}
		else if(velArray[index].magnitude<n_vmin)
		{
			varNorm = (velArray[index].magnitude - vmin) / (n_vmin - vmin);
			range=0;
		}
		else if(velArray[index].magnitude>n_pmax)
		{
			varNorm = (velArray[index].magnitude - n_vmax) / (vmax - n_vmax);
			range=2;
		}
	}

	switch(range)
	{
	case 0:
		cini=lowColor;
		cfin=gradientInitialColor;
		break;
	case 1:
		cini=gradientInitialColor;
		cfin=gradientFinalColor;
		break;
	case 2:
		cini=gradientFinalColor;
		cfin=highColor;
		break;
	}

	float difr = cfin[0] - cini[0];
	float difg = cfin[1] - cini[1];
	float difb = cfin[2] - cini[2];

	float varPrima = varNorm;
	float colorMin = cini[0];
	if (difr < 0) {
		difr = -difr;
		varPrima = 1 - varPrima;
		colorMin = cfin[0];
	}
	r[0] = varPrima * difr + colorMin;
	varPrima = varNorm;
	colorMin = cini[1];
	if (difg < 0) {
		difg = -difg;
		varPrima = 1 - varPrima;
		colorMin = cfin[1];
	}
	r[1] = varPrima * difg + colorMin;

	varPrima = varNorm;
	colorMin = cini[2];
	if (difr < 0) {
		difb = -difb;
		varPrima = 1 - varPrima;
		colorMin = cfin[2];
	}
	r[2] = varPrima * difb + colorMin;
	return range;
}

char* ParticleSystem::getColor(float valor) {

	//if(colorRangeMode==COLOR_GRADIENT)//{do this... } else {.. too (for now)} //TODO!
	float *cini;// = gradientInitialColor;
	float *cfin;// = gradientFinalColor;


	float* r=(float*)calloc(3,sizeof(float));

	float varNorm = 0; //in [0,1]
	int range=1;

	switch (currentVariable) {
	case VAR_TEMPERATURE:
		if(valor>=n_tmin&&valor<=n_tmax)
			{
			varNorm = (valor - n_tmin) / (n_tmax - n_tmin);
			range=1;
			}
		else if(valor<n_tmin)
			{
			varNorm = (valor - tmin) / (n_tmin - tmin);
			range=0;
			}
		else if(valor>n_tmax)
			{
			varNorm = (valor - n_tmax) / (tmax - n_tmax);
			range=2;
			}
		break;
	case VAR_PRESSURE:
		//varNorm = (valor - pmin) / (pmax - pmin);
		if(valor>=n_pmin&&valor<=n_pmax)
		{
			varNorm = (valor - n_pmin) / (n_pmax - n_pmin);
			range=1;
		}
		else if(valor<n_pmin)
		{
			varNorm = (valor - pmin) / (n_pmin - pmin);
			range=0;
		}
		else if(valor>n_pmax)
		{
			varNorm = (valor - n_pmax) / (pmax - n_pmax);
			range=2;
		}

		break;
	case VAR_VELOCITY:
		//varNorm=valor/vmax;
		if(valor>=n_vmin&&valor<=n_vmax)
		{
			varNorm = (valor - n_vmin) / (n_vmax - n_vmin);
			range=1;
		}
		else if(valor<n_vmin)
		{
			varNorm = (valor - vmin) / (n_vmin - vmin);
			range=0;
		}
		else if(valor>n_pmax)
		{
			varNorm = (valor - n_vmax) / (vmax - n_vmax);
			range=2;
		}
	}

	switch(range)
	{
	case 0:
		cini=lowColor;
		cfin=gradientInitialColor;
		break;
	case 1:
		cini=gradientInitialColor;
		cfin=gradientFinalColor;
		break;
	case 2:
		cini=gradientFinalColor;
		cfin=highColor;
		break;
	}

	float difr = cfin[0] - cini[0];
	float difg = cfin[1] - cini[1];
	float difb = cfin[2] - cini[2];

	float varPrima = varNorm;
	float colorMin = cini[0];
	if (difr < 0) {
		difr = -difr;
		varPrima = 1 - varPrima;
		colorMin = cfin[0];
	}
	r[0] = varPrima * difr + colorMin;
	varPrima = varNorm;
	colorMin = cini[1];
	if (difg < 0) {
		difg = -difg;
		varPrima = 1 - varPrima;
		colorMin = cfin[1];
	}
	r[1] = varPrima * difg + colorMin;

	varPrima = varNorm;
	colorMin = cini[2];
	if (difr < 0) {
		difb = -difb;
		varPrima = 1 - varPrima;
		colorMin = cfin[2];
	}
	r[2] = varPrima * difb + colorMin;

	int red=int(r[0]*255),green=int(r[1]*255),blue=int(r[2]*255);
	printf("getcolor:%d,%d,%d...",red,green,blue);

	std::stringstream stream;
	stream << "0x";
	stream << std::setfill ('0') << std::setw(2)
			<< std::hex << red;
	stream << std::setfill ('0') << std::setw(2)
				<< std::hex << green;
	stream << std::setfill ('0') << std::setw(2)
				<< std::hex << blue;

	return &stream.str()[0];
}

void ParticleSystem::_initialize(int numParticles) {
	assert(!m_bInitialized);

	printf("inicializando numP: %d\n", numParticles);

	m_numParticles = numParticles;

	m_histogram = new uint[m_numberHistogramIntervals];
	// allocate host storage
	m_hPos = new float[m_numParticles * 8];
	m_hVel = new float[m_numParticles * 8];
	memset(m_hPos, 0, m_numParticles * 8 * sizeof(float));
	memset(m_hVel, 0, m_numParticles *8* sizeof(float));
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;

	if (m_bUseOpenGL) {
		//m_posVbo = createVBO(memSize);
		printf("memsize:%d",memSize);
		m_posVbo = createVBO(memSize*2);
		printf("posvbo created");
			fflush(stdout);

		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);

	} else {
		checkCudaErrors(cudaMalloc((void ** )&m_cudaPosVBO, memSize));

	}

	allocateArray((void **) &m_dSortedPos, memSize);
	allocateArray((void **) &m_dSortedVel, memSize);


	if (m_bUseOpenGL) {
		m_colorVBO = createVBO(m_numParticles * 4 * sizeof(float));
		m_colorVBO_vect = createVBO(m_numParticles * 8 * sizeof(float));
		m_pressureColor = createVBO(m_numParticles * 4 * sizeof(float));
		m_temperatureColor = createVBO(m_numParticles * 4 * sizeof(float));
		registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
		registerGLBufferObject(m_colorVBO_vect, &m_cuda_colorvbo_vect_resource);
		initialSimulationColor();
	} else {
		checkCudaErrors(
				cudaMalloc((void ** )&m_cudaColorVBO,
						sizeof(float) * numParticles * 4));
	}

	sdkCreateTimer(&m_timer);


	m_bInitialized = true;
}

void ParticleSystem::initialSimulationColor() {
	//TODO ... make this only once, then just assign data=precalculated data
	// fill color buffer
	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	printf("inicia color");
	struct timeval start, end;
	long mtime, seconds, useconds;

	gettimeofday(&start, NULL);

	for (uint i = 0; i < m_numParticles; i++) {
		float t = i / (float) m_numParticles;

		colorVar(i, ptr);	//colorVariable(i, ptr);
		ptr += 3;

		*ptr++ = alpha;
	}

	gettimeofday(&end, NULL);

	seconds = end.tv_sec - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;

	mtime = ((seconds) * 1000 + useconds / 1000.0) + 0.5;

	printf("\nElapsed time color: %ld milliseconds\n", mtime);

	glUnmapBufferARB(GL_ARRAY_BUFFER);
	printf("\nunmapped arb");
}
void ParticleSystem::setAlpha(float al) {
	alpha = al;

	updateColor();
}
void ParticleSystem::updateFrame() {

	if (currentFrame >= nframes)
		currentFrame = nframes - 1;		//avoid invalid frame value
	else if (currentFrame < 0)
		currentFrame = 0;
	//just before:

	fflush(stdout);
	printf("\new frame: %d\n", currentFrame);
	fflush(stdout);
	velArray = frames[currentFrame].velocityPointer;
	pressureArray = frames[currentFrame].pressurePointer;
	temp = frames[currentFrame].temperaturePointer;
	printf("\nframe changed to: %d\n", currentFrame);
	fflush(stdout);
	updateColor();

}
void ParticleSystem::setCurrentFrame(int newframe) {
	if(newframe>-1&&newframe<nframes)
	{
		currentFrame = newframe;

	}
	else if(newframe<0)
	{
		currentFrame=nframes-1;
	}
	else
	{
		currentFrame=0;
	}
	updateFrame();

}
void ParticleSystem::updateColor() {
	//TODO switch case for every variable - should be a global variable so this is done for active variable!
	//use paramMax and paramMin values to the scale of color

	// fill color buffer
	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	if (!clipped) {
		for (uint i = 0; i < m_numParticles; i++) {

			int rango=colorVar(i, ptr);	//colorVariable(i, ptr);
			ptr += 3;
			switch(rango)
			{
			case 0:
				if(displayLow)
					*ptr++ = alpha;
				else *ptr++ = 0;
				break;
			case 1:
				if(displayMiddle)
					*ptr++ = alpha;
				else *ptr++ = 0;
				break;
			case 2:
				if(displayHigh)
					*ptr++ = alpha;
				else *ptr++ = 0;
				break;
			}

		}
	} else {

		float3 leftDownBack;
		float3 rightUpFront;
		if(enableCutting)
		{
			switch(currentCutter)
			{
			case 0:
				leftDownBack=cutterX.pos-cutterX.size/2;
				rightUpFront=cutterX.pos+cutterX.size/2;
				break;
			case 1:
							leftDownBack=cutterY.pos-cutterY.size/2;
							rightUpFront=cutterY.pos+cutterY.size/2;
							break;
			case 2:
							leftDownBack=cutterZ.pos-cutterZ.size/2;
							rightUpFront=cutterZ.pos+cutterZ.size/2;
							break;

			}
			float3 leftDownBack2 = cutterBox.pos - cutterBox.size / 2;
			float3 rightUpFront2 = cutterBox.pos + cutterBox.size / 2;

			bool intersecX=leftDownBack.x<=rightUpFront2.x&&leftDownBack2.x<=rightUpFront.x;
			bool intersecY=leftDownBack.y<=rightUpFront2.y&&leftDownBack2.y<=rightUpFront.y;
			bool intersecZ=leftDownBack.z<=rightUpFront2.z&&leftDownBack2.z<=rightUpFront.z;
			//printf("\n intersections: (%d,%d,%d)\n",intersecX,intersecY,intersecZ);
			if(intersecX&&intersecY&&intersecZ)
			{
				if(leftDownBack.x<leftDownBack2.x)
					leftDownBack.x=leftDownBack2.x;
				if(leftDownBack.y<leftDownBack2.y)
					leftDownBack.y=leftDownBack2.y;
				if(leftDownBack.z<leftDownBack2.z)
					leftDownBack.z=leftDownBack2.z;

				if(rightUpFront.x>rightUpFront2.x)
					rightUpFront.x=rightUpFront2.x;
				if(rightUpFront.y>rightUpFront2.y)
					rightUpFront.y=rightUpFront2.y;
				if(rightUpFront.z>rightUpFront2.z)
					rightUpFront.z=rightUpFront2.z;
			}

		}
		else{
			leftDownBack = cutterBox.pos - cutterBox.size / 2;
			rightUpFront = cutterBox.pos + cutterBox.size / 2;
		}

		for (uint i = 0; i < m_numParticles; i++) {
			if (m_hPos[i * 4] > leftDownBack.x && m_hPos[i * 4] < rightUpFront.x
					&& m_hPos[i * 4 + 1] > leftDownBack.y
					&& m_hPos[i * 4 + 1] < rightUpFront.y
					&& m_hPos[i * 4 + 2] > leftDownBack.z
					&& m_hPos[i * 4 + 2] < rightUpFront.z) {
				int rango=colorVar(i, ptr);
				ptr += 3;
				switch(rango)
				{
				case 0:
					if(displayLow)
						*ptr++ = alpha;
					else *ptr++ = 0;
					break;
				case 1:
					if(displayMiddle)
						*ptr++ = alpha;
					else *ptr++ = 0;
					break;
				case 2:
					if(displayHigh)
						*ptr++ = alpha;
					else *ptr++ = 0;
					break;
				}
			} else {
				ptr += 3;
				*ptr++ = 0;
			}
		}
	}
	glUnmapBufferARB(GL_ARRAY_BUFFER);

	updateColorVect();
}

void ParticleSystem::updateColorVect() {
	//TODO switch case for every variable - should be a global variable so this is done for active variable!
	//use paramMax and paramMin values to the scale of color

	// fill color buffer
	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO_vect);
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	if (!clipped) {
		for (uint i = 0; i < m_numParticles; i++) {
			int al=rand()%100;

			for (int i2 = 0; i2 < 2; ++i2) {
				colorVar(i, ptr);	//colorVariable(i, ptr);
				ptr += 3;
				*ptr++ = al<2?alpha:0;
			}

		}
	} else {
		float3 leftDownBack;
				float3 rightUpFront;
				if(enableCutting)
				{
					switch(currentCutter)
					{
					case 0:
						leftDownBack=cutterX.pos-cutterX.size/2;
						rightUpFront=cutterX.pos+cutterX.size/2;
						break;
					case 1:
									leftDownBack=cutterY.pos-cutterY.size/2;
									rightUpFront=cutterY.pos+cutterY.size/2;
									break;
					case 2:
									leftDownBack=cutterZ.pos-cutterZ.size/2;
									rightUpFront=cutterZ.pos+cutterZ.size/2;
									break;

					}

				}
				else{
					leftDownBack = cutterBox.pos - cutterBox.size / 2;
								rightUpFront = cutterBox.pos + cutterBox.size / 2;
				}

		for (uint i = 0; i < m_numParticles; i++) {
			int al=rand()%100;
			if (m_hPos[i * 8] > leftDownBack.x && m_hPos[i * 8] < rightUpFront.x
					&& m_hPos[i * 8 + 1] > leftDownBack.y
					&& m_hPos[i * 8 + 1] < rightUpFront.y
					&& m_hPos[i * 8 + 2] > leftDownBack.z
					&& m_hPos[i * 8 + 2] < rightUpFront.z) {
				for (int i2 = 0; i2 < 2; ++i2) {
					colorVar(i, ptr);	//colorVariable(i, ptr);
					ptr += 3;
					//*ptr++ = alpha;
					*ptr++ = al<2?alpha:0;
				}
			} else {
				for (int i2 = 0; i2 < 2; ++i2) {
					ptr += 3;
					*ptr++ = 0;
				}
			}
		}
	}
	glUnmapBufferARB(GL_ARRAY_BUFFER);
}

void ParticleSystem::changeActiveVariable() {
	//TODO iterate over available variables (should be a matrix, not named arrays + an array with the name of each variable.
	//Only fixed are x,y,z,vx,vy,vz... anything else could be even calculated
	//TODO Another probem is the velocity variables... keep in mind!!

	fflush(stdout);
	int comp = (int) _NUM_VARIABLES;
	printf("comp: %d", comp);
	currentVariable = (currentVariable < (comp - 1)) ? currentVariable + 1 : 0;
	printf("current:%d, numvars:%d", currentVariable, comp);

	updateColor();

}

void ParticleSystem::_finalize() {
	assert(m_bInitialized);

	delete[] m_hPos;
	delete[] m_hVel;

	//freeArray(m_dVel);
	freeArray(m_dSortedPos);
	freeArray(m_dSortedVel);

	if (m_bUseOpenGL) {
		unregisterGLBufferObject(m_cuda_posvbo_resource);
		glDeleteBuffers(1, (const GLuint *) &m_posVbo);
		glDeleteBuffers(1, (const GLuint *) &m_colorVBO);
		glDeleteBuffers(1, (const GLuint *) &m_colorVBO_vect);
	} else {
		checkCudaErrors(cudaFree(m_cudaPosVBO));
		checkCudaErrors(cudaFree(m_cudaColorVBO));
	}
}

float *
ParticleSystem::getArray(ParticleArray array) {
	assert(m_bInitialized);

	float *hdata = 0;
	float *ddata = 0;
	struct cudaGraphicsResource *cuda_vbo_resource = 0;

	switch (array) {
	default:
	case POSITION:
		hdata = m_hPos;
		ddata = m_dPos;
		cuda_vbo_resource = m_cuda_posvbo_resource;
		break;

	case VELOCITY:
		hdata = m_hVel;
		//ddata = m_dVel;
		break;
	}

	copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource,
			m_numParticles * 4 * sizeof(float));
	return hdata;
}

void ParticleSystem::setArray(ParticleArray array, const float *data, int start,
		int count) {
	assert(m_bInitialized);

	switch (array) {
	default:
	case POSITION: {
		if (m_bUseOpenGL) {
			unregisterGLBufferObject(m_cuda_posvbo_resource);
			glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
			glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float),
					count * 4 * sizeof(float), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
		}
	}
	break;

	case POSITION_VEL:
	{//TODO is the same as positioin, but it should use a different vector
		if (m_bUseOpenGL) {
			unregisterGLBufferObject(m_cuda_posvbo_resource);
			glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
			glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float),
					count * 4 * sizeof(float), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
		}
	}
	break;
	case VELOCITY:
		//copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float),count * 4 * sizeof(float));
		break;
	}
}

inline float frand() {
	return rand() / (float) RAND_MAX;
}


//TODO Refactor to generalize for more variables or different order
void ParticleSystem::loadSimulationData(string fileP) {
	printf("iniciaCarga");
	cout << fileP;
	fflush(stdout);

	xArray = (float*) malloc(MAX_CELLS * sizeof(float));
	yArray = (float*) malloc(MAX_CELLS * sizeof(float));
	zArray = (float*) malloc(MAX_CELLS * sizeof(float));

	float pressure, temperature, velMag, velX, velY, velZ, time, posX, posY,
	posZ;
	string wall;
	xmax = 0, ymax = 0, zmax = 0, xmin = 0, ymin = 0, zmin = 0;
	tmin = MAX_INT, tmax = -MAX_INT, pmin = MAX_INT, pmax = -MAX_INT, vmax =-MAX_INT, vmin=MAX_INT;

	std::ifstream data(fileP.c_str());

	std::string line;
	getline(data, line);    //skip first row
	std::cout << line << "\n";
	float lasttime = -1;
	int tam = 0;

//	try{//skip data to accelerate rendering
//				for (int var = 0; var < 150000; ++var) {
//					std::getline(data,line);
//				}
//
//			}
//			catch (int e) {
//
//			}
//			int tempCount=0;
	while (std::getline(data, line)){//&&tempCount<400000) {
		//tempCount++;
		if (tam < 2)
			std::cout << line << "\n";

		std::stringstream lineStream(line);
		std::string cell;
		int tempVar = 0;
		while (std::getline(lineStream, cell, ',')) {
			switch (tempVar) {
			case 0:
				pressure = (float) ::atof(cell.c_str());
				break;
			case 1:
				temperature = (float) ::atof(cell.c_str());
				break;
			case 3:
				velMag = (float) ::atof(cell.c_str());
				break;
			case 4:
				velX = (float) ::atof(cell.c_str());
				break;
			case 5:
				velY = (float) ::atof(cell.c_str());
				break;
			case 6:
				velZ = (float) ::atof(cell.c_str());
				break;
			case 7:
				time = (float) ::atof(cell.c_str());
				if (time != lasttime) {
					lasttime = time;
					frames[nframes].time = time;

					temp = (float*) malloc(MAX_CELLS * sizeof(float));
					pressureArray = (float*) malloc(MAX_CELLS * sizeof(float));

					velArray = (velocity*) malloc(MAX_CELLS * sizeof(velocity));


					frames[nframes].pressurePointer = pressureArray;
					frames[nframes].temperaturePointer = temp;
					frames[nframes].velocityPointer = velArray;
					//TODO here I assumed that coordinates order is the same in each frame. Check it out!!

					nframes++;
					if (tam > tamMax)
						tamMax = tam;
					tam = 0;
					printf("new time: %f;  nframes: %d", time, nframes);

				}
				break;
			case 8:
				posX = (float) ::atof(cell.c_str());
				break;
			case 9:
				posY = (float) ::atof(cell.c_str());
				break;
			case 10:
				posZ = (float) ::atof(cell.c_str());
				break;
			}
			tempVar++;
		}
		if (temperature > tmax)
			tmax = temperature;
		if (temperature < tmin) {
			tmin = temperature;
			//printf("p_tmin: %d -> %f\n",tam,tmin);
		}
		if (pressure > pmax)
			pmax = pressure;
		if (pressure < pmin)
			pmin = pressure;

		if (velMag > vmax)
		{
			vmax = velMag;
			//std::cout << line << " <= maxvel\n";
		}
		else if(velMag<vmin)
		{
			vmin=velMag;
			//std::cout << line << " <= minvel\n";
		}

		//...
		if (posX > xmax)
			xmax = posX;
		else if (posX < xmin)
			xmin = posX;

		if (posY > ymax)
			ymax = posY;
		else if (posY < ymin)
			ymin = posY;

		if (posZ > zmax)
			zmax = posZ;
		else if (posZ < zmin)
			zmin = posZ;

		xArray[tam] = posX;
		yArray[tam] = posY;
		zArray[tam] = posZ;
		temp[tam] = temperature;
		pressureArray[tam] = pressure;
		velArray[tam].direction[0] = velX;
		velArray[tam].direction[1] = velY;
		velArray[tam].direction[2] = velZ;
		velArray[tam].magnitude = velMag;
		tam++;
		try{//skip data to accelerate rendering
			for (int var = 0; var < 15; ++var) {
				//std::getline(data,line);
			}

		}
		catch (int e) {

		}
	}
	if (tam > tamMax)
		tamMax = tam;		//last time read

	n_tmin=tmin;
	n_tmax=tmax;
	n_pmin=pmin;
	n_pmax=pmax;
	n_vmin=vmin;
	n_vmax=vmax;

	currentFrame = 0;
	velArray = frames[currentFrame].velocityPointer;
	pressureArray = frames[currentFrame].pressurePointer;
	temp = frames[currentFrame].temperaturePointer;

	xMaxAllowed = max(fabsf(xmax), fabsf(xmin));
	yMaxAllowed = max(fabsf(ymax), fabsf(ymin));
	zMaxAllowed = max(fabsf(zmax), fabsf(zmin));
	printf("\ntam: %d\ntmax:%f;tmin:%f\npmax:%f;pmin:%f;vmax:%f;vmin:%f\n...\n", tam, tmax,
			tmin, pmax, pmin,vmax,vmin);

	printf("\nafter: xmax: %f, ymax: %f, zmax: %f", xmax, ymax, zmax);
	printf("after: xmin: %f, ymin: %f, zmin: %f", xmin, ymin, zmin);
	printf("\n xmall: %f, ymall: %f, zmall: %f\n", xMaxAllowed, yMaxAllowed,
			zMaxAllowed);
	fflush(stdout);
}

void ParticleSystem::initDefaultData() {
	loadSimulationData(DATAFILE_PATH);

}

void ParticleSystem::reset(ParticleConfig config) {
	float min = 100000, maxF = 0;
	switch (config) {
	default:
	case CONFIG_SIMULATION_DATA: {
		int p = 0;
		printf("xmall: %f, ymall: %f, zmall: %f", xMaxAllowed, yMaxAllowed,
				zMaxAllowed);
		fflush(stdout);

		 maxTotal = max(xMaxAllowed, max(yMaxAllowed, zMaxAllowed));

		try {
			uint i = 0;

			for (i = 0; i < m_numParticles; i++) {
				float point[3];

				point[0] = (xArray[i]) / (maxTotal);//scaled to fit into pre-stablished cube
				point[1] = (yArray[i]) / (maxTotal);
				point[2] = (zArray[i]) / (maxTotal);
				m_hPos[p++] = point[0];
				m_hPos[p++] = point[1];
				m_hPos[p++] = point[2];
				m_hPos[p++] = 1.0f; // radius
			}
			printf("vizcaya:  Maxtotal: %f\n", maxTotal);

		} catch (const std::exception &exc) {
			// catch anything thrown within try block that derives from std::exception
			std::cerr << exc.what();
		}

	}
	break;

	case CONFIG_SIMULATION_DATA_VEL: {
		int p = 0;
		maxTotal = max(xMaxAllowed, max(yMaxAllowed, zMaxAllowed));

		try {
			uint i = 0;

			for (i = 0; i < m_numParticles; i++) {
				float point[3], point2[3];

				point[0] = (xArray[i]) / (maxTotal);//scaled to fit into pre-stablished cube
				point[1] = (yArray[i]) / (maxTotal);
				point[2] = (zArray[i]) / (maxTotal);
				//calculateSecondPoint(point,point2,i);
				{
					float dirx=velArray[i].direction[0];
					float diry=velArray[i].direction[1];
					float dirz=velArray[i].direction[2];
					//TODO Afinar factor
					float scaleFactor=particleRadius*20/vmax;

					point2[0]=point[0]+dirx*scaleFactor;
					point2[1]=point[1]+diry*scaleFactor;
					point2[2]=point[2]+dirz*scaleFactor;

				}
				m_hPos[p++] = point[0];
				m_hPos[p++] = point[1];
				m_hPos[p++] = point[2];
				m_hPos[p++] = 1.0f; // radius
				m_hPos[p++] = point2[0];
				m_hPos[p++] = point2[1];
				m_hPos[p++] = point2[2];
				m_hPos[p++] = 1.0f; // radius
			}
		} catch (const std::exception &exc) {
			// catch anything thrown within try block that derives from std::exception
			std::cerr << exc.what();
		}
		setArray(POSITION, m_hPos, 0, m_numParticles*2);
		printf("velocidades!!");
		fflush(stdout);
		return;

	}

	break;
	}

	printf("\ncoordXmax: %f, coordXmin: %f\n", maxF, min);
	fflush(stdout);
	setArray(POSITION, m_hPos, 0, m_numParticles);
	setArray(VELOCITY, m_hVel, 0, m_numParticles);
	printf("vamoos!!");
	fflush(stdout);
}
void ParticleSystem::calculateSecondPoint(float *p1,float *p2,int index)
{
	float dirx=velArray[index].direction[0];
	float diry=velArray[index].direction[1];
	float dirz=velArray[index].direction[2];

	float scaleFactor=particleRadius*500/vmax;//TODO test this number
	p2[0]=p1[0]+dirx*scaleFactor;
	p2[1]=p1[1]+dirx*scaleFactor;
	p2[2]=p1[2]+dirx*scaleFactor;

}
void ParticleSystem::histogramFunc(int ind) {

	int index;
	switch (currentVariable) {
	case VAR_PRESSURE:
		index = (int) ((pressureArray[ind] - minLocalVar) / width_histogram);
		break;
	case VAR_TEMPERATURE:
		index = (int) ((temp[ind] - minLocalVar) / width_histogram);
		break;
	case VAR_VELOCITY:
		index = (int) ((velArray[ind].magnitude - minLocalVar) / width_histogram);
		break;
	}
	m_histogram[index]++;

}

void ParticleSystem::generateHistogram() {

	for (int var = 0; var < m_numberHistogramIntervals; ++var) {
		m_histogram[var] = 0;//memset?
	}

	if (!clipped) {
		//update min-max local:
		minLocalVar = currentVariable == VAR_TEMPERATURE ? tmin : currentVariable == VAR_PRESSURE?pmin:vmin;
		maxLocalVar = currentVariable == VAR_TEMPERATURE ? tmax : currentVariable == VAR_PRESSURE?pmax:vmax;

		width_histogram = maxLocalVar - minLocalVar;
		width_histogram = width_histogram / m_numberHistogramIntervals;
		for (uint i = 0; i < m_numParticles; i++) {

			histogramFunc(i);
		}
	} else {
		float3 leftDownBack;
				float3 rightUpFront;
				if(enableCutting)
				{
					switch(currentCutter)
					{
					case 0:
						leftDownBack=cutterX.pos-cutterX.size/2;
						rightUpFront=cutterX.pos+cutterX.size/2;
						break;
					case 1:
									leftDownBack=cutterY.pos-cutterY.size/2;
									rightUpFront=cutterY.pos+cutterY.size/2;
									break;
					case 2:
									leftDownBack=cutterZ.pos-cutterZ.size/2;
									rightUpFront=cutterZ.pos+cutterZ.size/2;
									break;

					}

				}
				else{
					leftDownBack = cutterBox.pos - cutterBox.size / 2;
								rightUpFront = cutterBox.pos + cutterBox.size / 2;
				}

		//update min-max local

		maxLocalVar = -MAX_INT;
		minLocalVar = MAX_INT;
		for (uint i = 0; i < m_numParticles; i++) {
			if (m_hPos[i * 4] > leftDownBack.x && m_hPos[i * 4] < rightUpFront.x
					&& m_hPos[i * 4 + 1] > leftDownBack.y
					&& m_hPos[i * 4 + 1] < rightUpFront.y
					&& m_hPos[i * 4 + 2] > leftDownBack.z
					&& m_hPos[i * 4 + 2] < rightUpFront.z) {
				switch (currentVariable) {
				case VAR_TEMPERATURE:
					if (temp[i] < minLocalVar)
						minLocalVar = temp[i];
					else if (temp[i] > maxLocalVar)
						maxLocalVar = temp[i];
					break;
				case VAR_PRESSURE:
					if (pressureArray[i] < minLocalVar)
						minLocalVar = pressureArray[i];
					else if (pressureArray[i] > maxLocalVar)
						maxLocalVar = pressureArray[i];
					break;
				case VAR_VELOCITY:
					if (velArray[i].magnitude < minLocalVar)
						minLocalVar = velArray[i].magnitude;
					else if (velArray[i].magnitude > maxLocalVar)
						maxLocalVar = velArray[i].magnitude;
					break;
				}
			}
		}
		//end update min-max

		width_histogram = maxLocalVar - minLocalVar;
		width_histogram = width_histogram / m_numberHistogramIntervals;
		for (uint i = 0; i < m_numParticles; i++) {
			if (m_hPos[i * 4] > leftDownBack.x && m_hPos[i * 4] < rightUpFront.x
					&& m_hPos[i * 4 + 1] > leftDownBack.y
					&& m_hPos[i * 4 + 1] < rightUpFront.y
					&& m_hPos[i * 4 + 2] > leftDownBack.z
					&& m_hPos[i * 4 + 2] < rightUpFront.z) {
				histogramFunc(i);
			}
		}
	}
	printf("minLocal:%f, maxLocal:%f\n\n", minLocalVar, maxLocalVar);

	FILE * myfile = fopen("histog.dat", "w");
   char * colorStr;
	int totalHist = 0;
	for (int var = 0; var < m_numberHistogramIntervals; ++var) {
		totalHist += m_histogram[var];
		colorStr=getColor(minLocalVar + var * width_histogram);

		printf("%s:%f\n",colorStr,minLocalVar + var * width_histogram);
		fprintf(myfile, "%f %d %s\n", minLocalVar + var * width_histogram,m_histogram[var],colorStr);
	}
	printf("\n\ntotal cuenta: %d", totalHist);
	fclose(myfile);

}
