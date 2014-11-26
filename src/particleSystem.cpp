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
#include "particles_kernel.cuh"

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

void ParticleSystem::setFileSource(string filePath) {
	m_bInitialized = false;
	loadSimulationData(filePath);
	m_numParticles = tamMax;
	_initialize(m_numParticles);
}

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize,
		bool bUseOpenGL) :
		m_bInitialized(false), m_bUseOpenGL(bUseOpenGL), m_numParticles(
				numParticles), m_hPos(0), m_hVel(0), m_dPos(0), m_dVel(0), m_gridSize(
				gridSize), m_timer(NULL), m_solverIterations(1), alpha(1), clipped(
				false), currentVariable(0), m_numberHistogramIntervals(MAX_HISTOGRAM_INTERVALS), m_histogram(0) {
	colorRangeMode=COLOR_GRADIENT;

	gradientInitialColor=new float[3]{1,1,0};//{1,1,0};//yellow default
	gradientFinalColor=new float[3]{1,0,0};//{1,0,0};//red default

	m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;
	//    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

	m_gridSortBits = 18;    // increase this for larger grids

	// set simulation parameters
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.numBodies = m_numParticles;

	//TODO set radius with intelligence
	m_params.particleRadius = 1.0f / 640.0f*3;

	//m_params.particleRadius = 1.0f / 64000.0f;
	m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	m_params.colliderRadius = 0.2f;

	m_params.selectSize =make_float3(0.4f,0.4f,0.4f);

	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	//    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
	float cellSize = m_params.particleRadius * 2.0f; // cell size equal to particle diameter
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

	m_params.spring = 0.5f;
	m_params.damping = 0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.0f;
	m_params.boundaryDamping = -0.5f;

	m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
	m_params.globalDamping = 1.0f;
	m_params.rangeColor = 255;

	_initialize(numParticles);

	//initDefaultData();
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


void ParticleSystem::colorVariable(int index, float *r) {

	if (currentVariable == VAR_TEMPERATURE) {
		r[2] = (temp[index] - tmin) / (tmax - tmin);
		r[1] = r[2];
		r[0] = r[2];
	} else {
		r[0] = 0;
		r[1] = (pressureArray[index] - pmin) / (pmax - pmin);
		r[2] = 0;
	}

}

void ParticleSystem::colorVar(int index, float *r) {

	/*
	float cini[3]={0,0,0};
	float cfin[3]={1,1,1};
	*/

	//if(colorRangeMode==COLOR_GRADIENT)//{do this... } else {.. too (for now)} //TODO!
	float *cini=gradientInitialColor;
	float *cfin=gradientFinalColor;

	float varNorm=0;//in [0,1]

	switch(currentVariable)
	{
	case VAR_TEMPERATURE:
		varNorm = (temp[index] - tmin) / (tmax - tmin);
		break;
	case VAR_PRESSURE:
		varNorm = (pressureArray[index] - pmin) / (pmax - pmin);
		break;
	}

	float difr=cfin[0]-cini[0];
	float difg=cfin[1]-cini[1];
	float difb=cfin[2]-cini[2];


	float varPrima=varNorm;
	float colorMin=cini[0];
	if(difr<0)
	{
		difr=-difr;
		varPrima=1-varPrima;
		colorMin=cfin[0];
	}
	r[0]=varPrima*difr+colorMin;
	varPrima=varNorm;
	colorMin=cini[1];
	if(difg<0)
	{
		difg=-difg;
		varPrima=1-varPrima;
		colorMin=cfin[1];
	}
	r[1]=varPrima*difg+colorMin;

	varPrima=varNorm;
	colorMin=cini[2];
	if(difr<0)
	{
		difb=-difb;
		varPrima=1-varPrima;
	colorMin=cfin[2];
	}
	r[2]=varPrima*difb+colorMin;
}

void colorRamp(float t, float *r) {
	const int ncolors = 7;
	float c[ncolors][3] = { { 1.0, 0.0, 0.0, }, { 1.0, 0.5, 0.0, }, { 1.0, 1.0,
			0.0, }, { 0.0, 1.0, 0.0, }, { 0.0, 1.0, 1.0, }, { 0.0, 0.0, 1.0, },
			{ 1.0, 0.0, 1.0, }, };
	t = t * (ncolors - 1);
	int i = (int) t;
	float u = t - floor(t);
	r[0] = lerp(c[i][0], c[i + 1][0], u);
	r[1] = lerp(c[i][1], c[i + 1][1], u);
	r[2] = lerp(c[i][2], c[i + 1][2], u);
}

void ParticleSystem::_initialize(int numParticles) {
	assert(!m_bInitialized);

	printf("inicializando numP: %d\n", numParticles);

	m_numParticles = numParticles;

	m_histogram=new uint[m_numberHistogramIntervals];
	// allocate host storage
	m_hPos = new float[m_numParticles * 4];
	m_hVel = new float[m_numParticles * 4];
	memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
	memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));

	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));

	// allocate GPU data
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;

	if (m_bUseOpenGL) {
		m_posVbo = createVBO(memSize);
		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
	} else {
		checkCudaErrors(cudaMalloc((void ** )&m_cudaPosVBO, memSize));
	}

	allocateArray((void **) &m_dVel, memSize);

	allocateArray((void **) &m_dSortedPos, memSize);
	allocateArray((void **) &m_dSortedVel, memSize);

	allocateArray((void **) &m_dGridParticleHash,
			m_numParticles * sizeof(uint));
	allocateArray((void **) &m_dGridParticleIndex,
			m_numParticles * sizeof(uint));

	allocateArray((void **) &m_dCellStart, m_numGridCells * sizeof(uint));
	allocateArray((void **) &m_dCellEnd, m_numGridCells * sizeof(uint));

	if (m_bUseOpenGL) {
		m_colorVBO = createVBO(m_numParticles * 4 * sizeof(float));
		m_pressureColor = createVBO(m_numParticles * 4 * sizeof(float));
		m_temperatureColor = createVBO(m_numParticles * 4 * sizeof(float));
		registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
		initialSimulationColor();
	} else {
		checkCudaErrors(
				cudaMalloc((void ** )&m_cudaColorVBO,
						sizeof(float) * numParticles * 4));
	}

	sdkCreateTimer(&m_timer);

	setParameters(&m_params);

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

		if (m_numParticles > 1)
			colorVar(i, ptr);//colorVariable(i, ptr);
		else
			colorRamp(t, ptr);		//here is the color initialization
		ptr += 3;

		*ptr++ = alpha;
	}

	gettimeofday(&end, NULL);

	seconds = end.tv_sec - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;

	mtime = ((seconds) * 1000 + useconds / 1000.0) + 0.5;

	printf("\nElapsed time color: %ld milliseconds\n", mtime);

	glUnmapBufferARB(GL_ARRAY_BUFFER);
}
void ParticleSystem::setAlpha(float al) {
	alpha = al;

	updateColor();
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

			colorVar(i, ptr);//colorVariable(i, ptr);
			ptr += 3;
			*ptr++ = alpha;
		}
	} else {
		float3 leftDownBack = m_params.colliderPos
				- m_params.selectSize/2;
		float3 rightUpFront = m_params.colliderPos
				+ m_params.selectSize/2;

		for (uint i = 0; i < m_numParticles; i++) {
			if (m_hPos[i * 4] > leftDownBack.x && m_hPos[i * 4] < rightUpFront.x
					&& m_hPos[i * 4 + 1] > leftDownBack.y
					&& m_hPos[i * 4 + 1] < rightUpFront.y
					&& m_hPos[i * 4 + 2] > leftDownBack.z
					&& m_hPos[i * 4 + 2] < rightUpFront.z) {
				colorVar(i, ptr);//colorVariable(i, ptr);
				ptr += 3;
				*ptr++ = alpha;
			} else {
				ptr += 3;
				*ptr++ = 0;
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
	delete[] m_hCellStart;
	delete[] m_hCellEnd;

	freeArray(m_dVel);
	freeArray(m_dSortedPos);
	freeArray(m_dSortedVel);

	freeArray(m_dGridParticleHash);
	freeArray(m_dGridParticleIndex);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);

	if (m_bUseOpenGL) {
		unregisterGLBufferObject(m_cuda_posvbo_resource);
		glDeleteBuffers(1, (const GLuint *) &m_posVbo);
		glDeleteBuffers(1, (const GLuint *) &m_colorVBO);
	} else {
		checkCudaErrors(cudaFree(m_cudaPosVBO));
		checkCudaErrors(cudaFree(m_cudaColorVBO));
	}
}

// step the simulation
void ParticleSystem::update(float deltaTime) {
	assert(m_bInitialized);

	float *dPos;

	if (m_bUseOpenGL) {
		dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
	} else {
		dPos = (float *) m_cudaPosVBO;
	}

	// update constants
	setParameters(&m_params);

	// integrate
	integrateSystem(dPos, m_dVel, deltaTime, m_numParticles);

	// calculate grid hash
	calcHash(m_dGridParticleHash, m_dGridParticleIndex, dPos, m_numParticles);

	// sort particles based on hash
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(m_dCellStart, m_dCellEnd, m_dSortedPos,
			m_dSortedVel, m_dGridParticleHash, m_dGridParticleIndex, dPos,
			m_dVel, m_numParticles, m_numGridCells);

	// process collisions
	collide(m_dVel, m_dSortedPos, m_dSortedVel, m_dGridParticleIndex,
			m_dCellStart, m_dCellEnd, m_numParticles, m_numGridCells);

	// note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
	if (m_bUseOpenGL) {
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}
}

void ParticleSystem::dumpGrid() {
	// dump grid information
	copyArrayFromDevice(m_hCellStart, m_dCellStart, 0,
			sizeof(uint) * m_numGridCells);
	copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0,
			sizeof(uint) * m_numGridCells);
	uint maxCellSize = 0;

	for (uint i = 0; i < m_numGridCells; i++) {
		if (m_hCellStart[i] != 0xffffffff) {
			uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

			//            printf("cell: %d, %d particles\n", i, cellSize);
			if (cellSize > maxCellSize) {
				maxCellSize = cellSize;
			}
		}
	}

	printf("maximum particles per cell = %d\n", maxCellSize);
}

void ParticleSystem::dumpParticles(uint start, uint count) {
	// debug
	copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource,
			sizeof(float) * 4 * count);
	copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float) * 4 * count);

	for (uint i = start; i < start + count; i++) {
		//        printf("%d: ", i);
		printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i * 4 + 0],
				m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
		printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i * 4 + 0],
				m_hVel[i * 4 + 1], m_hVel[i * 4 + 2], m_hVel[i * 4 + 3]);
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
		ddata = m_dVel;
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

	case VELOCITY:
		copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float),
				count * 4 * sizeof(float));
		break;
	}
}

inline float frand() {
	return rand() / (float) RAND_MAX;
}

void ParticleSystem::initGrid(uint *size, float spacing, float jitter,
		uint numParticles) {
	srand(1973);

	for (uint z = 0; z < size[2]; z++) {
		for (uint y = 0; y < size[1]; y++) {
			for (uint x = 0; x < size[0]; x++) {
				uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

				if (i < numParticles) {
					m_hPos[i * 4] = (spacing * x) + m_params.particleRadius
							- 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius
							- 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius
							- 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					m_hPos[i * 4 + 3] = 1.0f;

					m_hVel[i * 4] = 0.0f;
					m_hVel[i * 4 + 1] = 0.0f;
					m_hVel[i * 4 + 2] = 0.0f;
					m_hVel[i * 4 + 3] = 0.0f;
				}
			}
		}
	}
}

//TODO Refactor to generalize
void ParticleSystem::loadSimulationData(string fileP) {
	printf("iniciaCarga");
	cout << fileP;
	fflush(stdout);

	float pressure, temperature, velMag, velX, velY, velZ, time, posX, posY,
			posZ;
	string wall;
	xmax = 0, ymax = 0, zmax = 0, xmin = 0, ymin = 0, zmin = 0;
	tmin = MAX_INT, tmax = -MAX_INT, pmin = MAX_INT, pmax = -MAX_INT;

	std::ifstream data(fileP.c_str());

	std::string line;
	getline(data, line);    //skip first row
	std::cout << line << "\n";
	float lasttime=-1;
	int tam=0;
	while (std::getline(data, line)) {
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
				velMag=(float)::atof(cell.c_str());
				break;
			case 4:
				velX=(float)::atof(cell.c_str());
				break;
			case 5:
				velY=(float)::atof(cell.c_str());
				break;
			case 6:
				velZ=(float)::atof(cell.c_str());
				break;
			case 7:
				time=(float)::atof(cell.c_str());
				if(time!=lasttime)
				{
					lasttime=time;
					*times++=time;
					nframes++;
					if(tam>tamMax)
						tamMax=tam;
					tam=0;
					xArray = (float*) malloc(MAX_CELLS * sizeof(float));
					yArray = (float*) malloc(MAX_CELLS * sizeof(float));
					zArray = (float*) malloc(MAX_CELLS * sizeof(float));
					temp = (float*) malloc(MAX_CELLS * sizeof(float));
					pressureArray = (float*) malloc(MAX_CELLS * sizeof(float));

					//frames++;
					//datasimulation* newFrame=frames[nframes-1];
					//newFrame->pressurePointer=&pressureArray;
					//newFrame->temperaturePointer=&temp;
					//TODO here I assumed that coordinates order is the same in each frame. Check it out!!

					printf("new time: %f;  nframes: %d",time,nframes);

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
//		if(temperature>tempMax)
//		  tempMax=temperature;
//		if(temperature<tempMin)
//		  tempMin=temperature;
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
		//cout<<temp[tam];
		tam++;
//		try{//skip data to accelerate rendering
//			for (int var = 0; var < 10; ++var) {
//				std::getline(data,line);
//			}
//
//		}
//		catch (int e) {
//
//		}
	}
	if(tam>tamMax)
		tamMax=tam;//last time read
	//TODO assign pointers of temp and pressureArray to the first one of *frames
	xMaxAllowed = max(fabsf(xmax), fabsf(xmin));
	yMaxAllowed = max(fabsf(ymax), fabsf(ymin));
	zMaxAllowed = max(fabsf(zmax), fabsf(zmin));
	printf("\ntam: %d\ntmax:%f;tmin:%f\npmax:%f;pmin:%f\n...\n", tam, tmax,
			tmin, pmax, pmin);

	printf("\nafter: xmax: %f, ymax: %f, zmax: %f", xmax, ymax, zmax);
	printf("after: xmin: %f, ymin: %f, zmin: %f", xmin, ymin, zmin);
	printf("\n xmall: %f, ymall: %f, zmall: %f\n", xMaxAllowed, yMaxAllowed,
			zMaxAllowed);

	fflush(stdout);
}

void ParticleSystem::initDefaultData() {
	obj.Load(OBJ_PATH);
	printf("volumen cargado");
	fflush(stdout);
	loadSimulationData(DATAFILE_PATH);

}

void ParticleSystem::reset(ParticleConfig config) {
	float min = 100000, maxF = 0;
	switch (config) {
	default:
	case CONFIG_RANDOM: {
		int p = 0, v = 0;

		printf("npartrandom: %d", m_numParticles);

		for (uint i = 0; i < m_numParticles; i++) {
			float point[3];
			point[0] = frand();
			point[1] = frand();
			point[2] = frand();
			m_hPos[p++] = 2 * (point[0] - 0.5f);
			m_hPos[p++] = 2 * (point[1] - 0.5f);
			m_hPos[p++] = 2 * (point[2] - 0.5f);
			m_hPos[p++] = 0.1f; // radius
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			maxF < point[0] ?
					maxF = point[0] : (min > point[0] ? min = point[0] : min);
		}

	}
		break;

	case CONFIG_GRID: {
		float jitter = m_params.particleRadius;
		uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
		uint gridSize[3];
		gridSize[0] = gridSize[1] = gridSize[2] = s;
		initGrid(gridSize, m_params.particleRadius * 2.0f, jitter,
				m_numParticles);
	}
		break;
	case CONFIG_SIMULATION_DATA: {
		int p = 0, v = 0;
		printf("xmall: %f, ymall: %f, zmall: %f", xMaxAllowed, yMaxAllowed,
				zMaxAllowed);
		fflush(stdout);

		float maxTotal = max(xMaxAllowed, max(yMaxAllowed, zMaxAllowed));

		try {
			uint i = 0;

			for (i = 0; i < m_numParticles; i++) {
				float point[3];
				//TODO should I resize the points?? then should I save {x,y,z}Array?
				point[0] = (xArray[i]) / (maxTotal);
				point[1] = (yArray[i]) / (maxTotal);
				point[2] = (zArray[i]) / (maxTotal);
				m_hPos[p++] = point[0];
				m_hPos[p++] = point[1];
				m_hPos[p++] = point[2];
				m_hPos[p++] = 1.0f; // radius
				m_hVel[v++] = 0.0f;
				m_hVel[v++] = 0.0f;
				m_hVel[v++] = 0.0f;
				m_hVel[v++] = 0.0f;
				maxF < point[0] ?
						maxF = point[0] :
						(min > point[0] ? min = point[0] : min);
			}
			printf("vizcaya:  Maxtotal: %f\n", maxTotal);

		} catch (const std::exception &exc) {
			// catch anything thrown within try block that derives from std::exception
			std::cerr << exc.what();
		}

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

void ParticleSystem::histogramFunc(int ind){

	int index;
	switch(currentVariable)
	{
	case VAR_PRESSURE:
		index=(int)((pressureArray[ind]-minLocalVar)/width_histogram);
		m_histogram[index]++;
		break;
	case VAR_TEMPERATURE:
		index=(int)((temp[ind]-minLocalVar)/width_histogram);
		m_histogram[index]++;
		break;
	}

}

void ParticleSystem::generateHistogram() {

	for (int var = 0; var < m_numberHistogramIntervals; ++var) {
			m_histogram[var]=0;
		}

	if (!clipped) {
		//update mix-max local:
		minLocalVar=currentVariable==VAR_TEMPERATURE?tmin:pmin;
		maxLocalVar=currentVariable==VAR_TEMPERATURE?tmax:pmax;

		width_histogram=maxLocalVar-minLocalVar;
		width_histogram=width_histogram/m_numberHistogramIntervals;
			for (uint i = 0; i < m_numParticles; i++) {

				histogramFunc(i);
			}
		} else {
			float3 leftDownBack = m_params.colliderPos
					- m_params.selectSize/2;
			float3 rightUpFront = m_params.colliderPos
					+ m_params.selectSize/2;

			//update min-max local

			maxLocalVar=-MAX_INT;
			minLocalVar=MAX_INT;
			for (uint i = 0; i < m_numParticles; i++) {
				if (m_hPos[i * 4] > leftDownBack.x && m_hPos[i * 4] < rightUpFront.x
						&& m_hPos[i * 4 + 1] > leftDownBack.y
						&& m_hPos[i * 4 + 1] < rightUpFront.y
						&& m_hPos[i * 4 + 2] > leftDownBack.z
						&& m_hPos[i * 4 + 2] < rightUpFront.z) {
					switch(currentVariable)
					{
					case VAR_TEMPERATURE:
						if(temp[i]<minLocalVar) minLocalVar=temp[i];
						else if(temp[i]>maxLocalVar) maxLocalVar=temp[i];
						break;
					case VAR_PRESSURE:
						if(pressureArray[i]<minLocalVar) minLocalVar=pressureArray[i];
						else if(pressureArray[i]>maxLocalVar) maxLocalVar=pressureArray[i];
						break;
					}
				}
			}
			//end update min-max

			width_histogram=maxLocalVar-minLocalVar;
			width_histogram=width_histogram/m_numberHistogramIntervals;
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
	printf("minLocal:%f, maxLocal:%f\n\n",minLocalVar,maxLocalVar);

	 FILE * myfile=fopen("histog.dat","w");



	int totalHist=0;
	for (int var = 0; var < m_numberHistogramIntervals; ++var) {
		totalHist+=m_histogram[var];
		printf("%f,%d\n",minLocalVar+var*width_histogram,m_histogram[var]);
		//myfile << "Writing this to a file.\n";
		fprintf(myfile,"%f,%d\n",minLocalVar+var*width_histogram,m_histogram[var]);
	}
	printf("\n\ntotal cuenta: %d",totalHist);
	fclose(myfile);

}
void ParticleSystem::addSphere(int start, float *pos, float *vel, int r,
	float spacing) {
uint index = start;

for (int z = -r; z <= r; z++) {
	for (int y = -r; y <= r; y++) {
		for (int x = -r; x <= r; x++) {
			float dx = x * spacing;
			float dy = y * spacing;
			float dz = z * spacing;
			float l = sqrtf(dx * dx + dy * dy + dz * dz);
			float jitter = m_params.particleRadius * 0.01f;

			if ((l <= m_params.particleRadius * 2.0f * r)
					&& (index < m_numParticles)) {
				m_hPos[index * 4] = pos[0] + dx
						+ (frand() * 2.0f - 1.0f) * jitter;
				m_hPos[index * 4 + 1] = pos[1] + dy
						+ (frand() * 2.0f - 1.0f) * jitter;
				m_hPos[index * 4 + 2] = pos[2] + dz
						+ (frand() * 2.0f - 1.0f) * jitter;
				m_hPos[index * 4 + 3] = pos[3];

				m_hVel[index * 4] = vel[0];
				m_hVel[index * 4 + 1] = vel[1];
				m_hVel[index * 4 + 2] = vel[2];
				m_hVel[index * 4 + 3] = vel[3];
				index++;
			}
		}
	}
}

setArray(POSITION, m_hPos, start, index);
setArray(VELOCITY, m_hVel, start, index);
}
