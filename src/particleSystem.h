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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"


//custom
#include "ModelOBJ.h"

#define NINPUT	2500000

#define PATH_INI	"."
#define PATH_INI2	"/home/cxnv/workspace/Simulator"
#define OBJ_PATH	PATH_INI"/data/models3d/volumen.obj"
#define DATAFILE_PATH	PATH_INI"/data/datos.csv"

#define MAX_INT 2147483648
#define MAX_HISTOGRAM_INTERVALS 100

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL);
        ~ParticleSystem();

        bool clipped;
        int currentVariable;

        enum FixedVariables{
        	VAR_TEMPERATURE,
        	VAR_PRESSURE,
        	_NUM_VARIABLES
        };
        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            CONFIG_SIMULATION_DATA,
            _NUM_CONFIGS

        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
        };


        //TODO quitar después, es temporal
        void initialSimulationColor();
        void updateColor();
        void changeActiveVariable();
        void setAlpha(float al);
        void setClipped(bool cl)
        {
        	clipped=cl;
        }

        void colorTemperature(int t, float* r);
        void colorVariable(int t,float* r);
        void setFileSource(string filePath);
        void update(float deltaTime);
        void initDefaultData();
        void loadSimulationData(string fileP);
        void reset(ParticleConfig config);

        float *getArray(ParticleArray array);
        void   setArray(ParticleArray array, const float *data, int start, int count);

        int    getNumParticles() const
        {
            return m_numParticles;
        }

        unsigned int getCurrentReadBuffer() const
        {
            return m_posVbo;
        }
        unsigned int getColorBuffer()       const
        {
            return m_colorVBO;
        }

        void *getCudaPosVBO()              const
        {
            return (void *)m_cudaPosVBO;
        }
        void *getCudaColorVBO()            const
        {
            return (void *)m_cudaColorVBO;
        }

        void dumpGrid();
        void dumpParticles(uint start, uint count);

        void setIterations(int i)
        {
            m_solverIterations = i;
        }

        void setDamping(float x)
        {
            m_params.globalDamping = x;
        }
        void setGravity(float x)
        {
            m_params.gravity = make_float3(0.0f, x, 0.0f);
        }

        void setCollideSpring(float x)
        {
            m_params.spring = x;
        }
        void setCollideDamping(float x)
        {
            m_params.damping = x;
        }
        void setCollideShear(float x)
        {
            m_params.shear = x;
        }
        void setCollideAttraction(float x)
        {
            m_params.attraction = x;
        }

        void setColliderPos(float3 x)
        {
            m_params.colliderPos = x;
        }

        void setSelectedSize(float3 x)
        {
            m_params.selectSize = x;
        }

        float getParticleRadius()
        {
            return m_params.particleRadius;
        }
        float3 getColliderPos()
        {
            return m_params.colliderPos;
        }
        float getColliderRadius()
        {
            return m_params.colliderRadius;
        }

        float3 getSelectSize()
        {
        	return m_params.selectSize;
        }

        uint3 getGridSize()
        {
            return m_params.gridSize;
        }
        float3 getWorldOrigin()
        {
            return m_params.worldOrigin;
        }
        float3 getCellSize()
        {
            return m_params.cellSize;
        }

        float getAlpha(){return alpha;}
        void addSphere(int index, float *pos, float *vel, int r, float spacing);
        void generateHistogram();
        void histogramFunc(int index);

    protected: // methods
        ParticleSystem() {}
        uint createVBO(uint size);

        void _initialize(int numParticles);
        void _finalize();

        void initGrid(uint *size, float spacing, float jitter, uint numParticles);

    protected: // data
        bool m_bInitialized, m_bUseOpenGL;
        uint m_numParticles;

        // CPU data
        float *m_hPos;              // particle positions
        float *m_hVel;              // particle velocities

        uint  *m_hParticleHash;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;

        // GPU data
        float *m_dPos;
        float *m_dVel;

        float *m_dSortedPos;
        float *m_dSortedVel;

        // grid data for sorting method
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell

        uint   m_gridSortBits;

        uint   m_posVbo;            // vertex buffer object for particle positions
        uint m_temperatureColor;
        uint m_pressureColor;
        uint   m_colorVBO;          // vertex buffer object for colors

        float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
        float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

        uint* m_histogram;
        uint m_numberHistogramIntervals;
        float maxLocalVar,minLocalVar,width_histogram;

        struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
        struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

        // params
        SimParams m_params;
        uint3 m_gridSize;
        uint m_numGridCells;

        StopWatchInterface *m_timer;

        uint m_solverIterations;


        //custom
        Model_OBJ obj;

        float * xArray;
        float * yArray;
        float * zArray;
        float * temp;
        float * pressureArray;

        float * allVariables;
        int tam;
        float xmax,ymax,zmax;
        float xmin,ymin,zmin;
        float tmin,tmax,pmin,pmax;
        float alpha;//rango normal de 0 a 10
        float xMaxAllowed,yMaxAllowed,zMaxAllowed;

};

#endif // __PARTICLESYSTEM_H__
