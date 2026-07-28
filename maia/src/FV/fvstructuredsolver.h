// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

#ifndef STRCTBLCK_H
#define STRCTBLCK_H

#include <sys/stat.h>

#include <cstring>
#include <functional>
#include <vector>
#include "COMM/mpioverride.h"
#include "GRID/structuredgrid.h"
#include "GRID/structuredpartition.h"
#include "IO/parallelio.h"
#include "MEMORY/list.h"
#include "fvransmodelconstants.h"
#include "fvstructuredbndrycnd.h"
#include "fvstructuredcell.h"
#include "fvstructuredcomm.h"
#include "fvstructuredfqvariables.h"
#include "fvstructuredinterpolation.h"
#include "fvstructuredpostprocessing.h"
#include "fvstructuredsolverwindowinfo.h"
#include "fvstructuredtimers.h"
#include "fvstructuredzonalbc.h"
#include "solver.h"
#include "variables.h"

#include "ml_coupling.hpp"


class ParallelIoHdf5;

/** \brief Base class of the structured solver
 *
 * This is the base class of the structured solver
 * from which the 2D and 3D solvers are derived
 *
 */
template <MInt nDim>
class FvStructuredSolver : public Solver, public StructuredPostprocessing<nDim, FvStructuredSolver<nDim>> {
  template <MInt nDim_>
  friend class StructuredBndryCnd;
  template <MInt nDim_>
  friend class StructuredInterpolation;
  friend class FvStructuredSolver3DRans;
  friend class FvStructuredSolver2DRans;
  template <class SolverType>
  friend class AccesorStructured; // TODO labels:FV

 public:
  StructuredGrid<nDim>* m_grid;

  // Add fields used from template base class to avoid calling with 'this->'
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_averageStartTimestep;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_averageStopTimestep;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_averageInterval;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_noSamples;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_avgVariableNames;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_averagingFavre;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_avgFavreNames;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_averageVorticity;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_favre;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_kurtosis;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_skewness;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_tempWaveSample;
  using StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::m_postprocessing;
  using Timers = maia::structured::Timers_;

  FvStructuredSolver(MInt solverId, StructuredGrid<nDim>*, MBool* propertiesGroups, const MPI_Comm comm);
  MBool isActive() const { return m_isActive; }
  void initializeFQField();

  MInt timer(const MInt timerId) const;

  MInt noSolverTimers(const MBool allTimings) override {
    TERMM_IF_COND(!allTimings, "FIXME: reduced timings mode not yet supported by StructuredFvSolver.");
    return 16; // Default: load and idle timer
  }

  virtual void writeHeaderAttributes(ParallelIoHdf5* pio, MString fileType);
  virtual void writePropertiesAsAttributes(ParallelIoHdf5* pio, MString path);
  void saveSolverSolution(MBool = false, const MBool = false) override;
  void saveSolution(MBool);
  void savePartitions();
  void saveBoxes();
  virtual void savePointsToAsciiFile(MBool){};
  virtual void initPointsToAsciiFile(){};
  virtual void saveInterpolatedPoints(){};
  virtual void saveNodalBoxes(){};

  virtual void reIntAfterRestart(MBool){};
  MBool prepareRestart(MBool, MBool&) override { return false; };
  void writeRestartFile(MBool) override{};
  void writeRestartFile(const MBool, const MBool, const MString, MInt*) override{};

  // void reIntAfterRestart(MBool) override {};
  // MBool prepareRestart(MBool, MBool *) override;
  // needed for restart
  void shiftCellValuesRestart(MBool);

  void loadRestartFile();
  MBool loadBoxFile(MString, MString, MInt, MInt);
  virtual void loadRestartBC2600(){};
  virtual void loadRestartBC2601(){};
  virtual void loadRestartSTG(MBool) { mTerm(-1, AT_, "not implemented in basic solver"); };
  void saveForcesToAsciiFile(MBool);
  void saveAveragedVariables(MString, MInt, MFloat**);
  void saveProductionTerms(MString, MFloat**);
  void saveDissipation(MString, MFloat*);
  void saveGradients(MString, MFloat**, MString*);
  void saveAverageRestart(MString, MInt, MFloat**, MFloat**, MFloat**, MFloat**);

  void getSolverTimings(std::vector<std::pair<MString, MFloat>>& solverTimings, const MBool allTimings) override;
  void getDomainDecompositionInformation(std::vector<std::pair<MString, MInt>>& domainInfo) override;

  virtual MFloat getCellLengthY(MInt, MInt, MInt) { return 0; };
  virtual MFloat getCellCoordinate(MInt, MInt) { return 0; };

  MInt noInternalCells() const override { return m_noCells; }

  void saveAuxData();
  void saveForceCoefficient(ParallelIoHdf5* parallelIoHdf5);

  void computeAuxData();
  void computeAuxDataRoot();
  virtual void computeDomainWidth(){};
  void computeForceCoef();
  void computeForceCoefRoot();
  virtual void computeFrictionPressureCoef(MBool) = 0;
  virtual ~FvStructuredSolver();

  // SVD stuff
  MFloat computeRecConstSVD(const MInt ijk, const MInt noNghbrIds, MInt* nghbr, MInt ID, MInt sID,
                            MFloatScratchSpace& tmpA, MFloatScratchSpace& tmpC, MFloatScratchSpace& weights,
                            const MInt recDim);

  //! Structured Solver Constructor reads and allocate properties/variables
  void initializeFvStructuredSolver(MBool* propertiesGroups);
  void allocateAndInitBlockMemory();
  void allocateAuxDataMaps();
  void setRungeKuttaProperties();
  void setSamplingProperties();
  void setNumericalProperties();
  void setInputOutputProperties();
  void setZonalProperties();
  void allocateVariables();
  void setTestcaseProperties();
  void setMovingGridProperties();
  void setBodyForceProperties();
  void setPorousProperties();
  void readAndSetSpongeLayerProperties();
  void setSTGProperties();
  void setProfileBCProperties();
  void createMPIGroups();
  void readAndSetAuxDataMap();

  // methods:
  void computePV();
  MFloat dummy(MInt) const { return 0.0; }
  MFloat pressure_twoEqRans(MInt cellId) const { return -mMax(m_cells->variables[CV->RANS_VAR[0]][cellId], F0); }
  MFloat totalEnergy_twoEqRans(MInt cellId) const {
    return mMax(m_cells->pvariables[PV->RANS_VAR[0]][cellId], F0) * m_cells->pvariables[PV->RHO][cellId];
  }
  void partitionGrid();
  virtual void computePrimitiveVariables() { mTerm(-1, AT_, "not implemented in general solver"); };
  virtual void computeConservativeVariables(); //{mTerm(-1,AT_,"not implemented in general solver");};
  template <MFloat (FvStructuredSolver::*)(MInt) const = &FvStructuredSolver::dummy>
  void computeConservativeVariables_();
  void saveVarToPrimitive(MInt, MInt, MFloat);
  virtual void gcFillGhostCells(std::vector<MFloat*>&){};
  void computeSamplingInterval();
  void checkNans();
  void setVolumeForce();
  void computeVolumeForces();
  virtual void computePorousRHS(MBool /*isRans*/) { mTerm(1, "called virtual computePorousRHS()"); };
  void initPorous();
  virtual void applyBoundaryCondtition(){};
  virtual void moveGrid(MInt){};
  virtual void initMovingGrid(){};
  virtual void moveGrid(MBool, MBool){};
  virtual void initBodyForce(){};
  virtual void applyBodyForce(MBool, MBool){};
  virtual void computeLambda2Criterion(){};
  virtual void computeVorticity(){};


  void exchange();
  void exchange(std::vector<std::unique_ptr<StructuredComm<nDim>>>&,
                std::vector<std::unique_ptr<StructuredComm<nDim>>>&);
  void send(const MBool, std::vector<std::unique_ptr<StructuredComm<nDim>>>&, std::vector<MPI_Request>&);
  void receive(const MBool, std::vector<std::unique_ptr<StructuredComm<nDim>>>&, std::vector<MPI_Request>&);
  virtual void gather(const MBool, std::vector<std::unique_ptr<StructuredComm<nDim>>>&){};
  virtual void scatter(const MBool, std::vector<std::unique_ptr<StructuredComm<nDim>>>&){};
  MBool isPeriodicComm(std::unique_ptr<StructuredComm<nDim>>&);
  MBool skipPeriodicDirection(std::unique_ptr<StructuredComm<nDim>>&);

  virtual void zonalExchange(){};
  virtual void spanwiseAvgZonal(std::vector<MFloat*>&){};

  virtual void waveExchange(){};
  virtual void spanwiseWaveReorder(){};
  void setTimeStep();
  void setLimiterVisc();
  void fixTimeStepTravelingWave();
  void exchangeTimeStep();
  void initializeRungeKutta();
  virtual void computeTimeStep(){};
  MBool isInInterval(MInt);
  MInt getNoCells() { return m_grid->m_noCells; };
  MInt noVariables() const override { return PV->noVariables; };
  MInt getNoActiveCells() { return m_grid->m_noActiveCells; };
  MInt* getActiveCells() { return m_grid->m_nActiveCells; };
  MInt* getOffsetCells() { return m_nOffsetCells; };
  MInt getNoGhostLayers() { return m_grid->m_noGhostLayers; };
  MInt getWaveAvrgInterval() { return (m_waveNoStepsPerCell); };
  MInt getWaveStepOffset() { return (m_movingGridStepOffset); };
  MInt* getCellGrid() { return m_grid->m_nCells; };
  MBool isMovingGrid() { return m_movingGrid; };
  MInt getGridMovingMethod() { return m_gridMovingMethod; };
  MInt getBodyForceMethod() { return m_bodyForceMethod; };
  MBool isStreamwiseTravelingWave() { return m_streamwiseTravelingWave; };
  MBool isTravelingWave() { return m_travelingWave; };

  StructuredGrid<nDim>* getGrid() { return m_grid; };

  MFloat getGamma() { return m_gamma; };
  MFloat getSutherlandConstant() { return m_sutherlandConstant; };
  MFloat getRe0() { return m_Re0; };
  MFloat getMa() { return m_Ma; };
  MPI_Comm getCommunicator() { return m_StructuredComm; };
  virtual void computeCumulativeAverage(MBool){};


  virtual void loadSampleFile(MString){};
  virtual void getSampleVariables(MInt, MFloat*){};
  MFloat getPV(MInt var, MInt cellId) { return m_cells->pvariables[var][cellId]; }
  void setPV(MInt var, MInt cellId, MFloat value) { m_cells->pvariables[var][cellId] = value; }
  virtual MFloat getSampleVorticity(MInt, MInt) { return 0; };
  virtual MFloat dvardxyz(MInt, MInt, MFloat*) { return 0; };
  virtual MFloat dvardx(MInt, MFloat*) { return 0; };
  virtual void loadAverageRestartFile(const char*, MFloat**, MFloat**, MFloat**, MFloat**) {}
  virtual void loadAveragedVariables(const char*){};
  void convertRestartVariables(MFloat oldMa);
  virtual void convertRestartVariablesSTG(MFloat oldMa) {
    (void)oldMa;
    mTerm(-1, AT_, "not implemented for 0d,2d");
  };
  virtual bool rungeKuttaStep() = 0;
  virtual void viscousFlux() = 0;
  virtual void Muscl(MInt = -1) = 0;
  virtual void applyBoundaryCondition() = 0;
  virtual void initSolutionStep(MInt) = 0;
  virtual void initialCondition() = 0;

  void tred2(MFloatScratchSpace& A, MInt dim, MFloat* diag, MFloat* offdiag);
  void tqli2(MFloat* diag, MFloat* offdiag, MInt dim);
  void insertSort(MInt dim, MFloat* list);
  MFloat pythag(MFloat a, MFloat b);

  void resetRHS();
  void rhs();
  void rhsBnd();
  void lhsBnd();
  void initSolver() override;
  virtual MBool maxResidual() { return true; };
  MBool solutionStep() override;
  void preTimeStep() override {}
  void postTimeStep() override;
  void finalizeInitSolver() override;
  void cleanUp() override;
  virtual void updateSpongeLayer() = 0;

  MFloat time() const override { return m_time; }
  MInt determineRestartTimeStep() const override;
  MBool hasRestartTimeStep() const override { return true; }

  void init() { m_log << "Called Structured::init without Implementation" << std::endl; }

  inline std::array<MInt, nDim> beginP0() {
    IF_CONSTEXPR(nDim == 2) { return {0, 0}; }
    else {
      return {0, 0, 0};
    }
  }

  inline std::array<MInt, nDim> beginP1() {
    IF_CONSTEXPR(nDim == 2) { return {1, 1}; }
    else {
      return {1, 1, 1};
    }
  }

  inline std::array<MInt, nDim> beginP2() {
    IF_CONSTEXPR(nDim == 2) { return {2, 2}; }
    else {
      return {2, 2, 2};
    }
  }

  inline std::array<MInt, nDim> endM2() {
    IF_CONSTEXPR(nDim == 2) { return {m_nCells[1] - 2, m_nCells[0] - 2}; }
    else {
      return {m_nCells[2] - 2, m_nCells[1] - 2, m_nCells[0] - 2};
    }
  }

  inline std::array<MInt, nDim> endM1() {
    IF_CONSTEXPR(nDim == 2) { return {m_nCells[1] - 1, m_nCells[0] - 1}; }
    else {
      return {m_nCells[2] - 1, m_nCells[1] - 1, m_nCells[0] - 1};
    }
  }

  inline std::array<MInt, nDim> endM0() {
    IF_CONSTEXPR(nDim == 2) { return {m_nCells[1], m_nCells[0]}; }
    else {
      return {m_nCells[2], m_nCells[1], m_nCells[0]};
    }
  }


  // if zonal then this communicator is the subworld of each solver
  MPI_Comm m_StructuredComm;

  MInt m_restartTimeStep;
  MString m_outputFormat;
  MInt m_lastOutputTimeStep;

 protected:
  // variables
  std::unique_ptr<MConservativeVariables<nDim>> CV;
  std::unique_ptr<MPrimitiveVariables<nDim>> PV;

  // epsilon
  MFloat m_eps;

  // left/right States
  MFloat* m_QLeft = nullptr;
  MFloat* m_QRight = nullptr;

  // moving grids:
  MBool m_movingGrid;
  MBool m_mgExchangeCoordinates;
  MInt m_gridMovingMethod = 0;
  MInt m_movingGridStepOffset;
  MBool m_synchronizedMGOutput;
  MInt m_waveNoStepsPerCell;
  MFloat m_wallVel;
  MFloat m_movingGridTimeOffset;
  MBool m_movingGridSaveGrid;
  MBool m_travelingWave;
  MBool m_streamwiseTravelingWave;
  MFloat m_waveLengthPlus;
  MFloat m_waveAmplitudePlus;
  MFloat m_waveTimePlus;
  MFloat m_waveTime;
  MFloat m_waveLength;
  MFloat m_waveAmplitude;
  MFloat m_waveAmplitudeSuction;
  MFloat m_waveAmplitudePressure;
  MFloat m_waveGradientSuction;
  MFloat m_waveGradientPressure;
  MFloat m_waveSpeed;
  MFloat m_waveSpeedPlus;
  MFloat m_waveBeginTransition;
  MFloat m_waveEndTransition;
  MFloat m_waveOutBeginTransition;
  MFloat m_waveOutEndTransition;
  MFloat m_wavePressureBeginTransition;
  MFloat m_wavePressureEndTransition;
  MFloat m_wavePressureOutBeginTransition;
  MFloat m_wavePressureOutEndTransition;
  MFloat m_waveYBeginTransition;
  MFloat m_waveYEndTransition;
  MFloat m_waveAngle;
  MFloat m_waveTemporalTransition;
  MBool m_movingGridInitialStart;
  MBool m_waveRestartFadeIn;
  MInt m_waveTimeStepComputed;
  MInt m_waveCellsPerWaveLength;
  MFloat m_wavePenetrationHeight;
  MInt m_bodyForceMethod = 0;
  MBool m_bodyForce;
  MFloat* m_waveForceField = nullptr;
  MFloat* m_waveForceY = nullptr;
  MFloat* m_waveForceZ = nullptr;
  MString m_waveForceFieldFile;
  MFloat m_waveDomainWidth;

  MFloat m_oscAmplitude;
  MFloat m_oscSr;
  MFloat m_oscFreq;


  MFloat* m_rhs = nullptr;

  MBool m_dsIsComputed;

  // for IO
  MBool m_useNonSpecifiedRestartFile;
  MInt m_outputOffset;
  MBool m_vorticityOutput;
  MBool m_debugOutput;
  MInt m_outputIterationNumber;
  MBool m_sampleSolutionFiles;

  // structure for cell data
  StructuredCell* m_cells = nullptr;
  MString* m_variableNames = nullptr;
  MString* m_pvariableNames = nullptr;

  // for debugging only
  MFloat** pointProperties = nullptr;
  // end debugging

  MBool m_ignoreUID;

  std::unique_ptr<FvStructuredSolverWindowInfo<nDim>> m_windowInfo; // contains info about the window information
  MInt m_noGhostLayers;                                 // number of GhostLayerst to be added
  MInt* m_nPoints = nullptr;       // stores the maximum dimension of the partition with ghost points
  MInt* m_nActivePoints = nullptr; // stores the  maximum dimension of the partition without ghost points
  MInt* m_nOffsetPoints = nullptr;
  MInt m_noCells;                  // stores number of structured cells
  MInt m_noActiveCells;
  MInt m_noPoints; // gridpoints in the partition

  MInt m_noSurfaces;
  //  MInt       m_noActiveSurfaces;
  MFloat m_referenceLength;
  MFloat m_physicalReferenceLength;
  MFloat m_Pr;
  MFloat m_rPr;
  MFloat m_cfl;
  MInt m_orderOfReconstruction;
  MFloat m_inflowTemperatureRatio;
  MBool m_considerVolumeForces;
  MBool m_euler;
  MFloat* m_volumeForce = nullptr;
  MInt m_volumeForceMethod = 0;
  MInt m_volumeForceUpdateInterval;
  MFloat m_gamma;
  MFloat m_gammaMinusOne;
  MFloat m_fgammaMinusOne;
  MFloat m_Re0;
  MFloat m_ReTau;
  MFloat* m_angle = nullptr;
  MInt m_periodicConnection;
  MInt m_channelFlow;
  MFloat m_sutherlandConstant;
  MFloat m_sutherlandPlusOne;
  MFloat m_TinfS;
  MBool m_computeLambda2;


  //
  MFloat m_hInfinity;
  MFloat m_referenceEnthalpy;

  // 2-eq RANS models
  MString m_rans2eq_mode;
  //
  MBool m_keps_nonDimType;
  MBool m_solutionAnomaly;
  MFloat m_I;        // turb intensity at inflow
  MFloat m_epsScale; // either turb length scale or viscosity ratio
  MInt m_kepsICMethod;

  std::unique_ptr<StructuredFQVariables> FQ;
  MInt m_maxNoVariables;

  MBool m_bForce;
  MBool m_bPower;
  MFloat* m_forceCoef = nullptr;
  MBool m_detailAuxData;
  MBool m_bForceLineAverage;
  std::vector<MString> m_forceHeaderNames;
  MInt m_forceAveragingDir;
  MInt m_forceOutputInterval;
  MInt m_forceAsciiOutputInterval;
  MInt m_forceAsciiComputeInterval;
  MFloat** m_forceData = nullptr;
  MInt m_forceCounter;
  MInt m_lastForceOutputTimeStep;
  MInt m_lastForceComputationTimeStep;
  MInt m_noForceDataFields;
  MBool m_forceSecondOrder;
  MFloat m_globalDomainWidth;
  MBool m_auxDataCoordinateLimits;
  MFloat* m_auxDataLimits = nullptr;
  MBool m_setLocalWallDistance = false;

  MInt m_intpPointsOutputInterval;
  MInt m_intpPointsNoLines;
  MInt m_intpPointsNoLines2D;
  MInt m_intpPointsNoPointsTotal;
  MInt* m_intpPointsNoPoints = nullptr;
  MInt* m_intpPointsNoPoints2D = nullptr;
  MInt* m_intpPointsOffsets = nullptr;
  MFloat** m_intpPointsStart = nullptr;
  MFloat** m_intpPointsDelta = nullptr;
  MFloat** m_intpPointsDelta2D = nullptr;
  MFloat** m_intpPointsCoordinates = nullptr;
  MInt* m_intpPointsHasPartnerGlobal = nullptr;
  MInt* m_intpPointsHasPartnerLocal = nullptr;
  MFloat** m_intpPointsVarsGlobal = nullptr;
  MFloat** m_intpPointsVarsLocal = nullptr;
  MBool m_intpPoints;

  MInt m_boxNoBoxes;
  MInt m_boxOutputInterval;
  MInt* m_boxBlock = nullptr;
  MInt** m_boxOffset = nullptr;
  MInt** m_boxSize = nullptr;
  MBool m_boxWriteCoordinates;

  std::unique_ptr<StructuredInterpolation<nDim>> m_nodalBoxInterpolation;
  MInt m_nodalBoxNoBoxes;
  MString m_nodalBoxOutputDir;
  MInt m_nodalBoxOutputInterval;
  MInt* m_nodalBoxBlock = nullptr;
  MInt** m_nodalBoxOffset = nullptr;
  MInt** m_nodalBoxPoints = nullptr;
  MInt** m_nodalBoxLocalOffset = nullptr;
  MInt** m_nodalBoxLocalPoints = nullptr;
  MInt** m_nodalBoxLocalDomainOffset = nullptr;
  MInt* m_nodalBoxLocalSize = nullptr;
  MInt* m_nodalBoxPartnerLocal = nullptr;
  MFloat** m_nodalBoxCoordinates = nullptr;
  MFloat** m_nodalBoxVariables = nullptr;
  MBool m_nodalBoxWriteCoordinates;
  MInt m_nodalBoxTotalLocalSize;
  MInt* m_nodalBoxTotalLocalOffset = nullptr;
  MBool m_nodalBoxInitialized;

  MInt m_pointsToAsciiComputeInterval;
  MInt m_pointsToAsciiOutputInterval;
  MInt m_pointsToAsciiNoPoints;
  MInt m_pointsToAsciiLastOutputStep;
  MInt m_pointsToAsciiLastComputationStep;
  MFloat** m_pointsToAsciiCoordinates = nullptr;
  MFloat** m_pointsToAsciiVars = nullptr;
  MInt m_pointsToAsciiVarId;
  MInt m_pointsToAsciiCounter;
  MInt* m_pointsToAsciiHasPartnerGlobal = nullptr;
  MInt* m_pointsToAsciiHasPartnerLocal = nullptr;
  std::unique_ptr<StructuredInterpolation<nDim>> m_pointsToAsciiInterpolation;

  //
  MFloat m_referenceTemperature;
  MInt m_noSpecies;
  MFloat* m_referenceComposition = nullptr;
  MFloat* m_formationEnthalpy = nullptr;
  MInt m_noRKSteps;
  MFloat* m_RKalpha = nullptr;
  MFloat m_time;
  MInt m_RKStep;
  MFloat m_timeStep;
  MBool m_localTimeStep;
  MInt m_rungeKuttaOrder;
  MInt m_timeStepMethod;
  MInt m_timeStepComputationInterval;
  MFloat m_timeRef;
  MInt m_dualTimeStepping;
  MFloat m_physicalTimeStep;
  MFloat m_physicalTime;

  // Timers
  // Timer group which holds all solver-wide timers
  MInt m_timerGroup = -1;
  // Stores all solver-wide timers
  std::array<MInt, Timers::_count> m_timers{};

  // constant Timestepping
  MBool m_constantTimeStep;

  // rescaling
  MInt m_rescalingCommGrRoot = -1;
  MInt m_rescalingCommGrRootGlobal = -1;
  MPI_Comm m_rescalingCommGrComm = MPI_COMM_NULL;

  // synthetic turbulence generation
  // for synthetic turbulence generation
  MPI_Comm m_commStg = MPI_COMM_NULL;
  MInt m_commStgRoot = -1;
  MInt m_commStgRootGlobal = -1;
  MInt m_commStgMyRank = -1;
  MBool m_stgIsActive;
  MFloat m_stgBLT1;
  MFloat m_stgBLT2;
  MFloat m_stgBLT3;
  MFloat m_stgDelta99Inflow;
  MBool m_stgInitialStartup;
  MInt m_stgNoEddieProperties;
  MFloat** m_stgEddies = nullptr;
  MInt m_stgNoEddies;
  MInt m_stgMaxNoEddies;
  MFloat m_stgExple;
  MFloat m_stgEddieDistribution;
  MInt m_stgBoxSize[3];
  MBool m_stgLocal;
  MBool m_stgCreateNewEddies;
  MBool m_stgRootRank;
  MInt m_stgNoVariables;
  MInt m_stgShapeFunction;
  MBool m_stgEddieLengthScales;
  MBool m_stgSubSup;
  MBool m_stgSupersonic;
  MInt m_stgFace;
  MFloat* m_stgLengthFactors = nullptr;
  MFloat* m_stgRSTFactors = nullptr;
  MInt m_stgMyRank;

  // global Upwind Coefficient;
  MFloat m_chi;

  // Sponge Variables
  MInt m_noSpongeDomainInfos;
  MInt* m_spongeBcWindowInfo = nullptr;
  MBool m_useSponge;
  MInt m_spongeLayerType;
  MFloat* m_sigmaSponge = nullptr;
  MFloat* m_betaSponge = nullptr;
  MFloat* m_spongeLayerThickness = nullptr;
  MFloat m_targetDensityFactor;
  MBool m_computeSpongeFactor;

  // Workload
  MFloat m_workload;
  MFloat m_workloadIncrement;

  // Residual
  // MPI_derived datatypes
  typedef struct {
    MFloat maxRes;
    MFloat avrgRes;
    MInt* maxCellIndex = nullptr;
  } MRes;

  MRes* m_residualSnd = nullptr;
  MRes m_residualRcv;

  MInt m_residualOutputInterval;

  MFloat m_avrgResidual;
  MFloat m_firstMaxResidual;
  MFloat m_firstAvrgResidual;
  MLong m_totalNoCells;
  MBool m_residualFileExist;
  MPI_Op m_resOp;
  MPI_Datatype m_mpiStruct;
  FILE* m_resFile;

  // Convergence
  MBool m_convergence;
  MFloat m_convergenceCriterion;

  // Restart & initial Condition
  MBool m_restart;
  MInt m_initialCondition;
  MFloat m_deltaP;
  MBool m_restartInterpolation;

  // Strings for Output specification
  MString m_solutionFileName;
  MString m_boxOutputDir;
  MString m_intpPointsOutputDir;
  MString m_auxOutputDir;
  MBool m_savePartitionOutput;

  MInt* m_nCells = nullptr;       // cell array dimension with ghost layer
  MInt* m_nActiveCells = nullptr; // cell array dimension without ghost layers
  MInt* m_nOffsetCells = nullptr;
  MInt m_blockId;
  MInt m_noBlocks;

  // For Boundary Conditions
  MInt* m_noWindows = nullptr;     // contains the number of windows of each face
  MInt** m_bndryCndInfo = nullptr; // contains start and length of each window
  MInt** m_bndryCnd = nullptr;     // contains the boundary condition

  // limiter
  MBool m_limiter;         // for limiter choice
  MString m_limiterMethod; // reads the limiter from the properties
  MFloat m_venkFactor;     // customizable factor for the modified Venkatakrishnan Limiter
  MBool m_limiterVisc;
  MFloat m_CFLVISC;

  MString m_musclScheme;
  MString m_ausmScheme;

  MBool m_viscCompact;

  // zonal
  MBool m_zonal;
  MInt m_rans;
  RansMethod m_ransMethod = NORANS;
  MInt m_noRansEquations;
  MString m_zoneType;
  MFloat m_ransTransPos;
  MBool m_hasSTG;
  MInt m_zonalExchangeInterval;
  std::vector<MFloat*> m_zonalSpanwiseAvgVars;

  MPI_Comm m_commBC2600 = MPI_COMM_NULL;
  MInt m_commBC2600Root = -1;
  MInt m_commBC2600RootGlobal = -1;
  MInt m_commBC2600MyRank = -1;

  MPI_Comm* m_commZonal = nullptr;
  MInt* m_commZonalRoot = nullptr;
  MInt* m_commZonalRootGlobal = nullptr;
  MInt m_commZonalMyRank = -1;
  MBool m_zonalRootRank = false;


  // zonal cumulative averaging
  MBool m_zonalExponentialAveraging;
  MInt m_zonalStartAvgTime;
  MFloat m_zonalAveragingFactor;


  MFloat m_rhoNuTildeInfinty;
  MFloat m_nutInfinity;
  MFloat m_mutInfinity;

  // parallel
  std::vector<std::unique_ptr<StructuredComm<nDim>>> m_sndComm;
  std::vector<std::unique_ptr<StructuredComm<nDim>>> m_rcvComm;

  MInt m_currentPeriodicDirection;

  std::vector<std::unique_ptr<StructuredComm<nDim>>> m_waveSndComm;
  std::vector<std::unique_ptr<StructuredComm<nDim>>> m_waveRcvComm;

  MInt m_noNghbrDomains;
  MInt* m_nghbrDomainId = nullptr;
  MInt* m_noNghbrDomainBufferSize = nullptr;
  MFloat** m_bufferCellsSndRcv = nullptr;   // comunicator cells==>cells to be communicated
  MFloat** m_bufferPointsSendRcv = nullptr; // comunicator points ==> points to be communicated
  MInt* m_nghbrFaceId = nullptr;
  MInt** m_nghbrFaceInfo = nullptr;
  MPI_Request* mpi_sndRequest = nullptr;
  MPI_Request* mpi_rcvRequest = nullptr;
  MPI_Status* mpi_sndRcvStatus = nullptr;

  // plenum boundary condition
  MPI_Comm m_plenumComm = MPI_COMM_NULL;
  MInt m_plenumRoot = -1;

  // channel boundary condition
  MPI_Comm m_commChannelIn = MPI_COMM_NULL;
  MPI_Comm m_commChannelOut = MPI_COMM_NULL;
  MPI_Comm m_commChannelWorld = MPI_COMM_NULL;
  MInt* m_channelRoots = nullptr;
  MFloat m_channelPresInlet;
  MFloat m_channelPresOutlet;
  MFloat m_channelLength;
  MFloat m_channelHeight;
  MFloat m_channelWidth;
  MFloat m_channelInflowPlaneCoordinate;
  MFloat m_channelC1;
  MFloat m_channelC2;
  MFloat m_channelC3;
  MFloat m_channelC4;
  MBool m_channelFullyPeriodic;
  MFloat m_inflowVelAvg;

  // for the periodic rotation Boundary
  MPI_Comm m_commPerRotOne = MPI_COMM_NULL;
  MPI_Comm m_commPerRotTwo = MPI_COMM_NULL;
  MPI_Comm m_commPerRotWorld = MPI_COMM_NULL;
  MInt* m_commPerRotRoots = nullptr;
  MInt m_commPerRotGroup = -1;

  // profile BC 2600
  MBool m_bc2600IsActive;
  MBool m_bc2600InitialStartup;
  MFloat** m_bc2600Variables = nullptr;
  MBool m_bc2600;
  MInt* m_bc2600noCells = nullptr;
  MInt* m_bc2600noActiveCells = nullptr;
  MInt* m_bc2600noOffsetCells = nullptr;
  MInt m_bc2600RootRank;
  MInt m_bc2600Face;

  // effective boundary condition 2601
  MBool m_bc2601IsActive;
  MBool m_bc2601InitialStartup;
  MFloat** m_bc2601Variables = nullptr;
  MFloat** m_bc2601ZerothOrderSolution = nullptr;
  MBool m_bc2601;
  MInt* m_bc2601noCells = nullptr;
  MInt* m_bc2601noActiveCells = nullptr;
  MInt* m_bc2601noOffsetCells = nullptr;
  MFloat m_bc2601GammaEpsilon;

  // convective solution output
  MBool m_useConvectiveUnitWrite;
  MFloat m_convectiveUnitInterval;
  MInt m_noConvectiveOutputs;

  // timer
  void initTimers();

  // Initialization status
  MBool m_isInitTimers = false;

  // infinity Values
  MFloat m_UInfinity;
  MFloat m_VInfinity;
  MFloat m_WInfinity;
  MFloat m_PInfinity;
  MFloat m_TInfinity;
  MFloat m_DthInfinity;
  MFloat m_muInfinity;
  MFloat m_DInfinity;
  MFloat m_VVInfinity[3]; // 3 is the max value!!!!
  MFloat m_rhoUInfinity;
  MFloat m_rhoVInfinity;
  MFloat m_rhoWInfinity;
  MFloat m_rhoEInfinity;
  MFloat m_rhoInfinity;
  MFloat m_rhoVVInfinity[3];

  MBool m_changeMa;

  // for modes:
  MInt m_restartBc2800;
  MFloat m_restartTimeBc2800;
  MFloat* m_adiabaticTemperature = nullptr;
  MInt m_useAdiabaticRestartTemperature;

  // singularity
  SingularInformation* m_singularity = nullptr;
  MInt m_hasSingularity;

  // sandpaper tripping
  MBool m_useSandpaperTrip;
  MInt m_tripNoTrips;
  MFloat* m_tripDelta1 = nullptr;
  MFloat* m_tripXOrigin = nullptr;
  MFloat* m_tripXLength = nullptr;
  MFloat* m_tripYOrigin = nullptr;
  MFloat* m_tripYHeight = nullptr;
  MFloat* m_tripCutoffZ = nullptr;
  MFloat* m_tripMaxAmpSteady = nullptr;
  MFloat* m_tripMaxAmpFluc = nullptr;
  MInt m_tripNoModes;
  MFloat* m_tripDeltaTime = nullptr;
  MInt* m_tripTimeStep = nullptr;
  MInt m_tripSeed;
  MFloat* m_tripG = nullptr;
  MFloat* m_tripH1 = nullptr;
  MFloat* m_tripH2 = nullptr;
  MFloat* m_tripModesG = nullptr;
  MFloat* m_tripModesH1 = nullptr;
  MFloat* m_tripModesH2 = nullptr;
  MFloat* m_tripCoords = nullptr;
  MInt m_tripNoCells;
  MFloat m_tripDomainWidth;
  MBool m_tripUseRestart;
  MBool m_tripAirfoil;

  MFloat* m_airfoilCoords = nullptr;
  MFloat* m_airfoilNormalVec = nullptr;
  MInt m_airfoilNoWallPoints;
  MFloat* m_airfoilWallDist = nullptr;

  // Porous stuff
  MBool m_porous;
  MString m_blockType;
  //  MInt m_porousID;
  std::vector<MInt> m_porousBlockIds;
  MFloat m_c_Dp, m_c_Dp_eps, m_c_wd, m_c_t, m_c_eps;

  // FSC flow
  MInt m_fsc;
  MFloat* m_fsc_eta;
  MFloat* m_fsc_fs;
  MFloat* m_fsc_f;
  MFloat* m_fsc_g;
  MInt m_fsc_noPoints;
  MFloat m_fsc_x0;
  MFloat m_fsc_y0;
  MFloat m_fsc_dx0;
  MFloat m_fsc_m;
  MFloat m_fsc_Re;
  virtual void initFsc();
  virtual MFloat getFscPressure(MInt cellId);
  virtual MFloat getFscPressure(MFloat coordX);
  virtual MFloat getFscEta(MFloat coordX, MFloat coordY);
  virtual void getFscVelocity(MInt cellId, MFloat* const vel);
  virtual void getFscVelocity(MFloat coordX, MFloat coordY, MFloat* const vel);


  // Blasius boundary layer
  MBool m_useBlasius;
  MFloat* m_blasius_eta;
  MFloat* m_blasius_f;
  MFloat* m_blasius_fp;
  MInt m_blasius_noPoints;
  MFloat m_blasius_x0;
  MFloat m_blasius_y0;
  MFloat m_blasius_dx0;

  virtual void initBlasius();
  virtual MFloat getBlasiusEta(MFloat coordX, MFloat coordY);
  virtual void getBlasiusVelocity(MInt cellId, MFloat* const vel);
  virtual void getBlasiusVelocity(MFloat coordX, MFloat coordY, MFloat* const vel);

 private:
  /// Convergence status of the current time step
  MBool m_timeStepConverged = false;

  MBool m_isActive = false;
  /// PostProcessingSolver interface:
  virtual void initStructuredPostprocessing() {
    StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::initStructuredPostprocessing();
  }
  virtual void saveAverageRestart() { StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>::saveAverageRestart(); }

  std::unique_ptr<MLCoupling<float, MFloat, float, float>> m_mlCoupler;
  std::vector<float> m_mlInputBuf[3];
  std::vector<MFloat> m_mlOutputBuf[3];
  // When true, update ghost cells / boundary conditions immediately after ML
  // inference injection. Physically more correct, but breaks bitwise agreement
  // with the legacy AIX CMI (which skipped this step on inference steps).
  // Default: false (legacy-compatible). Enable via mlUpdateGhostAfterInference=1
  // in properties.toml.
  MBool m_mlUpdateGhostAfterInference = false;
};

#endif
