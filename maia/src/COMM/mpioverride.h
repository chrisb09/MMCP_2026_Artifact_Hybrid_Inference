// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

#ifndef MAIA_MPI_OVERRIDE_H
#define MAIA_MPI_OVERRIDE_H

#include "mpi.h"

/* MPI-Wrapper manual:
 *
 * The following wrapper functions serve the purpose to allow easier debugging
 * of MPI code. They "wrap" around existing MPI routines, and catch potential
 * errors returned by them. Furthermore, using the MAIA_MPI_DEBUG-option in
 * CONFIG.H, they give insight about some information (function-dependent),
 * such as where in the code the function was called, which variables where
 * handed over, which communicator was it on, ...
 *
 * The current implementation covers all functions currently existing in the
 * MAIA code (as of September 2019). Additionally, deprecated-calls to several other
 * MPI-routines are implemented, such that whenever someone uses a new MPI
 * function (that returns an error value or requires communication, i.e.
 * not MPI_Wtime), this enforces the programmer to write a new wrapper function.
 *
 * In general, if you use a MPI function that has not yet been used before in the
 * code, please also write a corresponding wrapper function and add an deprecated
 * call if not yet existing. Some deprecated calls are only implemented at a
 * random basis to cover most function classes without writing a deprecated call
 * for every single function. Thus, if you e.g. use one-sided communication,
 * encounter a deprecated call for MPI_Get, and also use several other new methods
 * that do not have a deprecated call and/or wrapper function yet, please create
 * the corresponding call and wrapper function as well.
 *
 * Therefore, you can find brief instructions here on how to to this:
 *
 * >> If you use a yet existing MPI function:
 *    Call the wrapper function in the code instead of the original one.
 *
 * >> If you use a MPI function, that has not been used before in MAIA:
 * Most conveniently, copy an existing (if possible, similar) wrapper function
 * in MPIOVERRIDE.CPP(i.e. if you implement Allgatherw, use Allgatherv etc.).
 * The function call corresponds exactly to the original call, with a few extra
 * variables:
 *     - every wrapper should have "const MString& name", which is handed over
 *       the "AT_"-macro
 *     - if data is transferred, you can hand over the variable names
 *       (as in varname, sndvarname, rcvvarname)
 *     - feel free to hand over additional information if you find it approbiate
 * Now inside, make sure that the m_log-calls are adjusted to the new wrapper,
 * call the original MPI function using the arguments received by the wrapper,
 * and handle "result" according to the error values that can arise for the new
 * function (online).
 * Apart from the error values, use ONLY the OpenMPI-documentation as a reference,
 * as different MPI implementations might handle some variables differently,
 * and your code might not compile.
 *
 * Furthermore, make sure, that your new wrapper is declared in the following
 * here in MPIOVERRIDE.H.
 *
 * At last, ensure, that you call only wrapper function in the code.
 */


// Communicator-related
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm, const MString& name, const MString& varname);
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm, const MString& name, const MString& varname);
int MPI_Comm_free(MPI_Comm* comm, const MString& name, const MString& varname);
int MPI_Comm_group(MPI_Comm comm, MPI_Group* group, const MString& name, const MString& varname);

// Point-to-point communication
int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, const MString& name,
             const MString& varname);
int MPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request,
              const MString& name, const MString& varname);
int MPI_Issend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
               MPI_Request* request, const MString& name, const MString& varname);
int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status,
             const MString& name, const MString& varname);
int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request,
              const MString& name, const MString& varname);
int MPI_Send_init(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                  MPI_Request* request, const MString& name, const MString& varname);
int MPI_Recv_init(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request,
                  const MString& name, const MString& varname);

// Wait - Test
int MPI_Wait(MPI_Request* request, MPI_Status* status, const MString& name);
int MPI_Waitall(int count, MPI_Request* request, MPI_Status* status, const MString& name);
int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int* outcount, int array_of_indices[],
                 MPI_Status array_of_statuses[], const MString& name);
int MPI_Test(MPI_Request* request, int* flag, MPI_Status* status, const MString& name);

// Collectives
int MPI_Barrier(MPI_Comm comm, const MString& name);
int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
               const MString& name, const MString& sndvarname, const MString& rcvvarname);
int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                  const MString& name, const MString& sndvarname, const MString& rcvvarname);
int MPI_Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                   MPI_Request* request, const MString& name, const MString& sndvarname, const MString& rcvvarname);
int MPI_Scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                MPI_Datatype recvtype, int root, MPI_Comm comm, const MString& name, const MString& sndvarname,
                const MString& rcvvarname);
int MPI_Scatterv(const void* sendbuf, const int sendcount[], const int displs[], MPI_Datatype sendtype, void* recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, const MString& name,
                 const MString& sndvarname, const MString& rcvvarname);
int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, const MString& name,
              const MString& varname);
int MPI_Ibcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request* request,
               const MString& name, const MString& varname);
int MPI_Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
               MPI_Datatype recvtype, int root, MPI_Comm comm, const MString& name, const MString& sndvarname,
               const MString& rcvvarname);
int MPI_Gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
                const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, const MString& name,
                const MString& sndvarname, const MString& rcvvarname);
int MPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                  MPI_Datatype recvtype, MPI_Comm comm, const MString& name, const MString& sndvarname,
                  const MString& rcvvarname);
int MPI_Allgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
                   const int displs[], MPI_Datatype recvtype, MPI_Comm comm, const MString& name,
                   const MString& sndvarname, const MString& rcvvarname);
int MPI_Alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                 MPI_Datatype recvtype, MPI_Comm comm, const MString& name, const MString& sndvarname,
                 const MString& rcvvarname);
int MPI_Alltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype,
                  void* recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                  const MString& name, const MString& sndvarname, const MString& rcvvarname);
int MPI_Exscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
               const MString& name, const MString& sndvarname, const MString& rcvvarname);

// MPI Datatypes
int MPI_Type_commit(MPI_Datatype* datatype, const MString& name);
int MPI_Type_free(MPI_Datatype* datatype, const MString& name);
int MPI_Type_create_hindexed(int count, const int array_of_solverlengths[], const MPI_Aint array_of_displacements[],
                             MPI_Datatype oldtype, MPI_Datatype* newtype, const MString& name);
int MPI_Type_contiguous(int count, MPI_Datatype old_type, MPI_Datatype* new_type_p, const MString& name);

// MPI Group
int MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group* newgroup, const MString& name);
int MPI_Group_free(MPI_Group* group, const MString& name);


// MISC
int MPI_Start(MPI_Request* request, const MString& name);
int MPI_Startall(int count, MPI_Request array_of_requests[], const MString& name);
int MPI_Get_count(const MPI_Status* status, MPI_Datatype datatype, int* count, const MString& name);
int MPI_Get_address(const void* location, MPI_Aint* address, const MString& name);
int MPI_Abort(MPI_Comm comm, int errorcode, const MString& name);
int MPI_Request_free(MPI_Request* request, const MString& name);
int MPI_Cancel(MPI_Request* request, const MString& name);
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status* status, const MString& name);
int MPI_Iprobe(int source, int tag, MPI_Comm comm, int* flag, MPI_Status* status, const MString& name);

// MPI Info
int MPI_Info_create(MPI_Info* info, const MString& name);
int MPI_Info_free(MPI_Info* info, const MString& name);
int MPI_Info_get(MPI_Info info, const char* key, int valuelen, char* value, int* flag, const MString& name);
int MPI_Info_get_nthkey(MPI_Info info, int n, char* key, const MString& name);
int MPI_Info_get_nkeys(MPI_Info info, int* nkeys, const MString& name);
int MPI_Info_get_valuelen(MPI_Info info, const char* key, int* valuelen, int* flag, const MString& name);

// MPI File
int MPI_File_open(MPI_Comm comm, const char* filename, int amode, MPI_Info info, MPI_File* mpi_fh, const MString& name);
int MPI_File_seek(MPI_File mpi_fh, MPI_Offset offset, int whence, const MString& name);
int MPI_File_close(MPI_File* mpi_fh, const MString& name);
int MPI_File_write_shared(MPI_File mpi_fh, const void* buf, int count, MPI_Datatype datatype, MPI_Status* status,
                          const MString& name);
int MPI_File_iwrite_shared(MPI_File mpi_fh, const void* buf, int count, MPI_Datatype datatype, MPI_Request* request,
                           const MString& name);


#if defined(MAIA_GCC_COMPILER)
//-- GNU
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
//-- CLANG
#elif defined(MAIA_CLANG_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++14-extensions"
#endif

// Do not mark MPI functions deprecated for paraview plugin
#ifndef PVPLUGIN
#if defined(MAIA_GCC_COMPILER) || defined(MAIA_CLANG_COMPILER)
// Mark all native MPI function calls as deprecated to allow only the use of the MPI wrapper
// functions taking additional arguments to allow easier debugging and error checking.
// The deprecated keyword only works with the GCC and CLANG compiler

// Communicators
[[deprecated("Use the wrapper MPI_Comm_create(..., AT_, <communicatorName>) instead!")]] int
MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm);

[[deprecated("Use the wrapper MPI_Comm_split(..., AT_, <communicatorName>) instead!")]] int
MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm);

[[deprecated("Use the wrapper MPI_Comm_free(..., AT_, <communicatorName>) instead!")]] int
MPI_Comm_free(MPI_Comm* comm);

[[deprecated("Use the wrapper MPI_Comm_group(..., AT_, <communicatorName>) instead!")]] int
MPI_Comm_group(MPI_Comm comm, MPI_Group* group);

[[deprecated("Use the wrapper MPI_Comm_split_type(..., AT_, <...>) instead!")]] int
MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm* newcomm);

[[deprecated("Use the wrapper MPI_Comm_accept(..., AT_) instead!")]] int
MPI_Comm_accept(const char* port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm* newcomm);

[[deprecated("Use the wrapper MPI_Comm_call_errhandler(..., AT_) instead!")]] int
MPI_Comm_call_errhandler(MPI_Comm comm, int errorcode);

[[deprecated("Use the wrapper MPI_Comm_compare(..., AT_) instead!")]] int MPI_Comm_compare(MPI_Comm comm1,
                                                                                           MPI_Comm comm2, int* result);

[[deprecated("Use the wrapper MPI_Comm_connect(..., AT_) instead!")]] int
MPI_Comm_connect(const char* port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm* newcomm);

[[deprecated("Use the wrapper MPI_Comm_create_errhandler(..., AT_) instead!")]] int
MPI_Comm_create_errhandler(MPI_Comm_errhandler_function* function, MPI_Errhandler* errhandler);

[[deprecated("Use the wrapper MPI_Comm_create_group(..., AT_) instead!")]] int
MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm* newcomm);

[[deprecated("Use the wrapper MPI_Comm_create_keyval(..., AT_) instead!")]] int
MPI_Comm_create_keyval(MPI_Comm_copy_attr_function* comm_copy_attr_fn,
                       MPI_Comm_delete_attr_function* comm_delete_attr_fn, int* comm_keyval, void* extra_state);

[[deprecated("Use the wrapper MPI_Comm_disconnect(..., AT_) instead!")]] int MPI_Comm_disconnect(MPI_Comm* comm);

[[deprecated("Use the wrapper MPI_Comm_remote_group(..., AT_) instead!")]] int MPI_Comm_remote_group(MPI_Comm comm,
                                                                                                     MPI_Group* group);

[[deprecated("Use the wrapper MPI_Comm_remote_size(..., AT_) instead!")]] int MPI_Comm_remote_size(MPI_Comm comm,
                                                                                                   int* size);

// Point-to-point communication
[[deprecated("Use the wrapper MPI_Send(..., AT_) instead!")]] int MPI_Send(void* buf, int count, MPI_Datatype datatype,
                                                                           int dest, int tag, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Ssend(..., AT_) instead!")]] int
MPI_Ssend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Ssend_init(..., AT_) instead!")]] int MPI_Ssend_init(const void* buf, int count,
                                                                                       MPI_Datatype datatype, int dest,
                                                                                       int tag, MPI_Comm comm,
                                                                                       MPI_Request* request);

[[deprecated("Use the wrapper MPI_Isend(..., AT_) instead!")]] int
MPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Issend(..., AT_) instead!")]] int
MPI_Issend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Recv(..., AT_) instead!")]] int
MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status);

[[deprecated("Use the wrapper MPI_Irecv(..., AT_) instead!")]] int
MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Send_init(..., AT_) instead!")]] int
MPI_Send_init(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Recv_init(..., AT_) instead!")]] int
MPI_Recv_init(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Sendrecv(..., AT_) instead!")]] int
MPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void* recvbuf,
             int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status* status);

[[deprecated("Use the wrapper MPI_Sendrecv_replace(..., AT_) instead!")]] int
MPI_Sendrecv_replace(void* buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag,
                     MPI_Comm comm, MPI_Status* status);

[[deprecated("Use the wrapper MPI_Bsend(..., AT_) instead!")]] int
MPI_Bsend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Ibsend(..., AT_) instead!")]] int
MPI_Ibsend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Bsend_init(..., AT_) instead!")]] int MPI_Bsend_init(const void* buf, int count,
                                                                                       MPI_Datatype datatype, int dest,
                                                                                       int tag, MPI_Comm comm,
                                                                                       MPI_Request* request);

[[deprecated("Use the wrapper MPI_Mrecv(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Mrecv(void* buf, int count, MPI_Datatype type, MPI_Message* message, MPI_Status* status);

[[deprecated("Use the wrapper MPI_Imrecv(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Imrecv(void* buf, int count, MPI_Datatype type, MPI_Message* message, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Rsend(..., AT_) instead!")]] int
MPI_Rsend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);


// Wait - Test
[[deprecated("Use the wrapper MPI_Wait(..., AT_) instead!")]] int MPI_Wait(MPI_Request* request, MPI_Status* status);

[[deprecated("Use the wrapper MPI_Waitall(..., AT_) instead!")]] int MPI_Waitall(int count, MPI_Request* request,
                                                                                 MPI_Status* status);

[[deprecated("Use the wrapper MPI_Waitsome(..., AT_) instead!")]] int
MPI_Waitsome(int incount, MPI_Request array_of_requests[], int* outcount, int array_of_indices[],
             MPI_Status array_of_statuses[]);

[[deprecated("Use the wrapper MPI_Waitany(..., AT_) instead!")]] int
MPI_Waitany(int count, MPI_Request array_of_requests[], int* index, MPI_Status* status);

[[deprecated("Use the wrapper MPI_Test(..., AT_) instead!")]] int MPI_Test(MPI_Request* request, int* flag,
                                                                           MPI_Status* status);

[[deprecated("Use the wrapper MPI_Test_cancelled(..., AT_) instead!")]] int MPI_Test_cancelled(const MPI_Status* status,
                                                                                               int* flag);

[[deprecated("Use the wrapper MPI_Testany(..., AT_) instead!")]] int
MPI_Testany(int count, MPI_Request array_of_requests[], int* index, int* flag, MPI_Status* status);

[[deprecated("Use the wrapper MPI_Testall(..., AT_) instead!")]] int
MPI_Testall(int count, MPI_Request array_of_requests[], int* flag, MPI_Status array_of_statuses[]);

[[deprecated("Use the wrapper MPI_Testsome(..., AT_) instead!")]] int
MPI_Testsome(int incount, MPI_Request array_of_requests[], int* outcount, int array_of_indices[],
             MPI_Status array_of_statuses[]);


// Collectives
[[deprecated("Use the wrapper MPI_Barrier(..., AT_) instead!")]] int MPI_Barrier(MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Ibarrier(..., AT_) instead!")]] int MPI_Ibarrier(MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Reduce(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Reduce_local(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Reduce_local(const void* inbuf, void* inoutbuf, int count, MPI_Datatype datatype, MPI_Op op);

[[deprecated("Use the wrapper MPI_Reduce_scatter(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Reduce_scatter(const void* sendbuf, void* recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op,
                   MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Ireduce(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Ireduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
            MPI_Request* request);

[[deprecated("Use the wrapper MPI_Ireduce_scatter(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Ireduce_scatter(const void* sendbuf, void* recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op,
                    MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Allreduce(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Iallreduce(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
               MPI_Request* request);

[[deprecated("Use the wrapper MPI_Scatter(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
            MPI_Datatype recvtype, int root, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Scatterv(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Scatterv(const void* sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void* recvbuf,
             int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Iscatter(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Iscatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
             MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Iscatterv(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Iscatterv(const void* sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void* recvbuf,
              int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Reduce_scatter_solver(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Reduce_scatter_solver(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op,
                          MPI_Comm comm);

[[deprecated(
    "Use the wrapper MPI_Ireduce_scatter_solver(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Ireduce_scatter_solver(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op,
                           MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Bcast(..., AT_, <name of buffer>) instead!")]] int
MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Ibcast(..., AT_, <name of buffer>) instead!")]] int
MPI_Ibcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Gather(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
           MPI_Datatype recvtype, int root, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Igather(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Igather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
            MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Gatherv(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
            const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Gatherv(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Igatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
             const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Allgather(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
              MPI_Datatype recvtype, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Iallgather(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Iallgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
               MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Allgatherv(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Allgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
               const int displs[], MPI_Datatype recvtype, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Iallgatherv(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Iallgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
                const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request);


[[deprecated("Use the wrapper MPI_Alltoall(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
             MPI_Datatype recvtype, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Ialltoall(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Ialltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
              MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Alltoallv(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Alltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void* recvbuf,
              const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Ialltoallv(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Ialltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void* recvbuf,
               const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Alltoallw(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Alltoallw(const void* sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[],
              void* recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[],
              MPI_Comm comm);

[[deprecated("Use the wrapper MPI_Ialltoallw(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Ialltoallw(const void* sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[],
               void* recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[],
               MPI_Comm comm, MPI_Request* request);

[[deprecated("Use the wrapper MPI_Scan(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Scan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

[[deprecated("Use the wrapper Iscan(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Iscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
          MPI_Request* request);

[[deprecated("Use the wrapper MPI_Exscan(..., AT_, <name off sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Exscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);


[[deprecated("Use the wrapper MPI_Neighbor_allgather(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Neighbor_allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                       MPI_Datatype recvtype, MPI_Comm comm);

[[deprecated("Use the wrapper MPIX_Allgather_init(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPIX_Allgather_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                    MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request);


// MPI Datatypes
[[deprecated("Use the wrapper MPI_Type_commit(..., AT_) instead!")]] int MPI_Type_commit(MPI_Datatype* datatype);

[[deprecated("Use the wrapper MPI_Type_free(..., AT_) instead!")]] int MPI_Type_free(MPI_Datatype* datatype);

[[deprecated("Use the wrapper MPI_Type_create_hindexed(..., AT_) instead!")]] int
MPI_Type_create_hindexed(int count, const int array_of_solverlengths[], const MPI_Aint array_of_displacements[],
                         MPI_Datatype oldtype, MPI_Datatype* newtype);

[[deprecated("Use the wrapper MPI_Type_contiguous(..., AT_) instead!")]] int
MPI_Type_contiguous(int count, MPI_Datatype old_type, MPI_Datatype* new_type_p);

[[deprecated("Use the wrapper MPI_Type_commit(..., AT_) instead!")]] int MPI_Type_commit(MPI_Datatype* datatype);

[[deprecated("Use the wrapper MPI_Type_vector(..., AT_) instead!")]] int
MPI_Type_vector(int count, int solverlength, int stride, MPI_Datatype oldtype, MPI_Datatype* newtype);

[[deprecated("Use the wrapper MPI_Type_create_struct(..., AT_) instead!")]] int
MPI_Type_create_struct(int count, int* array_of_solverlengths, MPI_Aint* array_of_displacements, MPI_Datatype* array_of_types,
                MPI_Datatype* newtype);

[[deprecated("Use the wrapper MPI_Type_create_hvector(..., AT_) instead!")]] int
MPI_Type_create_hvector(int count, int solverlength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype* newtype);

[[deprecated("Use the wrapper MPI_Type_create_darray(..., AT_) instead!")]] int
MPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[], const int array_of_distribs[],
                       const int array_of_dargs[], const int array_of_psizes[], int order, MPI_Datatype oldtype,
                       MPI_Datatype* newtype);

[[deprecated("Use the wrapper MPI_Type_create_hindexed(..., AT_) instead!")]] int
MPI_Type_create_indexed_solver(int count, int solverlength, const int array_of_displacements[], MPI_Datatype oldtype,
                               MPI_Datatype* newtype);

[[deprecated("Use the wrapper MPI_Type_set_name(..., AT_, <...>) instead!")]] int
MPI_Type_set_name(MPI_Datatype type, const char* type_name);


// MPI Group
[[deprecated("Use the wrapper MPI_Group_incl(..., AT_) instead!")]] int
MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group* newgroup);

[[deprecated("Use the wrapper MPI_Group_free(..., AT_) instead!")]] int MPI_Group_free(MPI_Group* group);

[[deprecated("Use the wrapper MPI_Group_compare(..., AT_) instead!")]] int
MPI_Group_compare(MPI_Group group1, MPI_Group group2, int* result);

[[deprecated("Use the wrapper MPI_Group_excl(..., AT_) instead!")]] int
MPI_Group_excl(MPI_Group group, int n, const int ranks[], MPI_Group* newgroup);

[[deprecated("Use the wrapper MPI_Group_difference(..., AT_) instead!")]] int
MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group* newgroup);

[[deprecated("Use the wrapper MPI_Group_intersection(..., AT_) instead!")]] int
MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group* newgroup);

[[deprecated("Use the wrapper MPI_Group_range_excl(..., AT_) instead!")]] int
MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group* newgroup);

[[deprecated("Use the wrapper MPI_Group_range_incl(..., AT_) instead!")]] int
MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group* newgroup);

[[deprecated("Use the wrapper MPI_Group_rank(..., AT_) instead!")]] int MPI_Group_rank(MPI_Group group, int* rank);

[[deprecated("Use the wrapper MPI_Group_size(..., AT_) instead!")]] int MPI_Group_size(MPI_Group group, int* size);

[[deprecated("Use the wrapper MPI_Group_union(..., AT_) instead!")]] int
MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group* newgroup);

// MISC
[[deprecated("Use the wrapper MPI_Start(..., AT_) instead!")]] int MPI_Start(MPI_Request* request);

[[deprecated("Use the wrapper MPI_Startall(..., AT_) instead!")]] int MPI_Startall(int count,
                                                                                   MPI_Request array_of_requests[]);

[[deprecated("Use the wrapper MPI_Get_count(..., AT_) instead!")]] int MPI_Get_count(const MPI_Status* status,
                                                                                     MPI_Datatype datatype, int* count);

[[deprecated("Use the wrapper MPI_Get_address(..., AT_) instead!")]] int MPI_Get_address(const void* location,
                                                                                         MPI_Aint* address);

[[deprecated("Use the wrapper MPI_Abort(..., AT_) instead!")]] int MPI_Abort(MPI_Comm comm, int errorcode);

[[deprecated("Use the wrapper MPI_Status_set_cancelled(..., AT_, <...>) instead!")]] int
MPI_Status_set_cancelled(MPI_Status* status, int flag);

[[deprecated("Use the wrapper MPI_Alloc_mem(..., AT_, <...>) instead!")]] int
MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void* baseptr);

[[deprecated("Use the wrapper MPI_Op_create(..., AT_, <...>) instead!")]] int MPI_Op_create(MPI_User_function* function,
                                                                                            int commute, MPI_Op* op);

[[deprecated("Use the wrapper MPI_Op_commutative(..., AT_, <...>) instead!")]] int MPI_Op_commutative(MPI_Op op,
                                                                                                      int* commute);

[[deprecated("Use the wrapper MPI_MPI_Op_free(..., AT_, <...>) instead!")]] int MPI_Op_free(MPI_Op* op);

[[deprecated("Use the wrapper MPI_Compare_and_swap(..., AT_, <...>) instead!")]] int
MPI_Compare_and_swap(const void* origin_addr, const void* compare_addr, void* result_addr, MPI_Datatype datatype,
                     int target_rank, MPI_Aint target_disp, MPI_Win win);

[[deprecated("Use the wrapper MPI_Probe(..., AT_, <...>) instead!")]] int MPI_Probe(int source, int tag, MPI_Comm comm,
                                                                                    MPI_Status* status);

[[deprecated("Use the wrapper MPI_Request_get_status(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Request_get_status(MPI_Request request, int* flag, MPI_Status* status);

[[deprecated("Use the wrapper MPI_Request_free(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Request_free(MPI_Request* request);

[[deprecated("Use the wrapper MPI_Cancel(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Cancel(MPI_Request* request);


// MPI Info
[[deprecated("Use the wrapper MPI_Info_create(..., AT_) instead!")]] int MPI_Info_create(MPI_Info* info);

[[deprecated("Use the wrapper MPI_Info_free(..., AT_) instead!")]] int MPI_Info_free(MPI_Info* info);

[[deprecated("Use the wrapper MPI_Info_get(..., AT_) instead!")]] int
MPI_Info_get(MPI_Info info, char* key, int valuelen, char* value, int* flag);

[[deprecated("Use the wrapper MPI_Info_nthkey(..., AT_) instead!")]] int MPI_Info_get_nthkey(MPI_Info info, int n,
                                                                                             char* key);

[[deprecated("Use the wrapper MPI_Info_nkeys(..., AT_) instead!")]] int MPI_Info_get_nkeys(MPI_Info info, int* nkeys);

[[deprecated("Use the wrapper MPI_Info_valuelen(..., AT_) instead!")]] int
MPI_Info_get_valuelen(MPI_Info info, const char* key, int* valuelen, int* flag);

// MPI File
[[deprecated("Use the wrapper MPI_File_open(..., AT_) instead!")]] int
MPI_File_open(MPI_Comm comm, char* filename, int amode, MPI_Info info, MPI_File* mpi_fh);


[[deprecated("Use the wrapper MPI_File_seek(..., AT_) instead!")]] int MPI_File_seek(MPI_File mpi_fh, MPI_Offset offset,
                                                                                     int whence);

[[deprecated("Use the wrapper MPI_File_close(..., AT_) instead!")]] int MPI_File_close(MPI_File* mpi_fh);

[[deprecated("Use the wrapper MPI_File_write_shared(..., AT_) instead!")]] int
MPI_File_write_shared(MPI_File mpi_fh, const void* buf, int count, MPI_Datatype datatype, MPI_Status* status);

[[deprecated("Use the wrapper MPI_File_iwrite_shared(..., AT_) instead!")]] int
MPI_File_iwrite_shared(MPI_File mpi_fh, const void* buf, int count, MPI_Datatype datatype, MPI_Request* request);

// Topologies
[[deprecated("Use the wrapper MPI_Cart_coords(..., AT_, <...>) instead!")]] int
MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);

[[deprecated("Use the wrapper MPI_Cart_create(..., AT_, <...>) instead!")]] int
MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm* comm_cart);

[[deprecated("Use the wrapper MPI_Graph_create(..., AT_, <...>) instead!")]] int
MPI_Graph_create(MPI_Comm comm_old, int nnodes, const int index[], const int edges[], int reorder,
                 MPI_Comm* comm_graph);

// One-sided communication
[[deprecated("Use the wrapper MPI_Win_fence(..., AT_, <...>) instead!")]] int MPI_Win_fence(int assert, MPI_Win win);

[[deprecated("Use the wrapper MPI_Put(..., AT_, <name of sendbuf>, <name of recvbuf>) instead!")]] int
MPI_Put(const void* origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
        int target_count, MPI_Datatype target_datatype, MPI_Win win);

[[deprecated("Use the wrapper MPI_Accumulate(..., AT_, <...>) instead!")]] int
MPI_Accumulate(const void* origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank,
               MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win);

#endif
#endif

#if defined(MAIA_GCC_COMPILER)
#pragma GCC diagnostic pop
#elif defined(MAIA_CLANG_COMPILER)
#pragma clang diagnostic pop
#endif

#endif
