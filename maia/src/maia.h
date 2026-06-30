// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only


// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only


// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only


#ifndef MAIN_H_
#define MAIN_H_

#include <streambuf>



/// \brief Auxiliary class to allow creating versions of cout/cerr that only
///        output under certain conditions.
///
/// \author Michael Schlottke-Lakemper (mic) <mic@aia.rwth-aachen.de>
/// \date 2016-09-28
class NullBuffer : public std::streambuf {
 public:
  int overflow(int c) override { return c; }
};

class MAIA {
 public:
  MAIA(int argc, char** argv) : m_argc(argc), m_argv(argv) {}

  int run();

 private:
  NullBuffer nullBuffer;

  int m_argc = 0;
  char** m_argv = nullptr;
};

#endif // ifndef MAIN_H_
