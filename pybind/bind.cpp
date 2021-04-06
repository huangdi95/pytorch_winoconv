#include <string>

#include <torch/extension.h>

#include "pybind/extern.hpp"
#include "src/common.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  std::string name = std::string("Foo");
  py::class_<Foo>(m, name.c_str())
      .def(py::init<>())
      .def("setKey", &Foo::setKey)
      .def("getKey", &Foo::getKey)
      .def("__repr__", [](const Foo &a) { return a.toString(); });

  m.def("AddGPU", &AddGPU<float>);
//  m.def("AddGPU", &AddGPU<at::Half>);
  m.def("WinoSplit", &WinoSplit<float>);
  m.def("WinoConcat", &WinoConcat<float>);
  m.def("DWM", &DWM<float>);
  m.def("DWM2D", &DWM2D<float>);
//  m.def("DWM", &DWM<at::Half>);
  m.def("FDWM2D", &FDWM2D<float>);
}
