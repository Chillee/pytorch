
#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
#include <iterator>
#include <list>
#include <torch/script.h>
#include <pybind11/pybind11.h>

using namespace std;

namespace py = pybind11;
void warp_perspective(torch::Tensor image) { cout<<"HEY"<<endl; }
struct Foo : c10::intrusive_ptr_target {
  int x, y;
  Foo(): x(2), y(5){}
  Foo(int x_, int y_) : x(x_), y(y_) {}
  void display() {
    cout<<"x: "<<x<<' '<<"y: "<<y<<endl;
  }
  int64_t add(int64_t z) {
    return (x+y)*z;
  }
  Foo* combine(c10::intrusive_ptr<Foo> x) {
    this->x += x->x;
    this->y += x->y;
    return this;
  }
  ~Foo() {
    std::cout<<"Destroying object with values: "<<x<<' '<<y<<std::endl;
  }
};

static auto registry = torch::RegisterOperators("my_ops::warp_perspective",
                                                     &warp_perspective);
static auto test = torch::jit::class_<Foo>("Foo")
                    .def(torch::jit::init<int64_t, int64_t>())
                    // .def(torch::jit::init<>())
                    .def("display", &Foo::display)
                    .def("add", &Foo::add);
                    // .def("combine", &Foo::combine);
