#include <fmt/format.h>

#include "pangolin/configure.hpp"

int main(void) {
  fmt::print("version: {}.{}.{}\n", PANGOLIN_VERSION_MAJOR, PANGOLIN_VERSION_MINOR, PANGOLIN_VERSION_PATCH);
  fmt::print("branch:  {}\n", PANGOLIN_GIT_REFSPEC);
  fmt::print("sha:     {}\n", PANGOLIN_GIT_HASH);
  fmt::print("changes: {}\n", PANGOLIN_GIT_LOCAL_CHANGES);
}