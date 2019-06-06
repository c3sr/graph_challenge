#include "pangolin/pangolin.hpp"

int main(void) {
  std::cout << fmt::format("version: {}.{}.{}\n", PANGOLIN_VERSION_MAJOR,
                           PANGOLIN_VERSION_MINOR, PANGOLIN_VERSION_PATCH);
  std::cout << fmt::format("branch:  {}\n", PANGOLIN_GIT_REFSPEC);
  std::cout << fmt::format("sha:     {}\n", PANGOLIN_GIT_HASH);
  std::cout << fmt::format("changes: {}\n", PANGOLIN_GIT_LOCAL_CHANGES);
}