#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"
int main(int argc, char **argv)
{
    pangolin::init();
    pangolin::Config config;

    std::vector<int> gpus;
    std::string path;
    int iters = 1;
    bool help = false;
    bool debug = false;
    bool verbose = false;

    bool readMostly = false;
    bool accessedBy = false;
    bool prefetchAsync = false;

    clara::Parser cli;
    cli = cli | clara::Help(help);
    cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
    cli = cli |
        clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
    cli = cli | clara::Opt(gpus, "ids")["-g"]("gpus to use");
    cli = cli | clara::Opt(readMostly)["--read-mostly"](
                    "mark data as read-mostly by all gpus before kernel");
    cli = cli | clara::Opt(accessedBy)["--accessed-by"](
                    "mark data as accessed-by all GPUs before kernel");
    cli = cli | clara::Opt(prefetchAsync)["--prefetch-async"](
                    "prefetch data to all GPUs before kernel");
    cli = cli | clara::Opt(iters, "N")["-n"]("number of counts");
    cli =
        cli | clara::Arg(path, "graph file")("Path to adjacency list").required();

    auto result = cli.parse(clara::Args(argc, argv));
    if (!result) {
        LOG(error, "Error in command line: {}", result.errorMessage());
        exit(1);
    }

    if (help) {
        std::cout << cli;
        return 0;
    }

    // set logging level
    if (verbose) {
        pangolin::logger::set_level(pangolin::logger::Level::TRACE);
    } else if (debug) {
        pangolin::logger::set_level(pangolin::logger::Level::DEBUG);
    }

  // log command line before much else happens
    {
        std::string cmd;
        for (int i = 0; i < argc; ++i) {
            if (i != 0) {
                cmd += " ";
            }
                cmd += argv[i];
        }
        LOG(debug, cmd);
    }

    LOG(debug, "pangolin version: {}.{}.{}", PANGOLIN_VERSION_MAJOR,
        PANGOLIN_VERSION_MINOR, PANGOLIN_VERSION_PATCH);
    LOG(debug, "pangolin branch:  {}", PANGOLIN_GIT_REFSPEC);
    LOG(debug, "pangolin sha:     {}", PANGOLIN_GIT_HASH);
    LOG(debug, "pangolin changes: {}", PANGOLIN_GIT_LOCAL_CHANGES);

#ifndef NDEBUG
  LOG(warn, "Not a release build");
#endif

    if (gpus.empty()) {
        LOG(warn, "no GPUs provided on command line, using GPU 0");
        gpus.push_back(0);
    }

    //read data
    auto start = std::chrono::system_clock::now();
    pangolin::EdgeListFile file(path);

    std::vector<pangolin::EdgeTy<uint64_t>> edges;
    std::vector<pangolin::EdgeTy<uint64_t>> fileEdges;
    while (file.get_edges(fileEdges, 10)) {
        edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
    }
    double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "read_data time {}s", elapsed);
    LOG(debug, "read {} edges", edges.size());

    // create coo
    start = std::chrono::system_clock::now();
    auto upperTriangular = [](pangolin::EdgeTy<uint64_t> e) {
        return e.first < e.second;
    };
    auto coo = pangolin::COO<uint32_t>::from_edges(edges.begin(), edges.end());
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "num_nodes = {}", coo.num_nodes());
    LOG(info, "nnz = {}", coo.nnz());
    LOG(info, "create COO time {}s", elapsed);

    pangolin::CrossDecomp<pangolin::COO<uint32_t>> CD;

    CD.CrossDecompInit(coo);
    CD.Host_evalEdges(coo);
    // CD.Host_repartition(true, 1, coo);
    CD.Host_repartition(false, 3, coo);
    CD.Host_evalEdges(coo);


    
	return 0;
}
