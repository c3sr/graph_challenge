#include <iostream>
#include <fmt/format.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.hpp"

int main(int argc, char **argv)
{

	pangolin::Config config;

	std::string adjacencyListPath;
	bool help = false;
	bool debug = false;
	bool verbose = false;
	bool seedSet = false;

	clara::Parser cli;
	cli = cli | clara::Help(help);
	cli = cli | clara::Opt(debug)
					["--debug"]("print debug messages to stderr");
	cli = cli | clara::Opt(verbose)
					["--verbose"]("print verbose messages to stderr");
	cli = cli | clara::Opt(config.numCPUThreads_, "int")
					["-c"]["--num_cpu"]("number of cpu threads");
	cli = cli | clara::Opt(config.gpus_, "ids")
					["-g"]("gpus to use");
	cli = cli | clara::Opt(config.hints_)
					["--unified-memory-hints"]("use unified memory hints");
	cli = cli | clara::Opt(config.storage_, "zc|um")
					["-s"]("GPU memory kind");
	cli = cli | clara::Opt([&](unsigned int seed) {
			  seedSet = true;
			  config.seed_ = seed;
			  return clara::detail::ParserResult::ok(clara::detail::ParseResultType::Matched);
		  },
						   "int")["--seed"]("random seed");
	cli = cli | clara::Opt(config.type_, "cpu|csr|cudamemcpy|edge|hu|impact2018|impact2018|nvgraph|vertex")["-m"]["--method"]("method").required();
	cli = cli | clara::Opt(config.kernel_, "string")["-k"]["--kernel"]("kernel");
	cli = cli | clara::Arg(adjacencyListPath, "graph file")("Path to adjacency list").required();

	auto result = cli.parse(clara::Args(argc, argv));
	if (!result)
	{
		LOG(error, "Error in command line: {}", result.errorMessage());
		exit(1);
	}

	if (help)
	{
		std::cout << cli;
		return 0;
	}

	// set logging level
	if (verbose)
	{
		pangolin::logger::set_level(pangolin::logger::Level::TRACE);
	}
	else if (debug)
	{
		pangolin::logger::set_level(pangolin::logger::Level::DEBUG);
	}

	// log command line before much else happens
	{
		std::string cmd;
		for (int i = 0; i < argc; ++i)
		{
			if (i != 0)
			{
				cmd += " ";
			}
			cmd += argv[i];
		}
		LOG(debug, cmd);
	}
	LOG(debug, "pangolin version: {}.{}.{}", PANGOLIN_VERSION_MAJOR, PANGOLIN_VERSION_MINOR, PANGOLIN_VERSION_PATCH);
	LOG(debug, "pangolin branch:  {}", PANGOLIN_GIT_REFSPEC);
	LOG(debug, "pangolin sha:     {}", PANGOLIN_GIT_HASH);
	LOG(debug, "pangolin changes: {}", PANGOLIN_GIT_LOCAL_CHANGES);

	if (seedSet)
	{
		LOG(debug, "using seed {}", config.seed_);
		srand(config.seed_);
	}
	else
	{
		uint seed = time(NULL);
		LOG(debug, "using seed {}", seed);
		srand(time(NULL));
	}

#ifndef NDEBUG
	LOG(warn, "Not a release build");
#endif
	pangolin::TriangleCounter *tc = pangolin::TriangleCounter::CreateTriangleCounter(config);

	auto start = std::chrono::system_clock::now();
	tc->read_data(adjacencyListPath);
	const size_t numEdges = tc->num_edges();
	double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
	LOG(info, "read_data time {}s", elapsed);

	start = std::chrono::system_clock::now();
	tc->setup_data();
	elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
	LOG(info, "setup_data time {}s", elapsed);

	start = std::chrono::system_clock::now();
	auto numTriangles = tc->count();
	elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
	LOG(info, "count time {}s", elapsed);

	fmt::print("{} {} {} {}\n", adjacencyListPath, numTriangles, elapsed, numEdges / elapsed);

	delete tc;
	return 0;
}
