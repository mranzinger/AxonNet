#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <serialization/master.h>

#include "neural_net.h"
#include "linear_layer.h"
#include "neuron_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "convo_layer.h"
#include "logloss_cost.h"
#include "maxpool_layer.h"
#include "handwritten_loader.h"

using namespace std;
using namespace axon::serialization;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char *argv [])
{
    string datasetRoot;
    string networkFile;
    string checkpointFile;
    float learningRate;
    int testRate = -1;
    string configFile;
    string checkpointRoot;

    po::options_description desc("Allowed Options");
    desc.add_options()
            ("help", "produce help message")
            ;
        
    {
        po::options_description mand("Mandatory Options");

        mand.add_options()
           ("dataset,d", po::value(&datasetRoot), "Dataset Directory")
           ("net,n", po::value(&networkFile), "Network definition File")
           ("learn-rate,l", po::value(&learningRate), "Learning Rate")
           ;
 
        desc.add(mand);
    }
    {
        po::options_description opt("Optional");

        opt.add_options()
            ("checkpoint,c", po::value(&checkpointFile), "Checkpoint File")
            ("save-dir,s", po::value(&checkpointRoot)->default_value("test"), 
                "Checkpoint Save Directory")
            ("test-rate,f", po::value(&testRate)->default_value(-1), "Test Frequency.")
            ("cfg,g", po::value(&configFile), "Config File")
            ;
        
        desc.add(opt);   
    }

    po::variables_map varMap;
    po::store(po::parse_command_line(argc, argv, desc), varMap);
    po::notify(varMap);

    if (varMap.count("help"))
    {
        cout << desc << endl;
        return EXIT_SUCCESS;
    }

    if (fs::exists(configFile))
    {
        ifstream cfg(configFile);
        po::store(po::parse_config_file(cfg, desc), varMap);
        po::notify(varMap);
    }

    if (!fs::exists(datasetRoot))
    {
        cout << "The specified dataset root directory doesn't exist." << endl;
        cout << datasetRoot << endl;
        return EXIT_FAILURE;
    }

    HandwrittenLoader loader(datasetRoot);
   
    if (testRate <= 0)
        testRate = loader.Size();
    
    if (!fs::exists(networkFile))
    {
        cout << "The specified network configuration file doesn't exist." << endl;
        return EXIT_FAILURE;
    }

    // Load the network
    NeuralNet net;
    CJsonSerializer().DeserializeFromFile(networkFile, net);

    if (fs::exists(checkpointFile))
    {
        net.Load(checkpointFile);
    }

    if (checkpointRoot.empty())
    {
        cout << "The checkpoint save directory cannot be empty." << endl;
        return EXIT_FAILURE;
    }

    if (!fs::exists(checkpointRoot))
    {
        fs::create_directories(checkpointRoot);
    }

    net.SetLearningRate(learningRate);

	//string root = "/home/mike/dev/personal/mnist/";

	net.Train(loader, 100000000, testRate, checkpointRoot);
}
