#include <iostream>
#include <fstream>
#include <chrono>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <serialization/master.h>

#include <cuda_runtime_api.h>

#define _UNIT_TESTS_
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
    static const int s_testSize = 5;

    ConvoLayer convoTest("",
                        3, 128,
                        7, 7,
                        5, 5,
                        5, 5);

    Params computeInput(256, 256, 3,
                new CMatrix(CMatrix::Random(256 * 256 * 3, 128)));

    Params hostComp = convoTest.SCompute(computeInput, false);

    cout << "Running CPU Timing Test" << endl;

    auto timeStart = chrono::high_resolution_clock::now();

    for (int i = 0; i < s_testSize; ++i)
    {
        cout << i << endl;

        Params computeOutput = convoTest.SCompute(computeInput, false);
    }

    auto timeStop = chrono::high_resolution_clock::now();

    auto durMs = chrono::duration_cast<chrono::milliseconds>(timeStop - timeStart);

    cout << "Done. Total Time: " << durMs.count() << "ms." << endl
         << "Time Per Op: " << (durMs.count() / float(s_testSize)) << "ms." << endl;

    convoTest.SetDevicePreference(CudaDevicePreference::Create(1));

    // Run a dummy computation to get the buffer onto the device.
    // Memory transfer is not part of the test
    Params cudaComp = convoTest.SCompute(computeInput, false);

    /*if (!hostComp.GetHostMatrix().isApprox(cudaComp.GetHostMatrix(), 0.001))
    {
        cout << "Failed convolution computation" << endl;
        return 1;
    }*/

    cudaError_t cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess)
    {
        cout << "Invalid cuda execution" << endl;
        return 1;
    }

    cout << "Running GPU Timing Test" << endl;

    timeStart = chrono::high_resolution_clock::now();

    for (int i = 0; i < s_testSize; ++i)
    {
        cout << i << endl;

        Params computeOutput = convoTest.SCompute(computeInput, false);

        cudaErr = cudaDeviceSynchronize();

        if (cudaErr != cudaSuccess)
        {
            cout << "Invalid cuda execution" << endl;
            return 1;
        }
    }

    timeStop = chrono::high_resolution_clock::now();

    durMs = chrono::duration_cast<chrono::milliseconds>(timeStop - timeStart);

    cout << "Done. Total Time: " << durMs.count() << "ms." << endl
         << "Time Per Op: " << (durMs.count() / float(s_testSize)) << "ms." << endl;

    return 0;

    string datasetRoot;
    string networkFile;
    string checkpointFile;
    float learningRate;
    size_t testRate = 0;
    size_t batchSize = 32;
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
            ("test-rate,f", po::value(&testRate)->default_value(0), "Test Frequency.")
            ("batch-size,b", po::value(&batchSize)->default_value(32), "Batch Size.")
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

    net.Train(loader, batchSize, 1000000000, testRate, checkpointRoot);
}
