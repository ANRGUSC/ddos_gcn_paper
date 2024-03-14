# Graph-Based DDoS Attack Detection in IoT Systems with Lossy Network

This repository presents the source code and dataset for analyzing various graph toplogies and utilizing graph convolutional networks (GCN) for detecting DDoS attack in IoT systems with lossy networks.

## Instructions for running the codes

The requirements.txt file contains the modules needed to run these scripts and can be installed by running the following command in the terminal:
* pip install -r requirements.txt

## Dataset

The link to download the dataset is provided in the [/dataset](https://github.com/ANRGUSC/ddos_gcn_paper/tree/main/dataset) directory. Download the dataset before running the code.

## Project config file

The project config file can be found in [/source_code](https://github.com/ANRGUSC/ddos_gcn_paper/tree/main/source_code). The path to the output/dataset/source_code directory can be set in this file.
In order to run the program for getting the results of ICMLCN paper, set the USE_ROUTER parameter to False.

## Running the code

In the [/pre_process](https://github.com/ANRGUSC/ddos_gcn_paper/tree/main/source_code/pre_process) and [/gcn_models](https://github.com/ANRGUSC/ddos_gcn_paper/tree/main/source_code/gcn_models) directories you can find the automated codes for running the whole code.

## Acknowledgement

   This material is based upon work supported in part by Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0160 for the Open, Programmable, Secure 5G (OPS-5G) program. Any views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government. 



