# Framework of federated learning with client selection using MPI

## Introduction

This is an implementation of federated learning framework with client selection using MPI.

* Global hyperparameters are defined in config.yml
* server_main.py is the main file to start the server program
* client_main.py is the main file to start the client program

## Start program

If you want to run this program that contains up to 10 selected clients, you can input this command in the console:

``
mpiexec --oversubscribe -n 1 python server_main.py : -n 10 python client_main.py
``

Each client and the server run as a process, which communicate with others through MPI.


## Note
Make sure that the maximum number of selected clients is less than or equal to that of clients in the command.

For example, this command allows up to 100 selected clients in each epoch:
``
mpiexec --oversubscribe -n 1 python server_main.py : -n 100 python client_main.py
``
