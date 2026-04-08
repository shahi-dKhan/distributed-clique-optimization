# distributed-clique-optimization

1. Install MPI                                                                                                      
  brew install open-mpi
                                                                                                                      
  2. Compile both programs                                                     
  g++    -O3 -std=c++17 sequential.cpp -o seq_bin
  mpic++ -O3 -std=c++17 main.cpp       -o par_bin
                                                 
  3. Run the full Mac test suite (generates graphs, checks correctness, shows scaling)                                
  ./test_mac.sh                                                                                                       
                                                                                                                      
  Or if you want to do things manually:                                                                               
                                                                               
  3a. Generate a test graph                                                                                           
  python3 gen.py -N 60 --density 0.5 -B 120 -s 3 -o test.txt
                                                                                                                      
  3b. Run sequential                                                                                                  
  ./seq_bin test.txt output_seq.txt && cat output_seq.txt
                                                                                                                      
  3c. Run parallel                                                                                                    
  mpirun -np 4 ./par_bin test.txt output_par.txt && cat output_par.txt
                                                                                                                      
  3d. Compare both on a file                                                                                          
  NP=4 ./compare.sh test.txt                                                                                          
                                                                                                                      
  3e. Scaling sweep                                                                                                   
  PROCS="1 2 4 8" ./run_scaling.sh test.txt                                    
                                         
  ---                                                                                                                 
  That's it. Start with brew install open-mpi then ./test_mac.sh — everything else is automated from there.

  