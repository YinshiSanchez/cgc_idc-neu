# README

# Getting Started

1.  gcc/ g++ 9.4.0
    
2.  openmp
    

*   进入官网OpenMP，下载稳定版本。[https://www.open-mpi.org/software/ompi/v4.1/](https://www.open-mpi.org/software/ompi/v4.1/)
    

*   解压后进入文件夹，运行
    

    ./configure --prefix=/home/openmp

*   编译运行
    

    make -j4
    sudo make install

# Build & Run

    # Build
    cd /your_DIR/cgc_idc-neu/idc-neu
    make
    # Run
    ./idc-neu.exe 64 16 8 graph/1024_example_graph.txt embedding/1024.bin weight/W_64_16.bin weight/W_16_8.bin