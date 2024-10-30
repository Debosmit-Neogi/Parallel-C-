#include "mpi.h"
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);  // Initialize MPI

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // Number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  // Current process rank

    // Total size of the array to sum
    const int N = 100;
    vector<int> data(N);

    // Initialize array only on the root process
    if (world_rank == 0) {
        for (int i = 0; i < N; ++i) data[i] = 1;  // Fill with 1s for simplicity
    }

    // Determine segment size for each process
    int segment_size = N / world_size;
    vector<int> local_data(segment_size);

    // Distribute segments of data to all processes
    MPI_Scatter(data.data(), segment_size, MPI_INT,
                local_data.data(), segment_size, MPI_INT,
                0, MPI_COMM_WORLD);

    // Each process computes the sum of its segment
    int local_sum = 0;
    for (int i = 0; i < segment_size; ++i) {
        local_sum += local_data[i];
    }

    // Gather all partial sums to the root process
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Display the result on the root process
    if (world_rank == 0) {
        cout << "Total Sum: " << global_sum << std::endl;
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}
