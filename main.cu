
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>

//#include<windows.h>
//for linux
#include<unistd.h>
#define TEAM_SIZE 400


struct Runner {
	int dist;
	int vel;
};

struct Team {
	Runner* runners;
	int curRunner;
	int id;
};


cudaError_t createTeamsWithCuda(Team* teams, Runner* runners, const int size);

//cudaError_t simulateRaceWithCuda(Team* teams, Runner* runners, int* finished_team_count, int* placements, const int size);
cudaError_t simulateRaceWithCuda(Team* teams, Runner* runners, int* finishedTeamCount, int* placements, int* consoleTeams, const int consoleSize, const int size);


__global__ void createTeamKernel(Team* teams, Runner* runners)
{
	int i = threadIdx.x;
	//RUN_LIMIT
	int size = 4;
	teams[i].runners = &runners[i * size];
	for (int j = 0; j < size; j++)
	{
		teams[i].runners[j].dist = j * 100;
		//At first everybody is at stop.
		teams[i].runners[j].vel = 0;
	}
	teams[i].id = i + 1;
	teams[i].curRunner = 0;

}



__global__ void simulateRaceKernel(Team* teams, Runner* runners, int* finished_team_count, int* placements, int rand_seed)
{
	int i = threadIdx.x;
	teams[i].runners = &runners[i * 4];
	int* curRunner = &teams[i].curRunner;

	if (*curRunner == 4)
	{
		//This team has ended the race.
		return;
	}

	teams[i].runners[*curRunner].dist += teams[i].runners[*curRunner].vel;
	

	if (teams[i].runners[*curRunner].dist >= (*curRunner + 1) * 100)
	{
		teams[i].runners[*curRunner].vel = 0;
		*curRunner += 1;
	}

	if (*curRunner == 4)
	{

		int place_index = atomicAdd(finished_team_count, 1);
		placements[place_index] = i;
		if (place_index == 0)
		{
			printf("\nFirst team to arrive finish line is Team %d\n", i + 1);
			printf("-----------------------------------------------\n");
			for (int j = 0; j < 4; j++)
			{
				printf("Team %d Runner %d VEL:%d DIST:%d\n", i + 1, j + 1, teams[i].runners[j].vel, teams[i].runners[j].dist);
			}
			//printf("-----------------------------------------------\n");

			for (int k = 0; k < TEAM_SIZE; k++)
			{
				int tmpRunner = teams[k].curRunner;
				if (tmpRunner == 4)
				{
					tmpRunner -= 1;
				}

				printf("-----------------------------------------------\n");
				printf("Team %d Current Runner %d VEL:%d DIST:%d\n", k + 1, tmpRunner + 1, teams[k].runners[tmpRunner].vel, teams[k].runners[tmpRunner].dist);
			}
		}
	}


	else
	{
		//Race has not ended for this team.
		//Give new velocity to calculate next distance.
		curandState_t state;
		curand_init(rand_seed, i, 0, &state);

		teams[i].runners[*curRunner].vel = ceil(curand_uniform(&state) * 5);
	}
}





int main(int argc, char* argv[])
{

	//pointer for objects
	Team* teams = new Team[TEAM_SIZE];
	Runner* runners = new Runner[TEAM_SIZE * 4];
	//Pointer for placements
	int* placements = new int[TEAM_SIZE];


	cudaError_t cudaStatus = createTeamsWithCuda(teams, runners, TEAM_SIZE);

	//Pointers must be reassigned because 
	//Pointer values on the objects are for gpu memory (video-ram)
	//They are needed to be repointed to cpu memory (ram or virtual ram)
	for (int i = 0; i < TEAM_SIZE; i++) {
		teams[i].runners = &runners[i * 4];
	}


	int* consoleTeams;
	int consoleSize = 0;
	int finished_team_count = 0;

	if (argc <= 1)
	{
		consoleTeams = new int[TEAM_SIZE];
		printf("No arguments were passed while running the program.\nPlease state which teams will be shown on the console.\n");
		printf("All numbers must be seperated by space\n");
		do {
			scanf("%d", &consoleTeams[consoleSize++]);
		} while (getchar() != '\n' && consoleSize < TEAM_SIZE);
	}
	else
	{
		consoleTeams = new int[argc - 1];
		for (int i = 1; i < argc; i++)
		{
			sscanf(argv[i], "%d", &consoleTeams[consoleSize++]);
		}
	}
	for (int i = 0; i < consoleSize; i++)
	{
		if (consoleTeams[i] <= 0)
		{
			printf("Can't give an argument below or equal to 0 or NaN. Teams start at 1.");
			exit(-1);
		}
		else if (consoleTeams[i] > TEAM_SIZE)
		{
			printf("Can't select non existent team");
			exit(-2);
		}

	}

	simulateRaceWithCuda(teams, runners, &finished_team_count, placements, consoleTeams, consoleSize, TEAM_SIZE);


	printf("Race has ended The Results are\n");;

	for (int i = 0; i < TEAM_SIZE; i++)
	{
		printf("%d PLACE: TEAM %d\n", i + 1, teams[placements[i]].id);
	}



	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	free(teams);
	free(runners);
	free(placements);
	free(consoleTeams);
	return 0;
}

cudaError_t simulateRaceWithCuda(Team* teams, Runner* runners, int* finishedTeamCount, int* placements, int* consoleTeams, const int consoleSize, const int size)
{
	Team* dev_teams;
	Runner* dev_runners;
	int* dev_placements;
	int* dev_count;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	//Memory allocation and Memory copy for host to device.
	cudaStatus = cudaMalloc((void**)&dev_placements, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_placements, placements, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}



	cudaStatus = cudaMalloc((void**)&dev_teams, size * sizeof(Team));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_teams, teams, size * sizeof(Team), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_runners, size * sizeof(Runner) * 4);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_runners, runners, size * sizeof(Runner) * 4, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_count, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_count, finishedTeamCount, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//End of Memory Allocation and Memory Copy

	//Start race loop
	while (*finishedTeamCount < TEAM_SIZE) {

		srand(time(NULL));
		int rand_seed = rand() % 500 + 1000;

		// Launch a kernel on the GPU with one thread for each element.
		simulateRaceKernel <<<1, size >>> (dev_teams, dev_runners, dev_count, dev_placements, rand_seed);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "simulateRaceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simulateRaceKernel!\n", cudaStatus);
			goto Error;
		}

		// Copy output vectors from GPU buffer to host memory.

		cudaStatus = cudaMemcpy(teams, dev_teams, size * sizeof(Team), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy device to host on Teams failed!");
			goto Error;
		}


		cudaStatus = cudaMemcpy(runners, dev_runners, size * sizeof(Runner) * 4, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy device to host on Runners failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(finishedTeamCount, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy device to host on count failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(placements, dev_placements, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy device to host on placements failed!");
			goto Error;
		}

		//Pointers must be reassigned. 
		//Pointer values on the objects are for gpu memory (video-ram).
		//They are needed to be repointed to cpu memory (ram or virtual ram).
		for (int i = 0; i < TEAM_SIZE; i++)
		{
			teams[i].runners = &runners[i * 4];
		}

		for (int i = 0; i < consoleSize; i++)
		{
			printf("-------------------------\n");
			int outTeam = consoleTeams[i];
			for (int j = 0; j < 4; j++)
			{
				printf("Team %d Runner %d VEL:%d DIST:%d\n", outTeam, j + 1, teams[outTeam - 1].runners[j].vel, teams[outTeam - 1].runners[j].dist);
			}
		}

		printf("-------------------------\n");
		printf("Finished Team Count:%d\n", *finishedTeamCount);
		printf("|||||||||||||||||||||||||\n");

		//Sleep function in windows is in milliseconds
		//1000 olucak
		//Sleep(1 * 1000);
		//For linux based
		// It is in seconds for linux.
		sleep(1);

	}
	//End race loop

Error:
	cudaFree(dev_teams);
	cudaFree(dev_runners);
	cudaFree(dev_count);
	cudaFree(dev_placements);

	return cudaStatus;

}


cudaError_t createTeamsWithCuda(Team* teams, Runner* runners, const int size)
{
	Team* dev_teams;
	Runner* dev_runners;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_teams, size * sizeof(Team));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_teams, teams, size * sizeof(Team), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_runners, size * sizeof(Runner) * 4);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_runners, runners, size * sizeof(Runner) * 4, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.
	createTeamKernel << <1, size >> > (dev_teams, dev_runners);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "createTeamKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//printf("%d", dev_teams[0].runners[0].dist);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching createTeamKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.

	cudaStatus = cudaMemcpy(teams, dev_teams, size * sizeof(Team), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on Teams failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(runners, dev_runners, size * sizeof(Runner) * 4, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on Runners failed!");
		goto Error;
	}




Error:
	cudaFree(dev_teams);
	cudaFree(dev_runners);

	return cudaStatus;

}