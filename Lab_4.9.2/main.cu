#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "device_atomic_functions.h"

#include <fstream>
#include <iostream>
#include <string>
#include <ctime>
#include <iomanip>


using namespace std;


//Params for statistical hypothesis testing
const int Blocks = 14;
__device__ __constant__  int C = 133;
__device__ __constant__  int N = 32 * Blocks;
__device__ __constant__  int BLOCKS = Blocks;


//L1 Buffer
__device__ __managed__ unsigned int l1_buffer[8] = { 0 };
__device__ __managed__ unsigned int l1_gamma_buffer[8][Blocks] = { 0 };
__device__ __managed__ unsigned int l1_buffer_counter = 0;


//L2 Buffer
__device__ __managed__ unsigned int l2_buffer[24] = { 0 };
__device__ __managed__ unsigned int l2_gamma_buffer[24][Blocks] = { 0 };
__device__ __managed__ unsigned int l2_buffer_counter = 0;


//L3 Buffer
__device__ __managed__ unsigned int l3_buffer[16] = { 0 };
__device__ __managed__ unsigned int l3_buffer_counter = 0;


//LFSR structure
__device__ __constant__ unsigned int l1_feedback = 0b00110010100000000000000000000000;
__device__ __constant__ unsigned int l2_feedback = 0b01001000000000000000000000000000;
__device__ __constant__ unsigned int l3_feedback = 0b11110101000000000000000000000000;


//Service functions
unsigned int* read_data(string data_file, int& n);
bool gamma_cheack(const unsigned int* gamma, int n, unsigned int l1, unsigned int l2, unsigned int l3);
void lfsr_roll_back(unsigned int& l1, unsigned int& l2, unsigned int& l3, int n);
unsigned int host_parity(unsigned int n);
string bin(unsigned int);


//Kernals
__global__ void l1_register_brute_force(const unsigned int* gamma, const unsigned int state_prefix);
__global__ void l2_register_brute_force(const unsigned int* gamma, const unsigned int state_prefix);
__global__ void l3_register_brute_force(const unsigned int* gamma, const unsigned int l1_index, const unsigned int l2_index, const unsigned int state_prefix);



int main(int argc, char* argv[]) {
	//Reading data from file
	string file(argv[1]);
	int n = 0;
	unsigned int* host_gamma = read_data(file, n);
	unsigned int* device_gamma;
	unsigned int MAX_ROUND = n / Blocks;


	bool target = false;

	for (int i = 0; i < MAX_ROUND; i++) {
		//Duplication gamma from host memory to device memory
		cudaMalloc(&device_gamma, Blocks * sizeof(unsigned int));
		cudaMemcpy(device_gamma, &host_gamma[i * Blocks], Blocks * sizeof(unsigned int), cudaMemcpyHostToDevice);

		//Brute force
		//Kernals params
		unsigned int threadsPerBlock = 1 << 5;
		unsigned int blocksPerGrig = 1 << 23;

		float progress = 0;
		
		cout << "Attack round: " << i << endl;

		//L1 brute force
		cout << "L1 brute force: " << endl;
		for (unsigned int r = 0; r < 4; r++) {
			l1_register_brute_force <<< blocksPerGrig, threadsPerBlock >>> (device_gamma, r);
			cudaDeviceSynchronize();
			progress = (((float)r) / 4) * 100;
			cout << fixed << setprecision(2) << "\r\t[Progress: " << progress << "%]";
		}
		cout << "\r\t[Progress: complete]" << endl;

		//L2 brute force
		cout << "L2 brute force: " << endl;
		for (unsigned int r = 0; r < 8; r++) {
			l2_register_brute_force <<< blocksPerGrig, threadsPerBlock >>> (device_gamma, r);
			cudaDeviceSynchronize();
			progress = (((float)r) / 8) * 100;
			cout << fixed << setprecision(2) << "\r\t[Progress: " << progress << "%]";
		}
		cout << "\r\t[Progress: complete]" << endl;

		//L3 brute force
		unsigned int J, K;
		cudaMemcpy(&J, &l1_buffer_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&K, &l2_buffer_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);


		if ((J == 0) || (J >= 8)) {
			unsigned int zero = 0;
			cudaMemcpy(&l1_buffer_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
			cudaMemcpy(&l2_buffer_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
			cout << "L3 brute force: [failed]" << endl;
			continue;
		}
		if ((K == 0) || (K >= 24)) {
			unsigned int zero = 0;
			cudaMemcpy(&l1_buffer_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
			cudaMemcpy(&l2_buffer_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
			cout << "L3 brute force: [failed]" << endl;
			continue;
		}


		cudaDeviceSynchronize();


		cout << "L3 brute force:" << endl;


		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				//L3 brute force kernal
				for (int r = 0; r < 16; r++) {
					l3_register_brute_force <<< blocksPerGrig, threadsPerBlock >>> (device_gamma, j, k, r);
					cudaDeviceSynchronize();
					progress = (((float)(j * 16 * K + k * 16 + r)) / (J * K * 16)) * 100;
					cout << fixed << setprecision(2) << "\r\t[Progress: " << progress << "%]";
				}

				//Gamma cheack
				unsigned int M;
				cudaMemcpy(&M, &l3_buffer_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
				if ((M == 0) || (M >= 16)) {
					continue;
				}
				unsigned int l1_state, l2_state, l3_state;
				cudaMemcpy(&l1_state, &l1_buffer[j], sizeof(unsigned int), cudaMemcpyDeviceToHost);
				cudaMemcpy(&l2_state, &l2_buffer[k], sizeof(unsigned int), cudaMemcpyDeviceToHost);

				for (int m = 0; m < M; m++) {
					cudaMemcpy(&l3_state, &l3_buffer, sizeof(unsigned int), cudaMemcpyDeviceToHost);
					lfsr_roll_back(l1_state, l2_state, l3_state, i * 32 * Blocks);
					target = gamma_cheack(host_gamma, 32 * n, l1_state, l2_state, l3_state);
					if (target) {
						cout << "\r\t[Progress: complete]" << endl;
						cout << "Result:" << endl;
						cout << "L1 state: " << bin(l1_state) << " (" << hex << l1_state << ")" << endl;
						cout << "L2 state: " << bin(l2_state) << " (" << hex << l2_state << ")" << endl;
						cout << "L3 state: " << bin(l3_state) << " (" << hex << l3_state << ")" << endl;
						break;
					}
				}

				//Params reset
				M = 0;
				cudaMemcpy(&l3_buffer_counter, &M, sizeof(unsigned int), cudaMemcpyHostToDevice);

				if (target)
					break;
			}

			if (target)
				break;
		}
		if (target)
			break;

		cout << "\r\t[Progress: complete]" << endl;
		cout << "Attack round: [failed]" << endl;


		//Params reset
		unsigned int zero = 0;
		cudaMemcpy(&l1_buffer_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(&l2_buffer_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);

		//Memory free
		cudaFree(device_gamma);
	}

	system("pause");

	//Memory free
	delete[] host_gamma;
	cudaFree(device_gamma);

	return 0;
}


void lfsr_roll_back(unsigned int& l1, unsigned int& l2, unsigned int& l3, int n) {
	unsigned int l1_fb = 0b00110010100000000000000000000000;
	unsigned int l2_fb = 0b01001000000000000000000000000000;
	unsigned int l3_fb = 0b11110101000000000000000000000000;

	if (n == 0)
		return;

	for (int i = 0; i < n; i++) {
		unsigned int out_1, out_2, out_3;

		out_1 = l1 & 0x1;
		out_2 = l2 & 0x1;
		out_3 = l3 & 0x1;

		l1 = l1 >> 1;
		l2 = l2 >> 1;
		l3 = l3 >> 1;

		if (host_parity(l1 & l1_fb) != out_1)
			l1 ^= (1 << 29);
		if (host_parity(l2 & l2_fb) != out_2)
			l2 ^= (1 << 30);
		if (host_parity(l3 & l3_fb) != out_3)
			l3 ^= (1 << 31);
	}
}


string bin(unsigned int n) {
	string res = "";
	for (int i = 0; i < 32; i++) {
		if (i % 4 == 0)
			res = " " + res;
		if ((n & 0x1) == 1)
			res = '1' + res;
		else
			res = '0' + res;
		n = n >> 1;
	}
	return res;
}


inline __device__ unsigned int parity(unsigned int x) {
	return __popc(x) & 0b1;
}


inline __device__ unsigned int weigth(unsigned int x) {
	return __popc(x);
}


unsigned int* read_data(string data_file, int& n) {
	string data;

	ifstream in(data_file);
	in >> data;
	in.close();

	n = ceill(data.length() / 32);

	unsigned int* gamma = new unsigned int[n];

	for (int i = 0; i < n; i++)
		gamma[i] = 0;

	for (int i = 0; i < data.length(); i++)
		if (data[i] == '1')
			gamma[int(i / 32)] += (1 << (i % 32));

	return gamma;
}


bool gamma_cheack(const unsigned int* gamma, int n, unsigned int l1, unsigned int l2, unsigned int l3) {
	unsigned int l1_fb = 0b00110010100000000000000000000000;
	unsigned int l2_fb = 0b01001000000000000000000000000000;
	unsigned int l3_fb = 0b11110101000000000000000000000000;

	for (int i = 0; i < (int(n / 32)); i++) {
		unsigned int l1_gamma = 0;
		unsigned int l2_gamma = 0;
		unsigned int l3_gamma = 0;

		unsigned int out;

		for (int j = 0; j < 32; j++) {
			out = host_parity(l1 & l1_fb);
			l1_gamma += (((l1 >> 29) & 0x1) << j);
			l1 = (l1 << 1) ^ out;
			out = host_parity(l2 & l2_fb);
			l2_gamma += (((l2 >> 30) & 0x1) << j);
			l2 = (l2 << 1) ^ out;
			out = host_parity(l3 & l3_fb);
			l3_gamma += (((l3 >> 31) & 0x1) << j);
			l3 = (l3 << 1) ^ out;
		}

		unsigned int temp = (((l1_gamma ^ l2_gamma) & l3_gamma) ^ l2_gamma) ^ gamma[i];
		if (temp != 0)
			return false;
	}
	return true;
}


unsigned int host_parity(unsigned int n) {
	n = (n >> 16) ^ n;
	n = (n >> 8) ^ n;
	n = (n >> 4) ^ n;
	n = (n >> 2) ^ n;
	n = (n >> 1) ^ n;
	return (n & 0x1);
}


__global__ void l1_register_brute_force(const unsigned int* gamma, const unsigned int state_prefix) {
	unsigned int s = (state_prefix << 28) ^ (blockIdx.x << 5) ^ (threadIdx.x);
	unsigned int R = 0;

	if (s == 0)
		return;

	for (int i = 0; i < BLOCKS; i++) {
		unsigned int temp = 0;

		for (int j = 0; j < 32; j++) {
			unsigned int out = parity(s & l1_feedback);
			temp += (((s >> 29) & 0x1) << j);
			s = (s << 1) ^ out;
		}

		R += weigth(gamma[i] ^ temp);
	}

	if (R < C) {
		unsigned int i = atomicAdd(&l1_buffer_counter, 1);

		if (i >= 8)
			return;

		s = (state_prefix << 28) ^ (blockIdx.x << 5) ^ (threadIdx.x);
		l1_buffer[i] = s;

		for (int j = 0; j < BLOCKS; j++) {
			unsigned int temp = 0;

			for (int k = 0; k < 32; k++) {
				unsigned int out = parity(s & l1_feedback);
				temp += (((s >> 29) & 0x1) << k);
				s = (s << 1) ^ out;
			}

			l1_gamma_buffer[i][j] = temp;
		}
	}
}


__global__ void l2_register_brute_force(const unsigned int* gamma, const unsigned int state_prefix) {
	unsigned int s = (state_prefix << 28) ^ (blockIdx.x << 5) ^ (threadIdx.x);
	unsigned int R = 0;

	if (s == 0)
		return;

	for (int i = 0; i < BLOCKS; i++) {
		unsigned int temp = 0;

		for (int j = 0; j < 32; j++) {
			unsigned int out = parity(s & l2_feedback);
			temp += (((s >> 30) & 0x1) << j);
			s = (s << 1) ^ out;
		}

		R += weigth(gamma[i] ^ temp);
	}

	if (R < C) {
		unsigned int i = atomicAdd(&l2_buffer_counter, 1);
		if (i >= 24)
			return;

		s = (state_prefix << 28) ^ (blockIdx.x << 5) ^ (threadIdx.x);
		l2_buffer[i] = s;

		for (int j = 0; j < BLOCKS; j++) {
			unsigned int temp = 0;

			for (int k = 0; k < 32; k++) {
				unsigned int out = parity(s & l2_feedback);
				temp += (((s >> 30) & 0x1) << k);
				s = (s << 1) ^ out;
			}

			l2_gamma_buffer[i][j] = temp;
		}
	}
}


__global__ void l3_register_brute_force(const unsigned int* gamma, const unsigned int l1_index, const unsigned int l2_index, const unsigned int state_prefix) {
	unsigned int s = (state_prefix << 28) ^ (blockIdx.x << 5) ^ (threadIdx.x);
	if (s == 0)
		return;

	for (int i = 0; i < BLOCKS; i++) {
		unsigned int temp = 0;

		for (int j = 0; j < 32; j++) {
			unsigned int out = parity(s & l3_feedback);
			temp += (((s >> 31) & 0x1) << j);
			s = (s << 1) ^ out;
		}

		temp = (((l1_gamma_buffer[l1_index][i] ^ l2_gamma_buffer[l2_index][i]) & temp) ^ l2_gamma_buffer[l2_index][i]) ^ gamma[i];
		if (temp != 0)
			return;
	}

	unsigned int i = atomicAdd(&l3_buffer_counter, 1);
	if (i >= 16)
		return;

	l3_buffer[i] = (state_prefix << 28) ^ (blockIdx.x << 5) ^ (threadIdx.x);
}