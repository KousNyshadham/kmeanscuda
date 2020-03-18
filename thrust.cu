#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>     
#include <iomanip>
using namespace std;
using namespace std::chrono; 

int n;
int num_cluster;
int dims;
string inputfilename;
int max_num_iter;
double threshold;
bool flag;
unsigned int seed = 8675309;

static unsigned long int nextstd = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    nextstd = nextstd * 1103515245 + 12345;
    return (unsigned int)(nextstd/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int funcseed) {
    nextstd = funcseed;
}

thrust::host_vector<double> hostdataset(0);
thrust::host_vector<double> hostcentroids(0);
thrust::host_vector<int> hostlabels(0);

struct findNearestCentroids{
	int dims;
	int num_cluster;
	double* dataset;
	double* centroids;
	int * labels;

	findNearestCentroids(int _dims, int _num_cluster, double* _dataset, double* _centroids, int* _labels):
	dims(_dims), num_cluster(_num_cluster), dataset(_dataset), centroids(_centroids), labels(_labels){};

	__host__ __device__
	void operator()(int i){
        int label = 0;
        double dist=0;
        double running = 0;
        for(int k = 0; k < dims; k++){
            running += pow(dataset[i*dims+k]-centroids[0*dims+k],2);
        }
        running = sqrt(running);
        dist = running;
        for(int j = 1; j < num_cluster; j++){
            double inner = 0;
            for(int k = 0; k < dims; k++){
                inner += pow(dataset[i*dims+k]-centroids[j*dims+k],2);
            }
            inner = sqrt(inner);
            if(inner < dist){
               label = j;
               dist = inner;
            }
        }
		labels[i] = label;
	}
};

struct converged{
	double threshold;
	double* oldCentroids;
	double* newCentroids;
	int* convergedVector;

	converged(double _threshold, double* _oldCentroids, double* _newCentroids, int* _convergedVector):
	threshold(_threshold), oldCentroids(_oldCentroids), newCentroids(_newCentroids), convergedVector(_convergedVector){};


	
	__host__ __device__
	void operator()(int i){
		if(!(abs(oldCentroids[i] - newCentroids[i]) < threshold)){
			convergedVector[i] = 1;
		}
		else{
			convergedVector[i] = 0;
		}
	}
};


void randomCentroids(){
	int counter = 0;
    for (int i=0; i<num_cluster; i++){
        int index = kmeans_rand() % n;
        for (int j = 0; j < dims; j++){
			hostcentroids[counter] = hostdataset[index*dims+j];
			counter++;
        }
    }
}

struct newadd{
	int label;
	int dims;
	double* newcentroids;
	double* dataset;

	newadd(int _dims, int _label, double* _centroids, double* _dataset):	
	label(_label), newcentroids(_centroids), dataset(_dataset), dims(_dims){};

	__host__ __device__
	void operator()(int i){
		for(int j = 0; j < dims; j++){
			newcentroids[label*dims+j] += dataset[i*dims+j];	
		}
	}
	
};

struct bruh{
	int dims;
	int n;
	int num_cluster;
	int* indices;
	int* prefixSum;
	double* centroids;
	double* dataset;

	bruh(int _n, int _dims, int _num_cluster, int* _indices, int* _prefixSum, double* _centroids, double* _dataset):
	n(_n), dims(_dims),num_cluster(_num_cluster), indices(_indices), prefixSum(_prefixSum), centroids(_centroids), dataset(_dataset){};

	__host__ __device__
	void operator()(int i){
		if(i == num_cluster-1){
			thrust::for_each(indices + prefixSum[i], indices+n,newadd(dims, i, centroids, dataset));
		}
		else{
			thrust::for_each(indices + prefixSum[i], indices+prefixSum[i+1],newadd(dims, i, centroids, dataset));
		}
		
	}

};

void kmeans(){
	thrust::device_vector<double> dataset(n*dims);
	thrust::device_vector<double> centroids(num_cluster*dims);
	thrust::device_vector<int> labels(n);
	hostcentroids.resize(dims*num_cluster);
	hostlabels.resize(n);
    randomCentroids();
	dataset = hostdataset;
	centroids = hostcentroids;
    int iterations = 0;
    bool done = false;
	//double totalduration = 0;
    while(!done){
		//auto start = high_resolution_clock::now();

		//
		//
		// FIND NEAREST CENTROIDS 
		//
		//
		thrust::device_vector<double> oldcentroids = centroids;
        iterations++;
		thrust::fill(labels.begin(),labels.end(), 0);
		thrust::device_vector<int> temp(n);
		thrust::sequence(temp.begin(),temp.end());
		thrust::for_each(temp.begin(), temp.end(), findNearestCentroids(dims, num_cluster, thrust::raw_pointer_cast(dataset.data()), thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(labels.data())));
		//
		//
		// AVERAGE LABELED CENTROIDS
		//
		//
		thrust::device_vector<int> indices(n);
		thrust::sequence(indices.begin(), indices.end());
		thrust::sort_by_key(labels.begin(), labels.end(), indices.begin());
		thrust::device_vector<int> ones(n);
		thrust::fill(ones.begin(),ones.end(), 1);
		thrust::device_vector<int> uniqueLabels(num_cluster);
		thrust::device_vector<int> labelsCount(num_cluster);
		thrust::reduce_by_key(labels.begin(), labels.end(), ones.begin(), uniqueLabels.begin(), labelsCount.begin());
		thrust::device_vector<int> prefixSum(num_cluster);
		thrust::exclusive_scan(labelsCount.begin(), labelsCount.end(), prefixSum.begin());
		thrust::fill(centroids.begin(), centroids.end(), 0);
		thrust::device_vector<int> clusters(num_cluster);
		thrust::for_each(clusters.begin(), clusters.end(), bruh(n, dims,num_cluster, thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(prefixSum.data()), thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(dataset.data())));


	hostcentroids=centroids;
	hostlabels=labels;
		break;
			
		
        //centroids = averageLabeledCentroids();
		//
		//
		// CONVERGE CENTROIDS
		//
		//
		/*
		thrust::device_vector<int> convergedVector(num_cluster*dims);
		thrust::sequence(convergedVector.begin(), convergedVector.end())
		thrust::for_each(convergedVector.begin(), convergedVector.end(), converged(threshold, thrust::raw_pointer_cast(oldcentroids.data()),thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(convergedVector.data())));
		int sum = thrust::reduce(convergedVector.begin(), convergedVector.end(), (int) 0, thrust::plus<int>());
        done = (iterations == max_num_iter || sum == 0);
		*/
		//auto stop = high_resolution_clock::now();
		//double duration = duration_cast<microseconds>(stop - start).count(); 
		//totalduration += duration;
    }
	/*
	double time_per_iter_in_ms = totalduration/iterations;
	auto iter_to_converge = iterations;
	printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);
	*/
	if(flag){
		for(int i = 0; i < num_cluster; i++){
			printf("%d ", i);
			for(int j = 0; j < dims; j++){
				printf("%lf ", hostcentroids[i*dims +j]);
			}
			printf("\n");
		}
	}
	else{
		printf("clusters:");
		for (int p=0; p < n; p++)
			printf(" %d", hostlabels[p]);		
	}
}

int main(int argc, char* argv[]){
    string inputPath;
    for(int i = 1; i < argc; i++){
        string arg = string(argv[i]);
        if(arg == "-k"){
            num_cluster = stoi(argv[++i]);
        }
        if(arg == "-d"){
            dims = stoi(argv[++i]);
        }
        if(arg == "-i"){
            inputfilename = argv[++i]; 
        }
        if(arg == "-m"){
            max_num_iter = stoi(argv[++i]);
        }
        if(arg == "-t"){
            threshold = stod(argv[++i]);
        }
        if(arg == "-c"){
            flag = true;
        }
        if(arg == "-s"){
            seed = stoi(argv[++i]);
            kmeans_srand(seed);
        }
    }
    ifstream inputfile;
    inputfile.open(inputfilename);
    string stringn;
    getline(inputfile, stringn);
    n = stoi(stringn);
	hostdataset.resize(n*dims);
	int counter = 0;
    for(int i = 0; i < n; i++){
        string line;
        getline(inputfile,line);
        string value="";
        bool index = false;
        for(int j = 0; j < line.size(); j++){
            if(line[j] == ' '){
                if(index == false){
                    index = true;
                    value = "";
                    continue;
                }
				hostdataset[counter] = stod(value);
				counter++;
               value="";
            }
            else{
                value = value + line[j];
            }
        }
        if(value != ""){
			hostdataset[counter] = stod(value);
			counter++;
        }
    }
    kmeans();
}
