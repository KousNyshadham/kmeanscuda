#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
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

void findNearestCentroids(int labels[], double dataset[], double centroids[]){
    for(int i = 0; i < n;i++){
		labels[i] = 0;
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
        //labels.push_back(label);
    }
}
void randomCentroids(double centroids[],double dataset[]){
	int count = 0;
    for (int i=0; i<num_cluster; i++){
        int index = kmeans_rand() % n;
        for (int j = 0; j < dims; j++){
			centroids[count] = dataset[index*dims+j];
			count++;
            //centroids.push_back(dataset[index*dims+j]);
        }
    }
}
void zeroArray(double centroids[]){
	for(int i = 0; i < num_cluster*dims;i++){
		centroids[i] = 0;
	}
}
void averageLabeledCentroids(double newcentroids[], double dataset[], int labels[]){
	double labelcounts[num_cluster]={0};
   for(int i = 0; i < n; i++){
       labelcounts[labels[i]] = labelcounts[labels[i]]+1;
       for(int j = 0; j < dims; j++){
        newcentroids[labels[i]*dims+j] += dataset[i*dims+j];
       }
   }

   for(int i = 0; i < num_cluster; i++){
       for(int j = 0; j < dims; j++){
           newcentroids[i*dims+j] = newcentroids[i*dims+j]/labelcounts[i];
       }
   }
}
bool converged(double oldCentroids[], double newCentroids[]){
	for(int i = 0; i < num_cluster*dims; i++){
		if(!(abs(oldCentroids[i] - newCentroids[i]) < threshold)){
			return false;
		}
	}
	return true;
}
void kmeans(double dataset[]){
	double centroids[num_cluster*dims]={0};
	int labels[n]={0};
    randomCentroids(centroids, dataset);
    int iterations = 0;
    bool done = false;
	double totalduration = 0;
    while(!done){
		auto start = high_resolution_clock::now();
		double oldCentroids[num_cluster*dims]={0};
        //vector<double> oldCentroids;
        for(int i = 0; i < num_cluster*dims; i++){
			oldCentroids[i] = centroids[i];
        }
        iterations++;
        findNearestCentroids(labels,dataset,centroids);
		zeroArray(centroids);
        averageLabeledCentroids(centroids,dataset,labels);
        done = (iterations == max_num_iter || converged(centroids, oldCentroids));
		auto stop = high_resolution_clock::now();
		double duration = duration_cast<microseconds>(stop - start).count(); 
		totalduration += duration;
    }
	double time_per_iter_in_ms = totalduration/iterations;
	auto iter_to_converge = iterations;
	printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);
	if(flag){
		for(int i = 0; i < num_cluster; i++){
			printf("%d ", i);
			for(int j = 0; j < dims; j++){
				printf("%lf ", centroids[i*dims +j]);
			}
			printf("\n");
		}
	}
	else{
		printf("clusters:");
		for (int p=0; p < n; p++)
			printf(" %d", labels[p]);		
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
	double dataset[n*dims]={0};
	int count = 0;
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
				dataset[count] = stod(value);
               //dataset.push_back(stod(value)); 
			   count++;
               value="";
            }
            else{
                value = value + line[j];
            }
        }
        if(value != ""){
			dataset[count] = stod(value);
			count++;
            //dataset.push_back(stod(value));
        }
    }
    kmeans(dataset);
}
