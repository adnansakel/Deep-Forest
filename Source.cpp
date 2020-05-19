#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>
#include <map>
#include <random>
#include <ctime>
#include <cmath>
#include <string>
using namespace cv;
using namespace std;


#define MAX_DEPTH  15
#define FILTER_SIZE 3
#define NUM_TRAINING_SAMPLES 99//99
#define NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE 300//300
#define IMAGE_DATA_DIMENSION 256
#define NUM_OF_FILTERS 1
#define NUM_OF_CLASSES 3
#define MIN_SAMPLE 5
#define NUM_OF_TREE 40//40
#define NUM_OF_TREE_TF 40//40
#define NUM_OF_TEST_SAMPLES 51//150
#define NUM_OF_THRESHOLD 10
#define ROUGH_NORMALIZATION 1
#define PATCH_SIZE 3
#define NUM_RAND_LOC_ON_IMAGE_PATCH 1//not used
#define MAX_FOREST_DEPTH 4
#define NUM_FEATURES_TO_BE_USED 3

double class_posterior[NUM_OF_CLASSES];
struct probability_info {
	int class_name;
	double class_postarior[NUM_OF_CLASSES];
	double max_prob;
};
struct path_info {
	int class_name;
	vector<vector<bool>> visited_nodes;//visited nodes by a single patch(pixel) at all trees
	vector<int> visited_leaf_ids;
};
vector<vector<int>>training_data;
vector<vector<int>>test_data;
vector<vector<int>>scaled_test_data;
vector<int>training_data_labels;
vector<vector<int>>scaled_training_data;
vector<vector<probability_info>>training_data_probability;
vector<vector<probability_info>>test_data_probability;
vector<vector<probability_info>>scaled_training_data_probability;
vector<vector<probability_info>>scaled_test_data_probability;
vector<vector<int>>training_data_path;
vector<vector<int>>test_data_path;
vector<vector<int>>scaled_training_data_path;
vector<vector<int>>scaled_test_data_path;
vector<vector<path_info>>training_data_path_visited_nodes;
vector<vector<path_info>>test_data_path_visited_nodes;
vector<vector<path_info>>scaled_training_data_path_visited_nodes;
vector<vector<path_info>>scaled_test_data_path_visited_nodes;
vector<vector<map<int, int>>>P_texton_hist_forest;
vector<vector<map<int, int>>>Q_texton_hist_forest;
int arr_corr[NUM_OF_CLASSES];
double arr_occur[NUM_OF_CLASSES];
int training_data_dist[NUM_OF_CLASSES];
int two_pow[MAX_DEPTH+1] = { 1,2,4,8,16,32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
int node_number_P[FILTER_SIZE][FILTER_SIZE];
int node_number_Q[FILTER_SIZE][FILTER_SIZE];
//double all_predictions[NUM_OF_TEST_SAMPLES][MAX_FOREST_DEPTH][NUM_OF_CLASSES];

int **filter_one;
int **filter_two;
int **filter_three;
int **filter_four;
int **filter_five;
int **filter_six;


int max_response = -9999999;//for debugging/testing
int min_response = 9999999;//for debugging/testing
long long rand_seed;
map<int, int>global_response;//for debugging/testing

							 //default_random_engine generator(getTickCount());
normal_distribution<double> row_distribution(14.0, 4.0);
int normal[29][29];

int tree_height = -1;
int tree_node_count = 0;
int tree_leaf_count = 0;
int largest_leaf_size[NUM_OF_TREE];
int largest_leaf_height[NUM_OF_TREE];
int shortest_leaf_height = 9999;
int largest_leaf_with_data_samples = 0;

struct Prediction_confidence_class {
	double confidences[NUM_OF_CLASSES];
	double max_confidence;
	int predicted_class;
	int actual_class;
} predictions_in_diff_layers[MAX_FOREST_DEPTH][NUM_OF_TEST_SAMPLES];
struct PatchLocation {
	int sample_index;
	int pixel_position;
};

class Node {
	//int histogram[10];//holds the number of samples in each class.
public:
	Node() {}
	Node *Left, *Right;
	int index;// will be removed later
	int imleaf = 0;// indicates a leaf node
	vector<PatchLocation> samples;//stores the index of splitted data samples
	int feature_index;//image index that gave maximum information gain or minimum gini index
	int split_loc;
	vector<int> histogram;
	double threshold;
	int impurity = 1;
	int **feature_filter;
	//map<int, int> histogram_filter_response;
	double classProbability[NUM_OF_CLASSES];
	int f_type; // 0 for intensity, 1 for probability estimate and 2 for the path.
	string path = "1";
	vector<bool> visited_nodes;
	int visited_leaf_id = 1;
	int path_num_of_left_shift = 0;
	int path_num_of_right_shift = 1;
	int leaf_size = 0;
	int leaf_height = 9999;
	int class_id_for_probability = 0;
	//int node_id = 0;
	PatchLocation texton_hist_position;
	vector<map<int, int>>texton_hist_forest;
};
struct SplittedSamples {
	vector<PatchLocation>left_samples;//holds indexes of the data
	vector<PatchLocation>right_samples;
	vector<int> left_histogram;
	vector<int> right_histogram;
	double threshold;
	PatchLocation texton_hist_position;
	vector<map<int, int>> texton_hist;
	int **filter;
	//map<int,int> histogram_filter_response;
	double gini;
	int split_loc;
	int class_id_for_probability;
};

struct Samples {
	int sample_index;
	int classname;
}sortedSamples[NUM_TRAINING_SAMPLES];

bool comp(struct Samples a, struct Samples b) {
	return a.classname < b.classname;
}
/*void split(Node *node, int depth, int index) {
if (depth == 4) return;
node->index = index;
node->Left = split()
}*/

double pathInfoKernel(PatchLocation P, PatchLocation Q, vector<vector<path_info>>& l_training_data) {
	if (P.sample_index == Q.sample_index && P.pixel_position == Q.pixel_position)return 0;

	map<int, int> P_texton_hist;
	map<int, int> Q_texton_hist;
	vector<int> unique_nodes;
	int image_size = 0;
	if (l_training_data[0].size() > 0) {
		image_size = (int)sqrt(l_training_data[0].size() - 1);
	}
	//else {  }
	int row_p = P.pixel_position / image_size;
	int col_p = P.pixel_position % image_size;
	int row_q = Q.pixel_position / image_size;
	int col_q = Q.pixel_position % image_size;
	int r = 0;// split_loc / PATCH_SIZE;
	int c = 0;// split_loc % PATCH_SIZE;
	double response = 0;
	r = 0;
	c = 0;
	//int t = 0;
	int len_p = 0;
	int len_q = 0;
	int node_number;
	int d = 0;
	int node;
	int h_insertion = 0;
	int prev_h_insertion = 0;
	double sum = 0;
	for (int t = 0; t < NUM_OF_TREE; t++) {
		unique_nodes.clear(); unique_nodes.shrink_to_fit();
		P_texton_hist.clear();
		Q_texton_hist.clear();
		row_p = P.pixel_position / image_size;
		col_p = P.pixel_position % image_size;
		row_q = Q.pixel_position / image_size;
		col_q = Q.pixel_position % image_size;
		for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
			for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
				//if (r + f_i > PATCH_SIZE || c + f_j > PATCH_SIZE) { continue; }
				if ((row_p + r + f_i)*image_size + (col_p + c + f_j) < l_training_data[0].size() && (row_p + r + f_i)*image_size + (col_p + c + f_j) >= 1) {
					len_p = l_training_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_nodes[t].size();
					for (int i = 0; i < len_p; i++) {
						//cout << "H" << endl;
						node_number = l_training_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_nodes[t][i];
						//cout << "H!" << endl;
						if (P_texton_hist[node_number] == 0 && node_number != 0) {
							unique_nodes.push_back(node_number);
						}
						P_texton_hist[node_number]++;
					}

					
					//response =  * filter[f_i][f_j];
				}
				if ((row_q + r + f_i)*image_size + (col_q + c + f_j) < l_training_data[0].size() && (row_q + r + f_i)*image_size + (col_q + c + f_j) >= 1) {
					len_q = l_training_data[Q.sample_index][(row_q + r + f_i)*image_size + (col_q + c + f_j)].visited_nodes[t].size();
					//cout << "H!!" << endl;
					for (int i = 0; i < len_q; i++) {
						Q_texton_hist[l_training_data[Q.sample_index][(row_q + r + f_i)*image_size + (col_q + c + f_j)].visited_nodes[t][i]]++;
						//cout << "H!!!" << endl;
					}
				}

			}
		}

		/*
		row = Q.pixel_position / image_size;
		col = Q.pixel_position % image_size;
		for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
			for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
				//if (r + f_i > PATCH_SIZE || c + f_j > PATCH_SIZE) { continue; }
				if ((row + r + f_i)*image_size + (col + c + f_j) < l_training_data[0].size() && (row + r + f_i)*image_size + (col + c + f_j) >= 1) {
					len = l_training_data[Q.sample_index][(row + r + f_i)*image_size + (col + c + f_j)].visited_nodes[t].size();
					//cout << "H!!" << endl;
					for (int i = 0; i < len; i++) {
						Q_texton_hist[l_training_data[Q.sample_index][(row + r + f_i)*image_size + (col + c + f_j)].visited_nodes[t][i]]++;
						//cout << "H!!!" << endl;
					}
					//response =  * filter[f_i][f_j];
				}
			}
		}*/
		//cout << "HHH" << endl;
		sort(unique_nodes.begin(), unique_nodes.end());
		//cout << "HHH" << endl;
		//d = 0;
		node;
		h_insertion = 0;
		prev_h_insertion = 0;
		sum = 0;
		int D = 0;
		if (unique_nodes.size() > 0) { (int)log2((double)unique_nodes[unique_nodes.size() - 1]); }
		int n = 0;
		for (int d = 0; d < D; d++) {
			if (n > unique_nodes.size() - 1) { break; }
			while (unique_nodes[n] < 2 * (d+1) + 1 && n < unique_nodes.size()) {
				node = unique_nodes[n];
				if (P_texton_hist[node] < Q_texton_hist[node]) {
					//cout << "HHHH!!" << endl;
					h_insertion += P_texton_hist[node];
				}
				else {
					//cout << "HHHH!!!" << endl;
					h_insertion += Q_texton_hist[node];
				}
				n++;
				if (n > unique_nodes.size() - 1) { break; }
			}

			if (d > 0) {
				sum += ((double)(prev_h_insertion - h_insertion)) / pow(2, D - d + 1);
			}
			prev_h_insertion = h_insertion;
			h_insertion = 0;

		}

		response += sum;
	}

	return response;
}
double pathInfoKernel_test(PatchLocation P, PatchLocation Q, vector<vector<path_info>>& l_test_data, vector<vector<path_info>>& l_training_data) {
	//if (P.sample_index == Q.sample_index && P.pixel_position == Q.pixel_position)return 0;
	//cout << "!!" << endl;
	map<int, int> P_texton_hist;
	map<int, int> Q_texton_hist;
	vector<int> unique_nodes;
	int image_size  = 0;
	if (l_training_data[0].size() > 0) { image_size = (int)sqrt(l_training_data[0].size() - 1); }
	int row_p = P.pixel_position / image_size;
	int col_p = P.pixel_position % image_size;
	int row_q = Q.pixel_position / image_size;
	int col_q = Q.pixel_position % image_size;
	int r = 0;// split_loc / PATCH_SIZE;
	int c = 0;// split_loc % PATCH_SIZE;
	double response = 0;
	r = 0;
	c = 0;
	//int t = 0;
	int len_p = 0;
	int len_q = 0;
	int node_number;
	int d = 0;
	int node;
	int h_insertion = 0;
	int prev_h_insertion = 0;
	double sum = 0;
	for (int t = 0; t < NUM_OF_TREE; t++) {
		unique_nodes.clear(); unique_nodes.shrink_to_fit();
		P_texton_hist.clear();
		Q_texton_hist.clear();
		row_p = P.pixel_position / image_size;
		col_p = P.pixel_position % image_size;
		row_q = Q.pixel_position / image_size;
		col_q = Q.pixel_position % image_size;
		for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
			for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
				//if (r + f_i > PATCH_SIZE || c + f_j > PATCH_SIZE) { continue; }
				if ((row_p + r + f_i)*image_size + (col_p + c + f_j) < l_test_data[0].size() && (row_p + r + f_i)*image_size + (col_p + c + f_j) >= 1) {
					len_p = l_test_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_nodes[t].size();
					for (int i = 0; i < len_p; i++) {
						node_number = l_test_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_nodes[t][i];
						if (P_texton_hist[node_number] == 0 && node_number != 0) {
							unique_nodes.push_back(node_number);
						}
						P_texton_hist[node_number]++;
					}
					
					//response =  * filter[f_i][f_j];
				}
				if ((row_q + r + f_i)*image_size + (col_q + c + f_j) < l_training_data[0].size() && (row_q + r + f_i)*image_size + (col_q + c + f_j) >= 1) {
					len_q = l_training_data[Q.sample_index][(row_q + r + f_i)*image_size + (col_q + c + f_j)].visited_nodes[t].size();
					for (int i = 0; i < len_q; i++) {
						Q_texton_hist[l_training_data[Q.sample_index][(row_q + r + f_i)*image_size + (col_q + c + f_j)].visited_nodes[t][i]]++;
					}
				}
			}
		}
		//cout << "++" << endl;
		sort(unique_nodes.begin(), unique_nodes.end());
		//cout << "--" << endl;
		node;
		h_insertion = 0;
		prev_h_insertion = 0;
		sum = 0;
		int D = 0;
		if (unique_nodes.size() > 0) { D = (int)log2((double)unique_nodes[unique_nodes.size() - 1]);}
		int n = 0;
		for (int d = 0; d < D; d++) {
			if (n > unique_nodes.size() - 1) { break; }
			while (unique_nodes[n] < 2 * (d + 1) + 1 ) {
				node = unique_nodes[n];
				//n++;
				if (P_texton_hist[node] < Q_texton_hist[node]) {
					//cout << "HHHH!!" << endl;
					h_insertion += P_texton_hist[node];
				}
				else {
					//cout << "HHHH!!!" << endl;
					h_insertion += Q_texton_hist[node];
				}
				n++;
				if (n > unique_nodes.size() - 1) { break; }
			}
			//cout << "HHHH!!!!" << endl;

			if (d > 0) {
				sum += ((double)(prev_h_insertion - h_insertion)) / pow(2, D - d + 1);
			}
			prev_h_insertion = h_insertion;
			h_insertion = 0;

		}

		response += sum;
	}
	//cout << "??" << endl;
	return response;
}

void populateTextonHistForest(PatchLocation P, vector<vector<path_info>>& l_data, vector<map<int, int>>&texton_hist_forest) {
	//if (l_data.size() == 0)return;
	map<int, int> P_texton_hist;
	vector<int> unique_nodes;
	int image_size = (int)sqrt(l_data[0].size() - 1);
	int row_p = P.pixel_position / image_size;
	int col_p = P.pixel_position % image_size;
	//map<int, int>mp;
	int r = 0;// split_loc / PATCH_SIZE;
	int c = 0;// split_loc % PATCH_SIZE;
	double response = 0;
	r = 0;
	c = 0;
	//int t = 0;
	int len_p = 0;
	int node;
	//int node_number;
	map<int, int>texton_hist_tree;

	for (int t = 0; t < NUM_OF_TREE; t++) {
		P_texton_hist.clear();
		texton_hist_tree.clear();
		
		unique_nodes.clear();
		row_p = P.pixel_position / image_size;
		col_p = P.pixel_position % image_size;
		
		for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
			for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
				if ((row_p + r + f_i)*image_size + (col_p + c + f_j) < l_data[0].size() && (row_p + r + f_i)*image_size + (col_p + c + f_j) >= 1) {
					node = l_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_leaf_ids[t];
					if (P_texton_hist[node] == 0) {
						unique_nodes.push_back(node);
					}
					P_texton_hist[node]++;
				}
			}
		}

		for (int n = 0; n < unique_nodes.size(); n++) {
			texton_hist_tree[unique_nodes[n]] = P_texton_hist[unique_nodes[n]];
			
			//texton_hist_tree.push_back(mp);
		}

		texton_hist_forest.push_back(texton_hist_tree);

	}


}
double getTextonHistogramComparisonWithLeafID(vector<map<int, int>>&P_texton_hist_f, vector<map<int, int>>&Q_texton_hist_f) {
	if (P_texton_hist_f.size() == 0 || Q_texton_hist_f.size() == 0) { return 0; }
	double response = 0;
	double h_insertion = 0;
	int hist_size;
	double sum = 0;
	for (int t = 0; t < NUM_OF_TREE; t++) {
		h_insertion = 0;
		hist_size = 0;
		
		for (map<int, int>::iterator hist = P_texton_hist_f[t].begin(); hist != P_texton_hist_f[t].end(); ++hist) {
			hist_size++;// = hist->second;
			//cout << hist->first << " " << hist->second << " " << Q_texton_hist_f[t][hist->first] << endl;
			if (hist->second < Q_texton_hist_f[t][hist->first]) {
				//cout << "HHHH!!" << endl;
				h_insertion += hist->second;
			}
			else {
				//cout << "HHHH!!!" << endl;
				h_insertion += Q_texton_hist_f[t][hist->first];
			}
		}

		if (hist_size > 0) { sum += h_insertion / (hist_size*FILTER_SIZE*FILTER_SIZE); }

	}

	if (sum > 0) { response = sum / NUM_OF_TREE*100; }
	//cout << response << " "<<endl;
	return response;
}
double getTextonHistogramComparisonWithLeafID_test(PatchLocation P, PatchLocation Q, vector<vector<path_info>>& l_test_data, vector<vector<path_info>>& l_training_data) {
	if (P.sample_index == Q.sample_index && P.pixel_position == Q.pixel_position)return 100;

	map<int, int> P_texton_hist;
	map<int, int> Q_texton_hist;
	vector<int> unique_nodes;
	int image_size = (int)sqrt(l_training_data[0].size() - 1);
	int row_p = P.pixel_position / image_size;
	int col_p = P.pixel_position % image_size;
	int row_q = Q.pixel_position / image_size;
	int col_q = Q.pixel_position % image_size;
	int r = 0;// split_loc / PATCH_SIZE;
	int c = 0;// split_loc % PATCH_SIZE;
	double response = 0;
	r = 0;
	c = 0;
	//int t = 0;
	int len_p = 0;
	int len_q = 0;
	//int node_number;
	int d = 0;
	int node;
	double h_insertion = 0;
	int prev_h_insertion = 0;
	double sum = 0;

	for (int t = 0; t < NUM_OF_TREE_TF; t++) {
		//unique_nodes.clear(); unique_nodes.shrink_to_fit();
		P_texton_hist.clear();
		Q_texton_hist.clear();
		unique_nodes.clear();
		row_p = P.pixel_position / image_size;
		col_p = P.pixel_position % image_size;
		row_q = Q.pixel_position / image_size;
		col_q = Q.pixel_position % image_size;

		for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
			for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
				if ((row_p + r + f_i)*image_size + (col_p + c + f_j) < l_training_data[0].size() && (row_p + r + f_i)*image_size + (col_p + c + f_j) >= 1) {
					node = l_training_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_leaf_ids[t];
					if (P_texton_hist[node] == 0) {
						unique_nodes.push_back(node);
					}
					P_texton_hist[node]++;

				}

				if ((row_p + r + f_i)*image_size + (col_p + c + f_j) < l_test_data[0].size() && (row_p + r + f_i)*image_size + (col_p + c + f_j) >= 1) {
					node = l_test_data[Q.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_leaf_ids[t];
					Q_texton_hist[node]++;
				}


			}
		}

		h_insertion = 0;

		for (int n = 0; n < unique_nodes.size(); n++) {
			if (P_texton_hist[unique_nodes[n]] < Q_texton_hist[unique_nodes[n]]) {
				//cout << "HHHH!!" << endl;
				h_insertion += P_texton_hist[unique_nodes[n]];
			}
			else {
				//cout << "HHHH!!!" << endl;
				h_insertion += Q_texton_hist[unique_nodes[n]];
			}
		}

		if (unique_nodes.size() > 0) { sum += h_insertion / unique_nodes.size(); }

	}

	if (sum != 0) { response += sum / (FILTER_SIZE*FILTER_SIZE); }

	if (response != 0) {
		response = response / NUM_OF_TREE_TF * 100;
	}

	return response;

}

double getTextonHistogramComparisonWithLeafID(PatchLocation P, PatchLocation Q, vector<vector<path_info>>& l_training_data) {
	if (P.sample_index == Q.sample_index && P.pixel_position == Q.pixel_position)return 100;

	map<int, int> P_texton_hist;
	map<int, int> Q_texton_hist;
	vector<int> unique_nodes;
	int image_size = (int)sqrt(l_training_data[0].size() - 1);
	int row_p = P.pixel_position / image_size;
	int col_p = P.pixel_position % image_size;
	int row_q = Q.pixel_position / image_size;
	int col_q = Q.pixel_position % image_size;
	int r = 0;// split_loc / PATCH_SIZE;
	int c = 0;// split_loc % PATCH_SIZE;
	double response = 0;
	r = 0;
	c = 0;
	//int t = 0;
	int len_p = 0;
	int len_q = 0;
	//int node_number;
	int d = 0;
	int node;
	double h_insertion = 0;
	int prev_h_insertion = 0;
	double sum = 0;

	for (int t = 0; t < NUM_OF_TREE_TF; t++) {
		//unique_nodes.clear(); unique_nodes.shrink_to_fit();
		P_texton_hist.clear();
		Q_texton_hist.clear();
		unique_nodes.clear();
		row_p = P.pixel_position / image_size;
		col_p = P.pixel_position % image_size;
		row_q = Q.pixel_position / image_size;
		col_q = Q.pixel_position % image_size;

		for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
			for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
				if ((row_p + r + f_i)*image_size + (col_p + c + f_j) < l_training_data[0].size() && (row_p + r + f_i)*image_size + (col_p + c + f_j) >= 1) {
					node = l_training_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_leaf_ids[t];
					if (P_texton_hist[node] == 0) {
						unique_nodes.push_back(node);
					}
					P_texton_hist[node]++;

					node = l_training_data[Q.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_leaf_ids[t];
					Q_texton_hist[node]++;
				}


			}
		}

		h_insertion = 0;

		for (int n = 0; n < unique_nodes.size(); n++) {
			if (P_texton_hist[unique_nodes[n]] < Q_texton_hist[unique_nodes[n]]) {
				//cout << "HHHH!!" << endl;
				h_insertion += P_texton_hist[unique_nodes[n]];
			}
			else {
				//cout << "HHHH!!!" << endl;
				h_insertion += Q_texton_hist[unique_nodes[n]];
			}
		}

		if (unique_nodes.size() > 0) { sum += h_insertion / unique_nodes.size(); }

	}

	if (sum != 0) { response += sum / (FILTER_SIZE*FILTER_SIZE); }

	if (response != 0) {
		response = response / NUM_OF_TREE_TF * 100;
	}

	return response;

}

double getTextonHistogramComparison_test(PatchLocation P, PatchLocation Q, vector<vector<path_info>>& l_test_data, vector<vector<path_info>>& l_training_data) {
	//if (P.sample_index == Q.sample_index && P.pixel_position == Q.pixel_position)return 100;
	//cout << "!!" << endl;
	map<int, int> P_texton_hist;
	map<int, int> Q_texton_hist;
	vector<int> unique_nodes;
	int image_size = 0;
	if (l_training_data[0].size() > 0) { image_size = (int)sqrt(l_training_data[0].size() - 1); }
	int row_p = P.pixel_position / image_size;
	int col_p = P.pixel_position % image_size;
	int row_q = Q.pixel_position / image_size;
	int col_q = Q.pixel_position % image_size;
	int r = 0;// split_loc / PATCH_SIZE;
	int c = 0;// split_loc % PATCH_SIZE;
	double response = 0;
	r = 0;
	c = 0;
	//int t = 0;
	int len_p = 0;
	int len_q = 0;
	//int node_number;
	int d = 0;
	int node;
	double h_insertion = 0;
	int prev_h_insertion = 0;
	double sum = 0;
	for (int t = 0; t < NUM_OF_TREE_TF; t++) {
		//unique_nodes.clear(); unique_nodes.shrink_to_fit();
		P_texton_hist.clear();
		Q_texton_hist.clear();
		row_p = P.pixel_position / image_size;
		col_p = P.pixel_position % image_size;
		row_q = Q.pixel_position / image_size;
		col_q = Q.pixel_position % image_size;

		for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
			for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
				node_number_P[f_i][f_j] = 1;
				node_number_Q[f_i][f_j] = 1;
			}
		}
		
		sum = 0;
		int depth_norm = 0;
		for (int d = 0; d < MAX_DEPTH; d++) {

			unique_nodes.clear(); unique_nodes.shrink_to_fit();

			for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
				for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
					//if (r + f_i > PATCH_SIZE || c + f_j > PATCH_SIZE) { continue; }
					if ((row_p + r + f_i)*image_size + (col_p + c + f_j) < l_training_data[0].size() && (row_p + r + f_i)*image_size + (col_p + c + f_j) >= 1) {
						//cout << "H" << endl;
						if (l_training_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_nodes[t].size() > d + 1) {
							int bit = l_training_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_nodes[t][d + 1];
							node_number_P[f_i][f_j] = node_number_P[f_i][f_j] + bit*two_pow[d + 1];

							//cout << "H!" << endl;
							if (P_texton_hist[node_number_P[f_i][f_j]] == 0) {
								unique_nodes.push_back(node_number_P[f_i][f_j]);
							}
							P_texton_hist[node_number_P[f_i][f_j]]++;
						}
						
						


						//response =  * filter[f_i][f_j];
					}
					if ((row_q + r + f_i)*image_size + (col_q + c + f_j) < l_test_data[0].size() && (row_q + r + f_i)*image_size + (col_q + c + f_j) >= 1) {
						if (l_test_data[Q.sample_index][(row_q + r + f_i)*image_size + (col_q + c + f_j)].visited_nodes[t].size() > d + 1) {

							node_number_Q[f_i][f_j] = node_number_Q[f_i][f_j] + l_test_data[Q.sample_index][(row_q + r + f_i)*image_size + (col_q + c + f_j)].visited_nodes[t][d + 1] * two_pow[d + 1];
							Q_texton_hist[node_number_Q[f_i][f_j]]++;

						}
						
					}

				}
			}

			
			h_insertion = 0;

			for (int n = 0; n < unique_nodes.size(); n++) {
				if (P_texton_hist[unique_nodes[n]] < Q_texton_hist[unique_nodes[n]]) {
					//cout << "HHHH!!" << endl;
					h_insertion += P_texton_hist[unique_nodes[n]];
				}
				else {
					//cout << "HHHH!!!" << endl;
					h_insertion += Q_texton_hist[unique_nodes[n]];
				}
			}

			if (unique_nodes.size() > 0) { sum += h_insertion / unique_nodes.size(); }

		}

		if (sum != 0) { 
			response += sum / (FILTER_SIZE*FILTER_SIZE*MAX_DEPTH); 
		}



	}

	if (response != 0) {
		return response / NUM_OF_TREE_TF * 100;
	}
	return response;
}
double getTextonHistogramComparison(PatchLocation P, PatchLocation Q, vector<vector<path_info>>& l_training_data) {
	if (P.sample_index == Q.sample_index && P.pixel_position == Q.pixel_position)return 100;

	map<int, int> P_texton_hist;
	map<int, int> Q_texton_hist;
	vector<int> unique_nodes;
	int image_size = (int)sqrt(l_training_data[0].size() - 1);
	int row_p = P.pixel_position / image_size;
	int col_p = P.pixel_position % image_size;
	int row_q = Q.pixel_position / image_size;
	int col_q = Q.pixel_position % image_size;
	int r = 0;// split_loc / PATCH_SIZE;
	int c = 0;// split_loc % PATCH_SIZE;
	double response = 0;
	r = 0;
	c = 0;
	//int t = 0;
	int len_p = 0;
	int len_q = 0;
	//int node_number;
	int d = 0;
	int node;
	double h_insertion = 0;
	int prev_h_insertion = 0;
	double sum = 0;
	for (int t = 0; t < NUM_OF_TREE_TF; t++) {
		//unique_nodes.clear(); unique_nodes.shrink_to_fit();
		P_texton_hist.clear();
		Q_texton_hist.clear();
		row_p = P.pixel_position / image_size;
		col_p = P.pixel_position % image_size;
		row_q = Q.pixel_position / image_size;
		col_q = Q.pixel_position % image_size;

		for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
			for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
				node_number_P[f_i][f_j] = 1;
				node_number_Q[f_i][f_j] = 1;
			}
		}

		sum = 0;
		int depth_norm = 0;
		for (int d = 0; d < MAX_DEPTH; d++) {

			unique_nodes.clear(); unique_nodes.shrink_to_fit();

			for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
				for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
					//if (r + f_i > PATCH_SIZE || c + f_j > PATCH_SIZE) { continue; }
					if ((row_p + r + f_i)*image_size + (col_p + c + f_j) < l_training_data[0].size() && (row_p + r + f_i)*image_size + (col_p + c + f_j) >= 1) {
							//cout << d<< "H" << endl;
							if (l_training_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_nodes[t].size() > d + 1) {
								int bit = l_training_data[P.sample_index][(row_p + r + f_i)*image_size + (col_p + c + f_j)].visited_nodes[t][d + 1];
								node_number_P[f_i][f_j] = node_number_P[f_i][f_j] + bit*two_pow[d + 1];

								//cout << d << "H!" << endl;
								if (P_texton_hist[node_number_P[f_i][f_j]] == 0) {
									unique_nodes.push_back(node_number_P[f_i][f_j]);
								}
								P_texton_hist[node_number_P[f_i][f_j]]++;
							}
						
						


						//response =  * filter[f_i][f_j];
					}
					if ((row_q + r + f_i)*image_size + (col_q + c + f_j) < l_training_data[0].size() && (row_q + r + f_i)*image_size + (col_q + c + f_j) >= 1) {
						//len_q = l_training_data[Q.sample_index][(row_q + r + f_i)*image_size + (col_q + c + f_j)].visited_nodes[t].size();
						//cout <<d << "H!!" << endl;
						
						if(l_training_data[Q.sample_index][(row_q + r + f_i)*image_size + (col_q + c + f_j)].visited_nodes[t].size() > d+1){
							node_number_Q[f_i][f_j] = node_number_Q[f_i][f_j] + l_training_data[Q.sample_index][(row_q + r + f_i)*image_size + (col_q + c + f_j)].visited_nodes[t][d + 1] * two_pow[d + 1];
							Q_texton_hist[node_number_Q[f_i][f_j]]++;

						}
						//	cout <<d << "H!!!" << endl;
						
					}

				}
			}

			
			

			h_insertion = 0;
			
			for (int n = 0; n < unique_nodes.size(); n++) {
				if (P_texton_hist[unique_nodes[n]] < Q_texton_hist[unique_nodes[n]]) {
					//cout << "HHHH!!" << endl;
					h_insertion += P_texton_hist[unique_nodes[n]];
				}
				else {
					//cout << "HHHH!!!" << endl;
					h_insertion += Q_texton_hist[unique_nodes[n]];
				}
			}
			
			if (unique_nodes.size() > 0) { sum += h_insertion / unique_nodes.size(); }

		}

		if (sum != 0) { response += sum / (FILTER_SIZE*FILTER_SIZE*MAX_DEPTH); }
		
		

	}

	if (response != 0) {
		return response / NUM_OF_TREE_TF * 100;
	}
	return response;
}
double getNewPathFeature(vector<PatchLocation>samples_index, PatchLocation P, vector<vector<path_info>>& l_training_data) {
	//cout << "Hello"<<endl;
	double sum = 0;
	for (int i = 0; i < samples_index.size(); i++) {
		sum += pathInfoKernel(P, samples_index.at(i), l_training_data);
	}
	
	return sum;
}
double getNewPathFeature_test(vector<PatchLocation>samples_index, PatchLocation P, vector<vector<path_info>>& l_test_data, vector<vector<path_info>>& l_training_data) {
	double sum = 0;
	for (int i = 0; i < samples_index.size(); i++) {
		sum += pathInfoKernel_test(P, samples_index.at(i), l_test_data, l_training_data);
	}

	return sum;
}
double getGini(vector<int> left_histogram, vector<int> right_histogram, int left_sample_size, int right_sample_size) {
	double gini_l = 0;
	double gini_r = 0;
	for (int i = 0; i < left_histogram.size(); i++) {
		if (left_sample_size != 0) { gini_l += ((double)left_histogram[i] / left_sample_size)*((double)1 - (double)left_histogram[i] / left_sample_size); }
		if (right_sample_size != 0) { gini_r += ((double)right_histogram[i] / right_sample_size)*((double)1 - (double)right_histogram[i] / right_sample_size); }
	}
	if (left_sample_size + right_sample_size != 0) {
		gini_l = gini_l*((double)left_sample_size / (left_sample_size + right_sample_size));
		gini_r = gini_r*((double)right_sample_size / (left_sample_size + right_sample_size));
	}
	return gini_l + gini_r;
}

/*
Following method considers histogram got from the filter response on each image for different points on image
*/
double getGini(map<int, int> &histogram_left, map<int, int> &histogram_right, int size_left, int size_right) {
	double gini_l = 0;
	double gini_r = 0;
	double p_l, p_r;
	//cout << "Histogram Left:" << endl;
	/*
	for (int i = 0; i < 36000 / ROUGH_NORMALIZATION; i++) {
	//cout << hist->first << ": " << hist->second << endl;
	//if (hist->first == 0)continue;
	if (size_left != 0) {
	p_l = (double)histogram_left[i] / (size_left);
	gini_l += p_l*(1 - p_l);
	}

	if (size_right != 0) {
	p_r = (double)histogram_right[i] / (size_right);

	gini_r += p_r*(1 - p_r);
	}

	}
	*/

	for (map<int, int>::iterator hist = histogram_left.begin(); hist != histogram_left.end(); ++hist) {
		if (size_left != 0) {
			p_l = (double)hist->second / (size_left);
			gini_l += p_l*(1 - p_l);
		}
	}

	for (map<int, int>::iterator hist = histogram_right.begin(); hist != histogram_right.end(); ++hist) {
		if (size_right != 0) {
			p_r = (double)hist->second / (size_right);
			gini_r += p_r*(1 - p_r);
		}
	}

	if (size_left + size_right != 0) {
		gini_l = gini_l*((double)size_left / (size_left + size_right));
		gini_r = gini_r*((double)size_right / (size_left + size_right));
	}

	return gini_l + gini_r;
}

double getGini(map<double, int> &histogram_left, map<double, int> &histogram_right, int size_left, int size_right) {
	double gini_l = 0;
	double gini_r = 0;
	double p_l, p_r;
	//cout << "Histogram Left:" << endl;
	/*
	for (int i = 0; i < 36000 / ROUGH_NORMALIZATION; i++) {
	//cout << hist->first << ": " << hist->second << endl;
	//if (hist->first == 0)continue;
	if (size_left != 0) {
	p_l = (double)histogram_left[i] / (size_left);
	gini_l += p_l*(1 - p_l);
	}

	if (size_right != 0) {
	p_r = (double)histogram_right[i] / (size_right);

	gini_r += p_r*(1 - p_r);
	}

	}
	*/

	for (map<double, int>::iterator hist = histogram_left.begin(); hist != histogram_left.end(); ++hist) {
		if (size_left != 0) {
			p_l = (double)hist->second / (size_left);
			gini_l += p_l*(1 - p_l);
		}
	}

	for (map<double, int>::iterator hist = histogram_right.begin(); hist != histogram_right.end(); ++hist) {
		if (size_right != 0) {
			p_r = (double)hist->second / (size_right);
			gini_r += p_r*(1 - p_r);
		}
	}

	if (size_left + size_right != 0) {
		gini_l = gini_l*((double)size_left / (size_left + size_right));
		gini_r = gini_r*((double)size_right / (size_left + size_right));
	}

	return gini_l + gini_r;
}

int getResponse(int patch_location, int response_location) {
	int res = 0;
	int row = patch_location / IMAGE_DATA_DIMENSION;
	int col = patch_location % IMAGE_DATA_DIMENSION;
	int i = response_location / PATCH_SIZE;
	int j = response_location % PATCH_SIZE;
	(row + i)*IMAGE_DATA_DIMENSION + (col + j);
	//if (res == 0)return 0;
	return (int)(round((double)res / ROUGH_NORMALIZATION));
}

SplittedSamples getSpittedSampleswithGini(int split_loc, vector<PatchLocation>samples_index, int **filter, vector<vector<path_info>>& l_training_data) {
	//cout << "From splitting function" << endl;
	struct SplittedSamples chosen_samples;
	map<double, int>histogram_filter_response_left;
	map<double, int>histogram_filter_response_right;
	histogram_filter_response_left.clear();
	histogram_filter_response_right.clear();
	vector<double> filter_response;
	vector<double> unique_filter_response;
	map<double, int>is_it_unique;
	int image_size = (int)sqrt(l_training_data[0].size() - 1);//lenghth and width of image//considering image is square sized
	int count_left = 0;
	int count_right = 0;
	double min_gini = 99999;
	//cout << "filter position: " << filter_loc<<" "<<filter_loc/IMAGE_DATA_DIMENSION <<","<<filter_loc%IMAGE_DATA_DIMENSION<<endl;
	vector<map<int, int>> P_hist; 
	vector<map<int, int>> Q_hist;
	//for (int cl = 0; cl < NUM_OF_CLASSES; cl++) {
		//int freq_resp_max = -1;
		//int freq_resp_max_val = -5;
		int bad_loc_count = 0;
		is_it_unique.clear();
		filter_response.clear(); filter_response.shrink_to_fit();
		unique_filter_response.clear(); unique_filter_response.shrink_to_fit();
		//global_response.clear();
		//cout << "OK !" << endl;
		//cout << "resp: " << freq_resp_max_val << " freq: " << freq_resp_max << endl;
		//rand_seed = getTickCount();
		//rand_seed += 123456789;
		//cout << "Choosing threshold " << rand_seed << endl;
		//srand(rand_seed);
		int num_of_threshold = (int)sqrt((double)samples_index.size());
		//num_of_threshold = unique_filter_response.size();
		for (int k = 0; k < num_of_threshold; k++) {
			int threshold_index = rand() % samples_index.size();
			unique_filter_response.clear(); unique_filter_response.shrink_to_fit();
			for (int mm = 0; mm < samples_index.size(); mm++) {
				//unique_filter_response.push_back(getTextonHistogramComparison(samples_index[threshold_index], samples_index[mm], l_training_data));
				//unique_filter_response.push_back(getTextonHistogramComparisonWithLeafID(samples_index[threshold_index], samples_index[mm], l_training_data));
				P_hist.clear(); P_hist.shrink_to_fit(); Q_hist.clear(); Q_hist.shrink_to_fit();
				populateTextonHistForest(samples_index[threshold_index],l_training_data, P_hist);
				populateTextonHistForest(samples_index[mm], l_training_data, Q_hist);
				unique_filter_response.push_back(getTextonHistogramComparisonWithLeafID(P_hist, Q_hist));
				
			}
			for (int thr = 0; thr < 10; thr++) {
				double threshold = unique_filter_response[rand() % unique_filter_response.size()];
				//threshold = rand() % 255;
				//cout << "Num of unique responses: "<< unique_filter_response.size()<<" Threshold index:"<<threshold_index<<" Threshold: " << threshold << endl;
				struct SplittedSamples samples;
				for (int i = 0; i < NUM_OF_CLASSES; i++) {
					samples.left_histogram.push_back(0);
					samples.right_histogram.push_back(0);
				}
				count_left = 0;
				count_right = 0;
				histogram_filter_response_left.clear();
				histogram_filter_response_right.clear();
				//cout << "response: ";
				for (int i = 0; i < samples_index.size(); i++) {

					//cout << unique_filter_response[i] << " ";

					if (unique_filter_response[i] < threshold) {
						//if (filter_response != 0) {
						//histogram_filter_response_left[filter_response[i]]++;
						//count_left++;
						//}

						samples.left_samples.push_back(samples_index.at(i));
						samples.left_histogram.at(l_training_data[samples_index.at(i).sample_index][0].class_name)++;
						samples.split_loc = split_loc;
					}
					else {
						//if (filter_response != 0) {
						//histogram_filter_response_right[filter_response[i]]++;
						//count_right++;
						//}
						samples.right_samples.push_back(samples_index.at(i));
						samples.right_histogram.at(l_training_data[samples_index.at(i).sample_index][0].class_name)++;
						samples.split_loc = split_loc;

					}
				}
				//cout << endl;
				//samples.gini = getGini(histogram_filter_response_left, histogram_filter_response_right, count_left, count_right);// passing histogrms of feature labels
				samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());// passing histogram of class labels
																																				   //cout << "left: " << count_left << " Right: " << count_right << " gini: "<<samples.gini<<endl;
				if (samples.gini < min_gini) {
					min_gini = samples.gini;
					samples.threshold = threshold;
					samples.texton_hist_position = samples_index[threshold_index];
					populateTextonHistForest(samples_index[threshold_index], l_training_data, samples.texton_hist);
					chosen_samples = samples;
					chosen_samples.gini = min_gini;
					chosen_samples.filter = filter;
					//chosen_samples.class_id_for_probability = cl;
				}
			}
			
		}


		//samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());

		//cout << "Left Sample size: " << samples.left_samples.size() << " right sample size: " << samples.right_samples.size() <<" Gini:"<<samples.gini<< endl;
	//}

	return chosen_samples;
}

SplittedSamples getSpittedSampleswithGini(int split_loc, vector<PatchLocation>samples_index, int **filter, vector<vector<probability_info>>& l_training_data) {
	//cout << "From splitting function" << endl;
	struct SplittedSamples chosen_samples;
	map<double, int>histogram_filter_response_left;
	map<double, int>histogram_filter_response_right;
	histogram_filter_response_left.clear();
	histogram_filter_response_right.clear();
	vector<double> filter_response;
	vector<double> unique_filter_response;
	map<double, int>is_it_unique;
	int image_size = (int)sqrt(l_training_data[0].size() - 1);//lenghth and width of image//considering image is square sized
	int count_left = 0;
	int count_right = 0;
	double min_gini = 99999;
	//cout << "filter position: " << filter_loc<<" "<<filter_loc/IMAGE_DATA_DIMENSION <<","<<filter_loc%IMAGE_DATA_DIMENSION<<endl;

	for (int cl = 0; cl < NUM_OF_CLASSES; cl++) {
		//int freq_resp_max = -1;
		//int freq_resp_max_val = -5;
		int bad_loc_count = 0;
		is_it_unique.clear();
		filter_response.clear(); filter_response.shrink_to_fit();
		unique_filter_response.clear(); unique_filter_response.shrink_to_fit();
		//global_response.clear();
		for (int i = 0; i < samples_index.size(); i++) {
			//int res = 0;
			int row = samples_index.at(i).pixel_position / image_size;
			int col = samples_index.at(i).pixel_position % image_size;
			int r = 0;// split_loc / PATCH_SIZE;
			int c = 0;// split_loc % PATCH_SIZE;
			double response = 0;
			r = 0;
			c = 0;
			for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
				for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
					//if (r + f_i > PATCH_SIZE || c + f_j > PATCH_SIZE) { continue; }
					if ((row + r + f_i)*image_size + (col + c + f_j) < l_training_data[0].size() && (row + r + f_i)*image_size + (col + c + f_j) >= 1) {
						response += l_training_data[samples_index.at(i).sample_index][(row + r + f_i)*image_size + (col + c + f_j)].class_postarior[cl] * filter[f_i][f_j];
					}
				}
			}
			//cout << response << split_loc <<endl;
			filter_response.push_back(response);
			if (is_it_unique[filter_response[i]] == 0) { unique_filter_response.push_back(filter_response[i]); }
			is_it_unique[filter_response[i]] = 1;
			//global_response[filter_response[i]]++;

		}
		//cout << "OK !" << endl;
		//cout << "resp: " << freq_resp_max_val << " freq: " << freq_resp_max << endl;
		//rand_seed = getTickCount();
		//rand_seed += 123456789;
		//cout << "Choosing threshold " << rand_seed << endl;
		//srand(rand_seed);
		int num_of_threshold = (int)sqrt((double)unique_filter_response.size());
		//num_of_threshold = unique_filter_response.size();
		for (int k = 0; k < num_of_threshold; k++) {
			int threshold_index = rand() % unique_filter_response.size();
			double threshold = unique_filter_response[threshold_index];
			//threshold = rand() % 255;
			//cout << "Num of unique responses: "<< unique_filter_response.size()<<" Threshold index:"<<threshold_index<<" Threshold: " << threshold << endl;
			struct SplittedSamples samples;
			for (int i = 0; i < NUM_OF_CLASSES; i++) {
				samples.left_histogram.push_back(0);
				samples.right_histogram.push_back(0);
			}
			count_left = 0;
			count_right = 0;
			histogram_filter_response_left.clear();
			histogram_filter_response_right.clear();
			for (int i = 0; i < samples_index.size(); i++) {

				//cout << "response:" << filter_response[i] << endl;
				
				if (filter_response[i] < threshold) {
					//if (filter_response != 0) {
					histogram_filter_response_left[filter_response[i]]++;
					count_left++;
					//}

					samples.left_samples.push_back(samples_index.at(i));
					samples.left_histogram.at(l_training_data[samples_index.at(i).sample_index][0].class_name)++;
					samples.split_loc = split_loc;
				}
				else {
					//if (filter_response != 0) {
					histogram_filter_response_right[filter_response[i]]++;
					count_right++;
					//}
					samples.right_samples.push_back(samples_index.at(i));
					samples.right_histogram.at(l_training_data[samples_index.at(i).sample_index][0].class_name)++;
					samples.split_loc = split_loc;

				}
			}

			//samples.gini = getGini(histogram_filter_response_left, histogram_filter_response_right, count_left, count_right);// passing histogrms of feature labels
			samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());// passing histogram of class labels
																																			   //cout << "left: " << count_left << " Right: " << count_right << " gini: "<<samples.gini<<endl;
			if (samples.gini < min_gini) {
				min_gini = samples.gini;
				samples.threshold = threshold;
				chosen_samples = samples;
				chosen_samples.gini = min_gini;
				chosen_samples.filter = filter;
				chosen_samples.class_id_for_probability = cl;
			}
		}


		//samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());

		//cout << "Left Sample size: " << samples.left_samples.size() << " right sample size: " << samples.right_samples.size() <<" Gini:"<<samples.gini<< endl;
	}
	
	return chosen_samples;
}

SplittedSamples getSpittedSampleswithGini(int split_loc, vector<PatchLocation>samples_index, int **filter, vector<vector<int>>& l_training_data) {
	//cout << "From splitting function" << endl;
	struct SplittedSamples chosen_samples;
	map<int, int>histogram_filter_response_left;
	map<int, int>histogram_filter_response_right;
	histogram_filter_response_left.clear();
	histogram_filter_response_right.clear();
	vector<int> filter_response;
	vector<int> unique_filter_response;
	map<int, int>is_it_unique;
	int image_size = (int)sqrt(l_training_data[0].size() - 1);//lenghth and width of image//considering image is square sized
	int count_left = 0;
	int count_right = 0;
	double min_gini = 99999;
	//cout << "filter position: " << filter_loc<<" "<<filter_loc/IMAGE_DATA_DIMENSION <<","<<filter_loc%IMAGE_DATA_DIMENSION<<endl;

	//int freq_resp_max = -1;
	//int freq_resp_max_val = -5;
	int bad_loc_count = 0;
	is_it_unique.clear();
	//global_response.clear();
	for (int i = 0; i < samples_index.size(); i++) {
		//int res = 0;
		int row = samples_index.at(i).pixel_position / image_size;
		int col = samples_index.at(i).pixel_position % image_size;
		int r = 0;// split_loc / PATCH_SIZE;
		int c = 0;// split_loc % PATCH_SIZE;
		int response = 0;
		r = 0;
		c = 0;
		for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
			for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
				//if (r + f_i > PATCH_SIZE || c + f_j > PATCH_SIZE) { continue; }
				if ((row + r + f_i)*image_size + (col + c + f_j) < l_training_data[0].size() && (row + r + f_i)*image_size + (col + c + f_j) >= 1) {
					response += l_training_data[samples_index.at(i).sample_index][(row + r + f_i)*image_size + (col + c + f_j)]
						* filter[f_i][f_j];
				}
			}
		}
		//cout << response << split_loc <<endl;
		filter_response.push_back(response);
		if (is_it_unique[filter_response[i]] == 0) { unique_filter_response.push_back(filter_response[i]); }
		is_it_unique[filter_response[i]] = 1;
		//global_response[filter_response[i]]++;

	}
	//cout << "OK !" << endl;
	//cout << "resp: " << freq_resp_max_val << " freq: " << freq_resp_max << endl;
	//rand_seed = getTickCount();
	//rand_seed += 123456789;
	//cout << "Choosing threshold " << rand_seed << endl;
	//srand(rand_seed);
	int num_of_threshold = (int)sqrt((double)unique_filter_response.size());
	//num_of_threshold = unique_filter_response.size();
	for (int k = 0; k < num_of_threshold; k++) {
		int threshold_index = rand() % unique_filter_response.size();
		int threshold = unique_filter_response[threshold_index];
		//threshold = rand() % 255;
		//cout << "Num of unique responses: "<< unique_filter_response.size()<<" Threshold index:"<<threshold_index<<" Threshold: " << threshold << endl;
		struct SplittedSamples samples;
		for (int i = 0; i < NUM_OF_CLASSES; i++) {
			samples.left_histogram.push_back(0);
			samples.right_histogram.push_back(0);
		}
		count_left = 0;
		count_right = 0;
		histogram_filter_response_left.clear();
		histogram_filter_response_right.clear();
		for (int i = 0; i < samples_index.size(); i++) {

			//cout << "response:" << filter_response[i] << endl;
			
			if (filter_response[i] < threshold) {
				//if (filter_response != 0) {
				histogram_filter_response_left[filter_response[i]]++;
				count_left++;
				//}

				samples.left_samples.push_back(samples_index.at(i));
				samples.left_histogram.at(l_training_data[samples_index.at(i).sample_index][0])++;
				samples.split_loc = split_loc;
			}
			else {
				//if (filter_response != 0) {
				histogram_filter_response_right[filter_response[i]]++;
				count_right++;
				//}
				samples.right_samples.push_back(samples_index.at(i));
				samples.right_histogram.at(l_training_data[samples_index.at(i).sample_index][0])++;
				samples.split_loc = split_loc;

			}
		}

		//samples.gini = getGini(histogram_filter_response_left, histogram_filter_response_right, count_left, count_right);//passing histograms of feature labels
		samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());//passing histograms of class labels
																																		   //cout << "left: " << count_left << " Right: " << count_right << " gini: "<<samples.gini<<endl;
		if (samples.gini < min_gini) {
			min_gini = samples.gini;
			samples.threshold = threshold;
			chosen_samples = samples;
			chosen_samples.gini = min_gini;
			chosen_samples.filter = filter;
		}
	}


	//samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());

	//cout << "Left Sample size: " << samples.left_samples.size() << " right sample size: " << samples.right_samples.size() <<" Gini:"<<samples.gini<< endl;
	return chosen_samples;
}

int ftype_zero_count = 0;
int ftype_one_count = 0;
int ftype_two_count = 0;
int ftype_three_count = 0;
Node* BuildTree(Node* node, int node_id, int depth, int num_of_feature_types, vector<PatchLocation>samples_index, vector<int>histogram, int impurity, vector<vector<int>>& l_training_data, vector<vector<probability_info>>& l_training_data_probability, vector<vector<int>>& l_training_data_path, vector<vector<path_info>>& l_training_data_path_visited_nodes) {
	/*
	This function builds a tree with training samples
	*/

	//cout << "From build tree function" << endl;
	vector<int>left_child_saple_index;
	vector<int>right_child_saple_index;
	tree_node_count++;
	if (depth > tree_height) { tree_height = depth; }
	if (node == NULL) {
		node = new Node();
	}
	//cout << "cc" << endl;
	//node->node_id = node_id;
	if (depth == MAX_DEPTH) {
		node->samples = samples_index;
		node->histogram = histogram;
		for (int k = 0; k < NUM_OF_CLASSES; k++) {
			node->classProbability[k] = (double)node->histogram[k] / node->samples.size();
		}
		node->imleaf = 1; tree_leaf_count++; node->leaf_height = depth;
		if (node->leaf_height < shortest_leaf_height) { shortest_leaf_height = node->leaf_height; }
		if (node->samples.size() > largest_leaf_with_data_samples) { largest_leaf_with_data_samples = node->samples.size(); }
		return node;
	}
	if (samples_index.size() < MIN_SAMPLE) {
		node->samples = samples_index;
		node->histogram = histogram;
		for (int k = 0; k < NUM_OF_CLASSES; k++) {
			node->classProbability[k] = (double)node->histogram[k] / node->samples.size();
		}
		node->imleaf = 1; tree_leaf_count++; node->leaf_height = depth;
		if (node->leaf_height < shortest_leaf_height) { shortest_leaf_height = node->leaf_height; }
		if (node->samples.size() > largest_leaf_with_data_samples) { largest_leaf_with_data_samples = node->samples.size(); }
		return node;
	}
	node->impurity = impurity;
	//cout << "dd" << endl;
	//node->index = index;
	//cout << node->index << endl;
	/*
	Check for spittiing feature interms of filter and image_index based on filter response
	Randomly choose five location on image, place each filter on each of the location, split the dataset
	based on the filter response. Calculate the gini index for each of the filter and index pairs for the entire training dataset.
	Choose the one with minimum gini index.
	*/
	double gini = 99999;
	double chosen_threshold = 0;
	struct SplittedSamples splitted_samples;
	struct SplittedSamples testsamples;
	//rand_seed += 123456789;
	//cout << "Building tree " << rand_seed << endl;
	//srand(rand_seed);
	int row, col;
	int splitting_position_on_image;
	//srand(getTickCount());
	//for (int j = 0; j < NUM_RAND_LOC_ON_IMAGE_PATCH; j++) {

		splitting_position_on_image = rand() % (PATCH_SIZE*PATCH_SIZE);//Not used now
		//double row = (int)row_distribution(generator);


		for (int fx = 0; fx < num_of_feature_types; fx++) { //num_of_feature_types;
			int f = fx;
			//int f = rand() % 2;
			//if (num_of_feature_types == 1) { f = 0; }
			for (int i = 0; i < NUM_OF_FILTERS; i++) {
				if (f == 0) {
					if (i == 0) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_one, l_training_data);
					}
					else if (i == 1) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_two, l_training_data);
					}
					else if (i == 2) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_three, l_training_data);
					}
					else if (i == 3) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_four, l_training_data);
					}
					else if (i == 4) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_five, l_training_data);
					}
					else {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_six, l_training_data);
					}
				}
				else if (f == 1) {
					if (i == 0) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_one, l_training_data_probability);
					}
					/* For features like probability estimates, a simple average filter should be used
					else if (i == 1) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_two, l_training_data_probability);
					}
					else if (i == 2) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_three, l_training_data_probability);
					}
					else if (i == 3) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_four, l_training_data_probability);
					}
					else if (i == 4) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_five, l_training_data_probability);
					}
					else {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_six, l_training_data_probability);
					}
					*/
				}
				else{
					if (i == 0) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_one, l_training_data_path_visited_nodes);
					}
					/* For features like path, a simple average filter should be used
					else if (i == 1) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_two, l_training_data_path);
					}
					else if (i == 2) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_three, l_training_data_path);
					}
					else if (i == 3) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_four, l_training_data_path);
					}
					else if (i == 4) {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_five, l_training_data_path);
					}
					else {
						testsamples = getSpittedSampleswithGini(splitting_position_on_image, samples_index, filter_six, l_training_data_path);
					}
					*/
				}
				



				//cout << "Left samples size: " << testsamples.left_samples.size() << "Right samples size: " << testsamples.right_samples.size() << " "<<"gini: "<<testsamples.gini<<endl;
				if (testsamples.gini < gini) {
					gini = testsamples.gini;
					splitted_samples = testsamples;
					chosen_threshold = testsamples.threshold;
					node->texton_hist_position = splitted_samples.texton_hist_position;
					node->texton_hist_forest = splitted_samples.texton_hist;
					node->feature_index = splitting_position_on_image;
					node->feature_filter = splitted_samples.filter;
					node->f_type = f;
					node->class_id_for_probability = testsamples.class_id_for_probability;
				}

			}
		}

	//}

	//	if (node->impurity < gini) { node->imleaf = 1; return node; }
	node->split_loc = splitted_samples.split_loc;
	node->threshold = chosen_threshold;
	node->samples = samples_index;
	node->histogram = histogram;
	for (int k = 0; k < NUM_OF_CLASSES; k++) {
		node->classProbability[k] = (double)node->histogram[k] / node->samples.size();
	}
	//cout << "chosen leftsamplesize: " << splitted_samples.left_samples.size() << " chosen rightsamplesize: " << splitted_samples.right_samples.size() << " Gini: " << gini << " feature type: "<< node->f_type<<endl;
	if (node->f_type == 0) { ftype_zero_count++; }//for analyzing
	else if (node->f_type == 1) { ftype_one_count++; }//for analyzing
	else if(node->f_type == 2){ ftype_two_count++; }//for analyzing
	else { ftype_three_count++; }

	if (splitted_samples.left_samples.size() == 0 || splitted_samples.right_samples.size() == 0) {
		node->imleaf = 1; tree_leaf_count++; node->leaf_height = depth;
		if (node->leaf_height < shortest_leaf_height) { shortest_leaf_height = node->leaf_height; }
		if (node->samples.size() > largest_leaf_with_data_samples) { largest_leaf_with_data_samples = node->samples.size(); }
		return node;
	}
	if (splitted_samples.left_samples.size() > 0) {
		node->Left = BuildTree(node->Left, 2*node_id+1, depth + 1, num_of_feature_types, splitted_samples.left_samples, splitted_samples.left_histogram, gini, l_training_data, l_training_data_probability, l_training_data_path, l_training_data_path_visited_nodes);
	}
	if (splitted_samples.right_samples.size() > 0) {
		node->Right = BuildTree(node->Right, 2*node_id+2, depth + 1, num_of_feature_types, splitted_samples.right_samples, splitted_samples.right_histogram, gini, l_training_data, l_training_data_probability, l_training_data_path, l_training_data_path_visited_nodes);
	}

	/*
	if (node->Left == NULL && node->Right == NULL) {
	node->imleaf = 1;
	}*/
	return node;
}

vector<map<int, int>>current_image_patch_texton_hist;
Node* getPredictionHistogram(Node* node, int sample_index, int location, vector<vector<int>>& l_test_data, vector<vector<probability_info>>& l_test_data_probability, vector<vector<int>>& l_test_data_path, vector<vector<path_info>>& l_test_data_path_visited_nodes, vector<vector<path_info>>& l_training_data_path_visited_nodes, int tree_index) {
	Node* nextRightNode = node->Right;
	Node* nextLeftNode = node->Left;
	Node* currNode = node;
	string t_path = "1";
	vector<bool> visited_nodes;
	visited_nodes.push_back(1);
	struct PatchLocation p_loc;
	p_loc.sample_index = sample_index;
	p_loc.pixel_position = location;
	//int t_path_num_of_left_shift = 0;
	//int t_path_num_of_right_shift = 0;
	
	/*
	cout << nextNode->index <<" "<<node->index<< endl;
	nextNode = nextNode->Right;
	cout << nextNode->index << " " << node->index << endl;
	nextNode = nextNode->Right;
	cout << nextNode->index << " " << node->index << endl;
	nextNode = nextNode->Right;
	cout << nextNode->index << " " << node->index << endl;
	if (nextNode->Right != NULL) {
	nextNode = nextNode->Right;
	cout << nextNode->index << " " << node->index << endl;
	}*/


	while (1) {
		//cout << currNode->index << endl;
		//if (currNode == NULL) { break; }
		//visited_nodes.push_back(currNode->node_id);
		if (currNode->imleaf) {
			currNode->leaf_size++;
			if (currNode->leaf_size >= largest_leaf_size[tree_index]) {
				largest_leaf_size[tree_index] = currNode->leaf_size;
				largest_leaf_height[tree_index] = currNode->leaf_height;
			}
			break;
		}
		int image_size = (int)sqrt(l_test_data[0].size() - 1);
		Node* nextRightNode = currNode->Right;
		Node* nextLeftNode = currNode->Left;
		int row = location / image_size;
		int col = location % image_size;
		//cout << row << " " << col << " f_type: "<<currNode->f_type<<endl;
		int r = currNode->split_loc / PATCH_SIZE;
		int c = currNode->split_loc % PATCH_SIZE;
		r = 0;
		c = 0;
		double response = 0;
		if (currNode->f_type == 0) {
			//if (l_test_data_path_shift_diff.size() > 0) { cout << "xxx" << currNode->f_type<< endl; }
			for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
				for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
					//if (r + f_i > FILTER_SIZE || c + f_j > FILTER_SIZE) { continue; }
					if ((row + r + f_i)*image_size + (col + c + f_j) < image_size*image_size + 1 && (row + r + f_i)*image_size + (col + c + f_j) >= 1) {
						response += l_test_data[sample_index][(row + r + f_i)*image_size + (col + c + f_j)]
							* node->feature_filter[f_i][f_j];
					}
				}
			}
		}
		else if (currNode->f_type == 1) {
			//if (l_test_data_path_shift_diff.size() > 0) { cout << "xxx" << currNode->f_type<< endl; }
			int cl = currNode->class_id_for_probability;
			for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
				for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
					//if (r + f_i > FILTER_SIZE || c + f_j > FILTER_SIZE) { continue; }
					if ((row + r + f_i)*image_size + (col + c + f_j) < image_size*image_size + 1 && (row + r + f_i)*image_size + (col + c + f_j) >= 1) {
						response += l_test_data_probability[sample_index][(row + r + f_i)*image_size + (col + c + f_j)].class_postarior[cl]
							* node->feature_filter[f_i][f_j];
					}
				}
			}
		}
		else if (currNode->f_type == 2) {
			//if (l_test_data_path_shift_diff.size() > 0) { cout << "xxx" << currNode->f_type<< endl; }
			//cout << location<<": texton_hist_start" << endl;
			//response = getTextonHistogramComparison_test(node->texton_hist_position, p_loc, l_test_data_path_visited_nodes, l_training_data_path_visited_nodes);
			//response = getTextonHistogramComparisonWithLeafID_test(node->texton_hist_position, p_loc, l_test_data_path_visited_nodes, l_training_data_path_visited_nodes);
			response = getTextonHistogramComparisonWithLeafID(node->texton_hist_forest, current_image_patch_texton_hist);
			//cout << "texton_hist_end" << endl;
			//cout << "texton hist response: " << response << endl;
			//getNewPathFeature_test(node->samples, p_loc, l_test_data_path_visited_nodes, l_training_data_path_visited_nodes);
		}
		else {
			//if (l_test_data_path_shift_diff.size() > 0) { cout << "xxx" << currNode->f_type<< endl; }
			
			for (int f_i = 0; f_i < FILTER_SIZE; f_i++) {
				for (int f_j = 0; f_j < FILTER_SIZE; f_j++) {
					//if (r + f_i > FILTER_SIZE || c + f_j > FILTER_SIZE) { continue; }
					//cout << "Here !!" << endl;
					if ((row + r + f_i)*image_size + (col + c + f_j) < image_size*image_size + 1 && (row + r + f_i)*image_size + (col + c + f_j) >= 1) {
						//cout << "Here !!!" << sample_index<<" "<< (row + r + f_i)*image_size + (col + c + f_j)<<endl;
						response += l_test_data_path[sample_index][(row + r + f_i)*image_size + (col + c + f_j)]
							* node->feature_filter[f_i][f_j];
					}
				}
			}
		}

		//cout << "asdas"<<endl;


		//filter_response.push_back(training_data[i][(row + r)*IMAGE_DATA_DIMENSION + (col + c)]);
		if (response < currNode->threshold) {
			//<threshold
			currNode = nextLeftNode;
			t_path.append("0");
			visited_nodes.push_back(0);
			//t_path_num_of_left_shift--;
		}
		else {

			currNode = nextRightNode;
			t_path.append("1");
			visited_nodes.push_back(1);
			//t_path_num_of_right_shift++;
		}


	}
	//for (int i = 0; i < NUM_OF_CLASSES; i++) { cout << currNode->classProbability[i] << " "; }cout << endl;
	currNode->path = t_path;
	currNode->visited_nodes = visited_nodes;
	//copy(visited_nodes.begin(), visited_nodes.end(), currNode->visited_nodes.begin());
	//currNode->path_num_of_left_shift = t_path_num_of_left_shift;
	//currNode->path_num_of_right_shift = t_path_num_of_right_shift;
	return currNode;// ->classProbability;
}

void populateFilters() {

	filter_one = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_two = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_three = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_four = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_five = (int**)malloc(FILTER_SIZE * sizeof(int*));


	for (int i = 0; i < FILTER_SIZE; i++) {
		filter_one[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_two[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_three[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_four[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_five[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));


	}

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			filter_one[i][j] = 1;
			cout << filter_one[i][j];
		}
		cout << endl;
	}

	cout << endl << endl;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			if (j < FILTER_SIZE / 3 || j >(FILTER_SIZE * 2 / 3) - 1) {
				filter_two[i][j] = 1;
			}
			else {
				filter_two[i][j] = 2;
			}
			cout << filter_two[i][j];
		}
		cout << endl;
	}

	cout << endl << endl;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			if (j > FILTER_SIZE / 2 - 1) {
				filter_three[i][j] = 1;
			}
			else {
				filter_three[i][j] = 2;
			}
			cout << filter_three[i][j];
		}
		cout << endl;
	}

	cout << endl << endl;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			if (i < FILTER_SIZE / 2 && j < FILTER_SIZE / 2) {
				filter_four[i][j] = 1;
			}
			else if (i >(FILTER_SIZE / 2 - 1) && j >(FILTER_SIZE / 2 - 1)) {
				filter_four[i][j] = 1;
			}
			else {
				filter_four[i][j] = 2;
			}
			cout << filter_four[i][j];
		}
		cout << endl;
	}

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			filter_five[i][j] = 1;
			cout << filter_five[i][j];
		}
		cout << endl;
	}
	cout << endl;
}
void populate_new_filters() {
	filter_one = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_two = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_three = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_four = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_five = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_six = (int**)malloc(FILTER_SIZE * sizeof(int*));


	for (int i = 0; i < FILTER_SIZE; i++) {
		filter_one[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_two[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_three[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_four[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_five[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_six[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));


	}

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			filter_one[i][j] = 1;
			cout << filter_one[i][j];
		}
		cout << endl;

	}
	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if ((i*FILTER_SIZE + j) % 2 == 0) {
				filter_two[i][j] = 2;

			}
			else {
				filter_two[i][j] = 1;
			}
			cout << filter_two[i][j];
		}
		cout << endl;
	}
	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if (i % 2 == 1 || j % 2 == 1) {
				filter_three[i][j] = 2;

			}
			else {
				filter_three[i][j] = 1;
			}
			cout << filter_three[i][j];
		}
		cout << endl;

	}
	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if (i % 2 == 1) {
				filter_four[i][j] = 2;

			}
			else {
				filter_four[i][j] = 1;
			}
			cout << filter_four[i][j];
		}
		cout << endl;

	}
	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if (j % 2 == 1) {
				filter_five[i][j] = 2;

			}
			else {
				filter_five[i][j] = 1;
			}
			cout << filter_five[i][j];
		}
		cout << endl;
	}

	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if (i + j == FILTER_SIZE - 1) {
				filter_six[i][j] = 2;
			}
			else {
				filter_six[i][j] = 1;
			}
			cout << filter_six[i][j];
		}
		cout << endl;
	}

	cout << endl;

	/*
	for (int i = 0; i < FILTER_SIZE; i++) {
	for (int j = 0; j < FILTER_SIZE; j++) {

	if (i == 1) {
	filter_one[i][j] = 2;
	}
	else {
	filter_one[i][j] = 1;
	}
	cout << filter_one[i][j];
	}
	cout << endl;
	}

	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
	for (int j = 0; j < FILTER_SIZE; j++) {

	if (j == 1) {
	filter_two[i][j] = 2;
	}
	else {
	filter_two[i][j] = 1;
	}
	cout << filter_two[i][j];
	}
	cout << endl;
	}
	*/
	cout << endl;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if (i + j == FILTER_SIZE - 1) {
				filter_three[i][j] = 2;
			}
			else {
				filter_three[i][j] = 1;
			}
			cout << filter_three[i][j];
		}
		cout << endl;
	}

	cout << endl;

}
/*
int ReverseInt(int i)
{
unsigned char ch1, ch2, ch3, ch4;
ch1 = i & 255;
ch2 = (i >> 8) & 255;
ch3 = (i >> 16) & 255;
ch4 = (i >> 24) & 255;
return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr)
{
arr.resize(NumberOfImages, vector<double>(DataOfAnImage));
ifstream file("D:\\data0", ios::binary);
if (file.is_open())
{
int magic_number = 0;
int number_of_images = 0;
int n_rows = 0;
int n_cols = 0;
file.read((char*)&magic_number, sizeof(magic_number));
magic_number = ReverseInt(magic_number);
file.read((char*)&number_of_images, sizeof(number_of_images));
number_of_images = ReverseInt(number_of_images);
file.read((char*)&n_rows, sizeof(n_rows));
n_rows = ReverseInt(n_rows);
file.read((char*)&n_cols, sizeof(n_cols));
n_cols = ReverseInt(n_cols);
for (int i = 0; i<number_of_images; ++i)
{
for (int r = 0; r<n_rows; ++r)
{
for (int c = 0; c<n_cols; ++c)
{
unsigned char temp = 0;
file.read((char*)&temp, sizeof(temp));
arr[i][(n_rows*r) + c] = (double)temp;
}
}
}
}
}
*/

void readMNSITfromCSV(string filepath, vector<vector<int>> &data_samples, int num_of_samples) {

	/*
	int random_sample_index[NUM_TRAINING_SAMPLES];
	for (int i = 0; i < NUM_TRAINING_SAMPLES; i++) {
	random_sample_index[i] = rand() % 600000;//60000 is the total data sample in origina training data.
	}*/
	ifstream inputfle(filepath);
	string current_line;
	int num_line_read = 1;
	while (getline(inputfle, current_line)) {
		// Now inside each line we need to seperate the cols
		//cout << "test";
		if (num_line_read > num_of_samples) break;
		vector<int> values;
		stringstream temp(current_line);
		string single_value;
		while (getline(temp, single_value, ',')) {
			// convert the string element to a integer value
			values.push_back(atoi(single_value.c_str()));
			cout << atoi(single_value.c_str()) << " ";
		}
		cout << endl;
		// add the row to the complete data vector
		data_samples.push_back(values);
		num_line_read++;
	}
	//cout << num_line_read << endl;
}

/*void maptest(map<int, int> &mymap) {
for (map<int, int>::iterator it = mymap.begin(); it != mymap.end(); ++it)
std::cout << it->first << " => " << it->second << '\n';
}*/

void printParameters() {
	cout << "MAX_DEPTH " << MAX_DEPTH << endl;
	cout << "FILTER_SIZE " << FILTER_SIZE << endl;
	cout << "PATCH_SIZE " << PATCH_SIZE << endl;
	cout << "NUM_TRAINING_SAMPLES " << NUM_TRAINING_SAMPLES << endl;
	cout << "NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE " << NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE << endl;
	cout << "IMAGE_DATA_DIMENSION " << IMAGE_DATA_DIMENSION << endl;
	cout << "NUM_RAND_LOC_ON_IMAGE_PATCH " << NUM_RAND_LOC_ON_IMAGE_PATCH << endl;
	cout << "NUM_OF_FILTERS " << NUM_OF_FILTERS << endl;
	cout << "NUM_OF_CLASSES " << NUM_OF_CLASSES << endl;
	cout << "MIN_SAMPLE " << MIN_SAMPLE << endl;
	cout << "NUM_OF_TREE " << NUM_OF_TREE << endl;
	cout << "NUM_OF_TEST_SAMPLES " << NUM_OF_TEST_SAMPLES << endl;
	cout << "NUM_OF_THRESHOLD " << NUM_OF_THRESHOLD << endl;
	cout << "ROUGH_NORMALIZATION " << ROUGH_NORMALIZATION << endl;


}
int classname_count[3];
void loadTrainingData(vector<vector<int>> &data_samples) {
	freopen("training_data_scene.txt", "r", stdin);
	int classname;
	vector<int> values;
	int intensity;
	int index = 0;
	int line_counter = 0;
	classname_count[0] = 0; classname_count[1] = 0; classname_count[2] = 0;
	while (scanf("%d", &classname) == 1) {
		//cout << classname<<endl;
		line_counter++;
		classname_count[classname]++;
		values.clear();
		values.push_back(classname);

		//cout << sortedSamples[index].classname << " " << sortedSamples[index].sample_index << endl;

		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			scanf("%d", &intensity);
			values.push_back(intensity);

		}
		if (classname_count[classname] > NUM_TRAINING_SAMPLES / NUM_OF_CLASSES) { continue; }
		training_data_dist[classname]++;
		sortedSamples[index].classname = classname;
		sortedSamples[index].sample_index = index;
		index++;
		data_samples.push_back(values);
		//if (data_samples.size() == NUM_TRAINING_SAMPLES) { break; }
		//if(i==10)break;
	}
	sort(sortedSamples, sortedSamples + NUM_TRAINING_SAMPLES, comp);

	for (int k = 0; k < NUM_OF_CLASSES; k++) {
		cout << k << " : " << training_data_dist[k] << endl;
	}
	/*
	for (int k = 0; k < NUM_TRAINING_SAMPLES; k++) {
	cout << sortedSamples[k].classname << " " << sortedSamples[k].sample_index<<endl;
	}*/
}


void loadTestSamples(vector<vector<int>> &data_samples, int num_of_samples) {
	freopen("test_data_set.txt", "r", stdin);
	int classname;
	vector<int> values;
	int intensity;
	int sample_count = 0;
	cout << "Came here" << endl;
	while (scanf("%d", &classname) == 1) {
		//cout << classname<<endl;
		values.clear();
		values.push_back(classname);

		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			scanf("%d", &intensity);
			values.push_back(intensity);

		}
		data_samples.push_back(values);
		sample_count++;
		if (sample_count == num_of_samples) { break; }
		//if(i==10)break;
	}
	cout << data_samples.size() << endl;
}

//this function assumes all binary numbers begin with a 1
int getInteger(string binary_num) {
	int len = binary_num.size();
	int sum = 0;
	int j = 0;
	for (int i = len - 1; i >= 0; i--) {
		//cout << (binary_num[i] - 48);
		sum += (binary_num[j] - 48)*(int)pow(2, i);
		j++;
	}
	//cout << endl;
	return sum;
}

int LevenshteinDistance(string s, string t) {
	int m = s.size();
	int n = t.size();
	int substitutionCost = 0;

	//int d[1000][1000];

	int** d = new int*[m + 1];
	for (int i = 0; i < m + 1; i++) {
		d[i] = new int[n + 1];
		for (int j = 0; j < n + 1; j++) {
			d[i][j] = 0;
		}
	}

	//return 0;



	for (int i = 1; i < m + 1; i++) {
		d[i][0] = i;
	}

	for (int j = 1; j < n + 1; j++) {
		d[0][j] = j;
	}

	//return 0;
	for (int j = 1; j < n + 1; j++) {
		//v1[0] = i+1;

		for (int i = 1; i < m + 1; i++) {
			if (s[i - 1] == t[j - 1]) {
				substitutionCost = 0;
			}
			else {
				substitutionCost = 1;
			}

			if (d[i - 1][j] + 1 < d[i][j - 1] + 1 && d[i - 1][j] + 1 < d[i - 1][j - 1] + substitutionCost) {

				d[i][j] = d[i - 1][j] + 1;
			}
			else if (d[i][j - 1] + 1 < d[i - 1][j] + 1 && d[i][j - 1] + 1 < d[i - 1][j - 1] + substitutionCost) {
				d[i][j] = d[i][j - 1] + 1;
			}
			else {
				d[i][j] = d[i - 1][j - 1] + substitutionCost;
			}

			// cout<<d[j][i]<<" ";

		}
		//cout<<endl;

		//temp = v0;
		//v0 = v1;
		//v1 = temp;
		//swap(v0,v1);

	}

	int res = d[m][n];
	for (int i = 0; i < m + 1; i++) {
		delete[] d[i];
	}
	delete[] d;
	return res;
}
int main() {
	freopen("output.txt", "w", stdout);
	/*
	vector<path_info> each_data_sample_path_visited_nodes;
	//probability_info p_info;
	path_info pa_info;
	vector<int>visited_nodes;
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 128; j++) {
			each_data_sample_path_visited_nodes.clear();
			for (int t = 0; t < 40; t++) {
				//paths.clear();
				visited_nodes.clear();
				for (int n = 0; n < 15; n++) {
					visited_nodes.push_back(n);
				}
				pa_info.visited_nodes.push_back(visited_nodes);
			}
			each_data_sample_path_visited_nodes.push_back(pa_info);
		}
		scaled_training_data_path_visited_nodes.push_back(each_data_sample_path_visited_nodes);
	}
	//try {
		training_data_path_visited_nodes = scaled_training_data_path_visited_nodes;
	//}
	//catch (Exception ex) {
		//cout << "exception" << endl;
	//}
	
	cout << "Exiting without exception" << endl;
	return 0;
	*/
	//cout << getInteger("1010") << " " << getInteger("1111") << " " << getInteger("11001011001") << endl;
	//return 0;
	/*
	double*** hist;
	hist = (double***)malloc(IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION * sizeof(double*));
	for (int i = 0; i < 10; i++) {
	hist[i] = (double**)malloc(NUM_OF_TREE * sizeof(double));
	for (int t = 0; t < NUM_OF_TREE; t++) {
	hist[i][t] = (double*)malloc(NUM_OF_CLASSES * sizeof(double));
	//free(hist[0][t]);
	}
	}
	//free(hist[0][0]);

	//return 0;

	for (int i = 0; i < 10; i++) {

	for (int t = 0; t < NUM_OF_TREE; t++) {

	free(hist[i][t]);
	}
	free(hist[i]);
	}

	free(hist);
	cout << "asdad" << endl;
	return 0;
	*/

	printParameters();
	long long start = clock();
	loadTrainingData(training_data);
	//readMNSITfromCSV("D:\\mnist_test.csv", training_data, 10000);//loads first 10000 samples for training
	
	//return 0;
	populate_new_filters();
	//populateFilters();//generating the filters which will be used later
	cout << training_data.size() << endl;
	vector<vector<PatchLocation>>t_samples_index;
	vector<vector<int>>h_histogram;
	vector<PatchLocation>samples_index;
	vector<int>histogram;
	//Node *root[MAX_FOREST_DEPTH][NUM_OF_TREE];
	Node *root[NUM_OF_TREE];

	rand_seed = 296709226116;// getTickCount();
	cout << "Main Func " << rand_seed << endl;
	srand(rand_seed);

	loadTestSamples(test_data, 150);
	cout << test_data.size() << endl;


	for (int forest_depth = 0; forest_depth < MAX_FOREST_DEPTH; forest_depth++) {
		cout << "WWW" << endl;
		int image_size = (int)sqrt(training_data[0].size() - 1);
		cout << "WWWWWW" << endl;
		h_histogram.clear(); h_histogram.shrink_to_fit();
		cout << "WWWWWWHHH" << endl;
		//initializing the histogram of 3 classes of the scenes
		for (int t = 0; t < NUM_OF_TREE; t++) {
			histogram.clear(); histogram.shrink_to_fit();
			for (int i = 0; i < NUM_OF_CLASSES; i++) {
				histogram.push_back(0);
			}
			h_histogram.push_back(histogram);
		}
		cout << "WWWWWWHHH!!!" << endl;
		t_samples_index.clear(); t_samples_index.shrink_to_fit();
		cout << "WWWWWWHHH!!!+++" << endl;
		samples_index.clear(); samples_index.shrink_to_fit();
		cout << "WWWWWWHHH!!!---" << endl;
		cout << "depth " << forest_depth << endl;
		//////////////////////////////////////////////////////////

		//////////////////////////////Random selection of samples and indexes////////////////////////////////

		for (int t = 0; t < NUM_OF_TREE; t++) {
			//rand_seed += 123456789;
			//cout << "Selecting sample for tree " << rand_seed << endl;
			//srand(rand_seed);
			int total_uniform_samples = 0;
			int rand_range_from = 0;
			int rand_range_to = 0;
			for (int classname = 0; classname < NUM_OF_CLASSES; classname++) {
				rand_range_to = training_data_dist[classname];
				//cout << "range:" << rand_range_from << "to " << rand_range_to << endl;
				struct PatchLocation patchlocation;
				//*((int)pow(2, forest_depth))
				for (int numsamples = 0; numsamples < NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE / (NUM_OF_CLASSES*((int)pow(2, forest_depth))); numsamples++) {
					int i = rand_range_from + rand() % (rand_range_to);
					patchlocation.sample_index = sortedSamples[i].sample_index;
					patchlocation.pixel_position = rand() % (image_size*image_size);
					samples_index.push_back(patchlocation);
					if (sortedSamples[i].classname == classname) { h_histogram[t][classname]++; }
				}
				rand_range_from += rand_range_to;

			}

			t_samples_index.push_back(samples_index);
			samples_index.clear(); samples_index.shrink_to_fit();
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////Building the nth forest/////////////////////////////////////////
		ftype_zero_count = 0;
		ftype_one_count = 0;
		ftype_two_count = 0;
		ftype_three_count = 0;
		int num_of_feature_types;
		for (int t = 0; t < NUM_OF_TREE; t++) {
			//cout << "xx" << endl;
			/*
			for (int i = 0; i < 10; i++) {
			cout << i << " " << h_histogram[t][i] << endl;
			}*/

			root[t] = new Node();
			//cout << "yy" << endl;
			//int patch_position_on_image;// = 1 + rand() % 784;
			//filter_position_on_image = 1 + rand() % 784;
			//int row = (int)row_distribution(generator);
			//int col = (int)row_distribution(generator);

			//if (row < 0) { row = 0; }
			//else if (row > IMAGE_DATA_DIMENSION - 1) { row = IMAGE_DATA_DIMENSION; }
			//if (col < 0) { col = 0; }
			//else if (col > IMAGE_DATA_DIMENSION ) { col = IMAGE_DATA_DIMENSION; }
			//patch_position_on_image = row*IMAGE_DATA_DIMENSION + col;
			//patch_position_on_image = 1 + rand() % (IMAGE_DATA_DIMENSION * IMAGE_DATA_DIMENSION);

			num_of_feature_types = 1;
			if (forest_depth > 0) { num_of_feature_types = NUM_FEATURES_TO_BE_USED; }//num_of_feature_types = 4;
			tree_height = -1; tree_leaf_count = 0; tree_node_count = 0; shortest_leaf_height = 999; largest_leaf_with_data_samples = 0;
			root[t] = BuildTree(root[t], 0, 0, num_of_feature_types, t_samples_index[t], h_histogram[t], 1, training_data, training_data_probability, training_data_path, training_data_path_visited_nodes);
			cout << "Tree " << t + 1 << " built. " << " height :" << tree_height << " shortest leaf height" << shortest_leaf_height << " total nodes: " << tree_node_count << " total leaf: " << tree_leaf_count << " largest leaf with data:"<< largest_leaf_with_data_samples << endl;
		}

		cout << "Num of feature types: " << num_of_feature_types << endl;
		cout << "intensity selected: " << ftype_zero_count << " times" << endl;
		cout << "probability estimate selected: " << ftype_one_count << " times" << endl;
		cout << "path info selected: " << ftype_two_count << " times" << endl;
		//cout << "path value shift diff selected: " << ftype_three_count << " times" << endl;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




		//////////////////////////////////////////////////////////////////////////////////////////////////////

		////////////////////////////////////Scaling the training data (Pooling)////////////////////////////////////////////

		map<double, int> prob_estimate_count;
		struct PatchLocation mpatchlocation;
		cout << "Training data layer: " << forest_depth << endl;
		image_size = (int)sqrt(training_data[0].size() - 1);
		double*** hist;
		hist = (double***)malloc(training_data.size() * sizeof(double**));
		double correct = 0;
		double E_rms_P = 0;
		double E_rms_P_ = 0;
		double E_rms_PC = 0;
		double E_rms_I = 0;
		double E_rms_I_ = 0;
		double E_rms_IC = 0;
		t_samples_index.clear();
		t_samples_index.shrink_to_fit();
		cout << "num of training data: " << training_data.size() << "image dimension: " << image_size << endl;
		//training_data_second_layer = training_data;
		for (int tt = 0; tt < NUM_OF_TREE; tt++) {
			largest_leaf_height[tt] = 0;
			largest_leaf_size[tt] = 0;
		}

		for (int i = 0; i < training_data.size(); i++) {
			for (int v = 0; v < NUM_OF_CLASSES; v++) {
				arr_occur[v] = 0;
			}
			cout << "Processing training sample: " << i << endl; //break;//for the time beaing
			mpatchlocation.sample_index = i;
			vector<int> each_data_sample;
			vector<probability_info> each_data_sample_probability;
			vector<int> each_data_sample_path;
			vector<path_info> each_data_sample_path_visited_nodes;

			each_data_sample.push_back(training_data[i][0]);//pushing name of the class
			probability_info p_info;
			path_info pa_info;
			p_info.class_name = training_data[i][0];
			pa_info.class_name = training_data[i][0];
			each_data_sample_probability.push_back(p_info);//pushing name of the class
			each_data_sample_path.push_back(training_data[i][0]);//pushing name of the class
			each_data_sample_path_visited_nodes.push_back(pa_info);
			//Mat vect = Mat::zeros(image_size / 2, image_size / 2, CV_8UC1);
			//int counter = 0;
			//cout << each_data_sample[0] << " ";

			for (int row = 0; row < image_size; row += 2) {
				for (int col = 0; col < image_size; col += 2) {
					
					double max_prob = -1;
					double max_class_posterior[NUM_OF_CLASSES];
					int selected_index = 0;
					int selected_path_value = 0;
					vector <vector<bool>> selected_path_value_visited_nodes;
					vector<int> selected_path_value_visited_leaf_ids;
					for (int nb_i = 0; nb_i < 2; nb_i++) {
						for (int nb_j = 0; nb_j < 2; nb_j++) {
							if (row + nb_i >= image_size || (col + nb_i) >= image_size) { continue; }
							int d = (row + nb_i)*image_size + (col + nb_j);
							mpatchlocation.pixel_position = d;
							if (forest_depth > 0) {
								//cout << "HHH" << endl;
								//cout << "Pixel number: " << d << " " << endl;
							}
							hist[i] = (double**)malloc(NUM_OF_TREE * sizeof(double*));
							int path_value = 0;
							vector <vector<bool>> path_value_visited_nodes;
							vector<int> path_value_visited_leaf_ids;
							current_image_patch_texton_hist.clear();
							current_image_patch_texton_hist.shrink_to_fit();
							if (forest_depth > 0) {
								//cout << "here" << endl;
								populateTextonHistForest(mpatchlocation, training_data_path_visited_nodes, current_image_patch_texton_hist);
								//cout << "here!" << endl;
							}
							
							for (int t = 0; t < NUM_OF_TREE; t++) {

								//hist[i][t] = (double*)malloc(NUM_OF_CLASSES * sizeof(double));
								//if (forest_depth > 0) { cout << "HHH!" << endl; }
								Node *nd = getPredictionHistogram(root[t], i, d, training_data, training_data_probability, training_data_path, training_data_path_visited_nodes, training_data_path_visited_nodes, t);
								hist[i][t] = nd->classProbability;
								//path_value += getInteger(nd->path);
								path_value += nd->path.size();
								//path_value_visited_nodes += nd->path_num_of_right_shift;// +nd->path_num_of_left_shift;
								
								path_value_visited_nodes.push_back(nd->visited_nodes);
								int visited_leaf_id = 0;
								for (int b = 0; b < nd->visited_nodes.size(); b++) {
									visited_leaf_id += nd->visited_nodes[b] * two_pow[nd->visited_nodes.size() - b-1];
								}
								path_value_visited_leaf_ids.push_back(visited_leaf_id);
								//if (forest_depth > 0) { cout << "HHH!!" << endl; }
								//cout << nd->path << endl;
								//free(hist[i][t]);
							}


							double max = -999;
							int res_class;
							double res_prob;
							

							for (int j = 0; j < NUM_OF_CLASSES; j++) {
								double sum = 0;
								for (int t = 0; t < NUM_OF_TREE; t++) {
									sum += hist[i][t][j];
									//cout << hist[i][t][j] << " ";
								}
								//cout << "sum" << sum <<" ";
								if (sum > max) {
									max = sum;
									res_class = j;
									res_prob = max / NUM_OF_TREE;
									 
								}
								class_posterior[j] = sum / NUM_OF_TREE;
							}
							//cout << "resclass:" << res_class<<" res_prob: "<<res_prob<<" ";
							arr_occur[res_class] += res_prob;// *res_prob;
							if (res_prob > max_prob) {
								max_prob = res_prob;
								copy(class_posterior, class_posterior + NUM_OF_CLASSES, max_class_posterior);
								selected_index = d;
								selected_path_value = path_value;// / NUM_OF_TREE;
								selected_path_value_visited_nodes = path_value_visited_nodes;
								selected_path_value_visited_leaf_ids = path_value_visited_leaf_ids;
							}

							//////////////////////////Freeing allocated memory. Important !!!//////////////////////////////
							//cout << "Trying to free some memory!!!!" << endl;
							//free(hist[0]);
							//cout << "HHHH" << endl;
							//getchar();

							free(hist[i]);

							//////////////////////////////////////////////////////////////////////////////


						}


					}
					//if (forest_depth > 0) { cout << "HHH!!!" << endl; }
					//cout << "max_prob: " << max_prob << endl;
					each_data_sample.push_back(training_data[i][selected_index]);
					copy(max_class_posterior, max_class_posterior + NUM_OF_CLASSES, p_info.class_postarior);
					//if (forest_depth > 0) { cout << "HHH!!!!" << endl; }
					p_info.max_prob = max_prob;
					pa_info.visited_nodes = selected_path_value_visited_nodes;
					pa_info.visited_leaf_ids = selected_path_value_visited_leaf_ids;
					each_data_sample_probability.push_back(p_info);
					each_data_sample_path.push_back(selected_path_value);
					each_data_sample_path_visited_nodes.push_back(pa_info);
					prob_estimate_count[max_prob]++;//for anaysis
					//vect.at<uchar>(row / 2, col / 2) = training_data[i][selected_index];
					//cout << (int)vect.at<uchar>(r, c)<<" ";
					//counter++;
					//cout << training_data[i][selected_index] << " ";
				}

			}

			scaled_training_data.push_back(each_data_sample);
			scaled_training_data_probability.push_back(each_data_sample_probability);
			scaled_training_data_path.push_back(each_data_sample_path);
			scaled_training_data_path_visited_nodes.push_back(each_data_sample_path_visited_nodes);
			//return 0;
			//namedWindow("Display window"+i, WINDOW_AUTOSIZE);// Create a window for display.
			//moveWindow("Display window" + i, 50, 100);
			//imshow("Display window" + i, vect);
			//waitKey(0);
			//getchar();
			//return 0;
			//cout << endl;
			//if (i == 1) { waitKey(0); return 0;}
		}


		for (int tt = 0; tt < NUM_OF_TREE; tt++) {
			cout << "tree" << tt + 1 << ": largest_leaf_size: " << largest_leaf_size[tt] << endl;
			cout << "tree" << tt + 1 << ": largest_leaf_height: " << largest_leaf_height[tt] << endl;
		}

		//////////////////////////Freeing allocated memory. Important !!!//////////////////////////////

		free(hist);

		/////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////Scaling Test Data (Pooling)////////////////////////////////////////////////

		hist = (double***)malloc(NUM_OF_TEST_SAMPLES * sizeof(double**));
		correct = 0;

		//readMNSITfromCSV("D:\\mnist_test.csv", test_data, NUM_OF_TEST_SAMPLES);

		int random_test_sample;

		for (int tt = 0; tt < NUM_OF_TREE; tt++) {
			largest_leaf_height[tt] = 0;
			largest_leaf_size[tt] = 0;
		}
		for (int i = 0; i < NUM_OF_TEST_SAMPLES; i++) {

			random_test_sample = i;

			
			
			if (forest_depth < 1) {
				if (i % 3 == 0) {
					random_test_sample = 100 + i;
					//cout << random_test_sample << endl;
				}
				else if (i % 2 == 0) {
					random_test_sample = 50 + i;
				}
				else {
					random_test_sample = 0 + i;
				}
			}
			//random_test_sample = 1;//for the time being
			//random_test_sample = rand() % 150;
			//random_test_sample = i;
			cout << "testing on sample number: " << random_test_sample << endl;
			for (int v = 0; v < NUM_OF_CLASSES; v++) {
				arr_occur[v] = 0;
			}

			vector<int> each_data_sample;
			vector<probability_info> each_data_sample_probability;
			vector<int> each_data_sample_path;
			vector<path_info> each_data_sample_path_visited_nodes;
			each_data_sample.push_back(test_data[random_test_sample][0]);
			probability_info p_info;
			path_info pa_info;
			mpatchlocation.sample_index = random_test_sample;
			p_info.class_name = test_data[random_test_sample][0];
			pa_info.class_name = test_data[random_test_sample][0];
			each_data_sample_probability.push_back(p_info);
			each_data_sample_path.push_back(test_data[random_test_sample][0]);
			each_data_sample_path_visited_nodes.push_back(pa_info);
			image_size = (int)sqrt(test_data[0].size() - 1);
			//if (forest_depth > 0) { cout << "Here !" << image_size << endl; }
			for (int row = 0; row < image_size; row += 2) {
				for (int col = 0; col < image_size; col += 2) {
					double max_prob = -1;
					double max_class_posterior[NUM_OF_CLASSES];
					int selected_index = 0;
					int selected_path_value = 0;
					vector<vector<bool>> selected_path_value_visited_nodes;// = 0;
					vector<int> selected_path_value_visited_leaf_ids;
					for (int nb_i = 0; nb_i < 2; nb_i++) {
						for (int nb_j = 0; nb_j < 2; nb_j++) {
							if (row + nb_i >= image_size || (col + nb_i) >= image_size) { continue; }
							int d = (row + nb_i)*image_size + (col + nb_j);
							mpatchlocation.pixel_position = d;
							hist[i] = (double**)malloc(NUM_OF_TREE * sizeof(double));
							int path_value = 0;
							vector<vector<bool>> path_value_visited_nodes;// = 0;
							vector<int> path_value_visited_leaf_ids;

							current_image_patch_texton_hist.clear();
							current_image_patch_texton_hist.shrink_to_fit();
							if (forest_depth > 0) {
								populateTextonHistForest(mpatchlocation, test_data_path_visited_nodes, current_image_patch_texton_hist);
							}
							
							for (int t = 0; t < NUM_OF_TREE; t++) {
								//if (forest_depth > 0) {cout << "Here !!" << endl;}
								//hist[i][t] = (double*)malloc(NUM_OF_CLASSES * sizeof(double));
								Node* nd = getPredictionHistogram(root[t], random_test_sample, d, test_data, test_data_probability, test_data_path, test_data_path_visited_nodes, training_data_path_visited_nodes, t);
								//if (forest_depth > 0) { cout << "Here !!!" << endl; }
								hist[i][t] = nd->classProbability;
								
								//path_value += getInteger(nd->path);
								path_value += nd->path.size();
								//path_value_visited_nodes += nd->path_num_of_right_shift;// +nd->path_num_of_left_shift;
								path_value_visited_nodes.push_back(nd->visited_nodes);
								int visited_leaf_id = 0;
								for (int b = 0; b < nd->visited_nodes.size(); b++) {
									visited_leaf_id += nd->visited_nodes[b] * two_pow[nd->visited_nodes.size() - b - 1];
								}
								path_value_visited_leaf_ids.push_back(visited_leaf_id);

							}
							//if (forest_depth > 0) { cout << "Here !!!!" << endl; }

							double max = -999;
							int res_class;
							double res_prob;

							for (int j = 0; j < NUM_OF_CLASSES; j++) {
								double sum = 0;
								for (int t = 0; t < NUM_OF_TREE; t++) {
									sum += hist[i][t][j];
								}
								if (sum > max) {
									max = sum;
									res_class = j;
									res_prob = max / NUM_OF_TREE;
								}
								class_posterior[j] = sum / NUM_OF_TREE;
							}
							//cout << "resclass:" << res_class<<endl;
							if (res_class == test_data[random_test_sample][0]) { E_rms_PC++; }
							E_rms_P += (1 - res_prob)*(1 - res_prob);
							arr_occur[res_class] += res_prob;// *res_prob;//taking weighted sum
							if (res_prob > max_prob) {
								max_prob = res_prob;
								copy(class_posterior, class_posterior + NUM_OF_CLASSES, max_class_posterior);
								selected_index = d;
								selected_path_value = path_value;// / NUM_OF_TREE;
								selected_path_value_visited_nodes = path_value_visited_nodes;
								selected_path_value_visited_leaf_ids = path_value_visited_leaf_ids;
							}

							//////////////////////////Freeing allocated memory. Important !!!//////////////////////////////

							free(hist[i]);

							//////////////////////////////////////////////////////////////////////////////
						}


					}
					//cout << "Here !!!!" << endl;
					each_data_sample.push_back(test_data[random_test_sample][selected_index]);
					//cout << "Here !!!!!" << endl;
					copy(max_class_posterior, max_class_posterior + NUM_OF_CLASSES, p_info.class_postarior);
					p_info.max_prob = max_prob;
					pa_info.visited_nodes = selected_path_value_visited_nodes;
					pa_info.visited_leaf_ids = selected_path_value_visited_leaf_ids;
					each_data_sample_probability.push_back(p_info);
					each_data_sample_path.push_back(selected_path_value);
					each_data_sample_path_visited_nodes.push_back(pa_info);
				}

			}

			scaled_test_data.push_back(each_data_sample);
			scaled_test_data_probability.push_back(each_data_sample_probability);
			scaled_test_data_path.push_back(each_data_sample_path);
			scaled_test_data_path_visited_nodes.push_back(each_data_sample_path_visited_nodes);

			int max_voted_class = 0;
			double vote_count = -1;
			double vote_sum = 0;
			for (int v = 0; v < NUM_OF_CLASSES; v++) {
				cout << v << " : " << arr_occur[v] << " ";
				vote_sum += arr_occur[v];//for analysis
				if (arr_occur[v]>vote_count) {
					vote_count = arr_occur[v];
					max_voted_class = v;
				}
				
			}

			for (int v = 0; v < NUM_OF_CLASSES; v++) {
				predictions_in_diff_layers[forest_depth][i].confidences[v] = arr_occur[v] / vote_sum * 100;
			}

			predictions_in_diff_layers[forest_depth][i].max_confidence = vote_count / vote_sum * 100;
			predictions_in_diff_layers[forest_depth][i].predicted_class = max_voted_class;
			predictions_in_diff_layers[forest_depth][i].actual_class = test_data[random_test_sample][0];
			cout << "Predicted class: " << max_voted_class << " Actual class: " << test_data[random_test_sample][0] << endl;
			if (max_voted_class == test_data[random_test_sample][0]) { correct++; }
			E_rms_I += (1 - (vote_count / vote_sum) )*(1 - (vote_count / vote_sum));
			/*
			/////////////////for debug/////////////////////
			for (int mn = 0; mn < each_data_sample.size(); mn++) {
				cout << each_data_sample[mn] << " ";
			}
			cout << endl << endl << endl << endl;
			for (int mn = 1; mn < each_data_sample_probability.size(); mn++) {
				cout << each_data_sample_probability[mn].class_postarior[0] << " "<< each_data_sample_probability[mn].class_postarior[1]<<" "<< each_data_sample_probability[mn].class_postarior[2]<<" ";
			}
			cout << endl << endl << endl << endl; break;//for the time being
			///////////////////////////////////////////////
			*/
		}
		//break;//for the time being
		for (int tt = 0; tt < NUM_OF_TREE; tt++) {
			cout << "tree" << tt + 1 << ": largest_leaf_size: " << largest_leaf_size[tt] << endl;
			cout << "tree" << tt + 1 << ": largest_leaf_height: " << largest_leaf_height[tt] << endl;
		}
		cout << "Correct prediction: " << correct << " " << correct / NUM_OF_TEST_SAMPLES * 100 << "%" << endl;
		cout << "E_rms_P: " << E_rms_P / (NUM_OF_TEST_SAMPLES*test_data[0].size() - 1) * 100 << "%" << endl;
		cout << "E_rms_PC: " << E_rms_PC / (NUM_OF_TEST_SAMPLES*test_data[0].size() - 1) * 100 << "%" << endl;
		cout << "E_rms_I: " << E_rms_I / (NUM_OF_TEST_SAMPLES) * 100 << "%" << endl;
		training_data.clear();
		training_data.shrink_to_fit();
		training_data = scaled_training_data;
		cout << "Here!" << endl;
		scaled_training_data.clear();
		scaled_training_data.shrink_to_fit();
		test_data.clear();
		test_data.shrink_to_fit();
		cout << "Here!!" << endl;
		test_data = scaled_test_data;
		cout << "Here!!!" << endl;
		training_data_probability.clear();
		training_data_probability.shrink_to_fit();
		scaled_test_data.clear();
		scaled_test_data.shrink_to_fit();
		cout << "Here!!!+" << endl;
		training_data_probability = scaled_training_data_probability;
		cout << "Here!!!++" << endl;
		test_data_probability.clear();
		test_data_probability.shrink_to_fit();
		scaled_training_data_probability.clear();
		scaled_training_data_probability.shrink_to_fit();
		cout << "Here!!!+++" << endl;
		test_data_probability = scaled_test_data_probability;
		cout << "Here!!!+++-" << endl;
		training_data_path.clear();
		training_data_path.shrink_to_fit();
		scaled_test_data_probability.clear();
		scaled_test_data_probability.shrink_to_fit();
		cout << "Here!!!+++--" << endl;
		training_data_path = scaled_training_data_path;
		cout << "Here!!!+++---" << endl;
		training_data_path_visited_nodes.clear();
		training_data_path_visited_nodes.shrink_to_fit();
		scaled_training_data_path.clear();
		scaled_training_data_path.shrink_to_fit();
		cout << "Here!!!+++---!" << endl;
		training_data_path_visited_nodes = move(scaled_training_data_path_visited_nodes);
		//training_data_path_visited_nodes = scaled_training_data_path_visited_nodes;
		/*
		for (int kkk = 0; kkk < scaled_training_data_path_visited_nodes.size(); kkk++) {
			cout << "#" << kkk << endl;
			training_data_path_visited_nodes.push_back(scaled_training_data_path_visited_nodes[kkk]);
			cout << "##" << kkk << endl;
			scaled_training_data_path_visited_nodes[kkk].clear();
			cout << "###" << kkk << endl;
			scaled_training_data_path_visited_nodes[kkk].shrink_to_fit();
		}*/
		cout << "Here!!!+++---!!" << endl;
		test_data_path_visited_nodes.clear();
		test_data_path_visited_nodes.shrink_to_fit();
		scaled_training_data_path_visited_nodes.clear();
		scaled_training_data_path_visited_nodes.shrink_to_fit();
		cout << "Here!!!+++---!!!" << endl;
		test_data_path_visited_nodes = move(scaled_test_data_path_visited_nodes);
		//test_data_path_visited_nodes = scaled_test_data_path_visited_nodes;
		/*
		for (int kkk = 0; kkk < scaled_test_data_path_visited_nodes.size(); kkk++) {
			cout << "#" << kkk << endl;
			test_data_path_visited_nodes.push_back(scaled_test_data_path_visited_nodes[kkk]);
			cout << "##" << kkk << endl;
			scaled_test_data_path_visited_nodes[kkk].clear();
			cout << "###" << kkk << endl;
			scaled_test_data_path_visited_nodes[kkk].shrink_to_fit();
		}
		*/
		cout << "Here!!!+++---!!!+" << endl;
		test_data_path.clear();
		test_data_path.shrink_to_fit();
		scaled_test_data_path_visited_nodes.clear();
		scaled_test_data_path_visited_nodes.shrink_to_fit();
		cout << "Here!!!+++---!!!++" << endl;
		test_data_path = scaled_test_data_path;
		cout << "Here!!!+++---!!!+++" << endl;
		scaled_test_data_path.clear();
		scaled_test_data_path.shrink_to_fit();
		cout << "Here!!!+++---!!!+++-" << endl;
		
		/*
		for (int ddd = 0; ddd < training_data.size(); ddd++) {
			for (int eee = 0; eee < training_data[ddd].size(); eee++) {
				printf("%d ", training_data[ddd][eee]);
				printf("%lf", training_data_probability[ddd][eee]);
				for (int fff = 0; fff < training_data_path_visited_nodes[ddd][eee].visited_nodes.size(); fff++) {
					printf("-1 ");
					for (int nnn = 0; nnn < training_data_path_visited_nodes[ddd][eee].visited_nodes[fff].size(); fff++) {
						printf("%d ", training_data_path_visited_nodes[ddd][eee].visited_nodes[fff][nnn]);
					}
					printf("-1 ");
				}
				
				printf("%d ", test_data[ddd][eee]);
				printf("%lf", test_data_probability[ddd][eee]);
				for (int fff = 0; fff < test_data_path_visited_nodes[ddd][eee].visited_nodes.size(); fff++) {
					printf("-1 ");
					for (int nnn = 0; nnn < test_data_path_visited_nodes[ddd][eee].visited_nodes[fff].size(); nnn++) {
						printf("%d ", test_data_path_visited_nodes[ddd][eee].visited_nodes[fff][nnn]);
					}
					printf("-1 ");
				}
				printf("\n");
			}
			
		}
		
		break;
		*/
		
		
		
		/*
		for (map<double, int>::iterator hist = prob_estimate_count.begin(); hist != prob_estimate_count.end(); ++hist) {
		cout << hist->first << " : " << hist->second << endl;
		}*/


		/////////////////////////////////////////////////////Output Scaled features to be used with classic RF//////////////////////////////
		
		if (forest_depth > 1) {
			cout << endl << "------------++++++++++++++++++++++++Output Scaled features to be used with classic RF++++++++++++++++++++--------------------" << endl;
			cout << "------------++++++++++++++++++++++++++++++++Intensity training data++++++++++++++++--------------------------" << endl;
			for (int p = 0; p < training_data.size(); p++) {
				for (int q = 0; q < training_data[p].size(); q++) {
					cout << training_data[p][q] << " ";
				}
				cout << endl;
			}

			cout << "------------++++++++++++++++++++++++++++++++Intensity test data++++++++++++++++--------------------------" << endl;
			for (int p = 0; p < test_data.size(); p++) {
				for (int q = 0; q < test_data[p].size(); q++) {
					cout << test_data[p][q] << " ";
				}
				cout << endl;
			}

			cout << "------------++++++++++++++++++++++++++++++++Probability estimate training data++++++++++++++++--------------------------" << endl;
			for (int p = 0; p < training_data_probability.size(); p++) {
				for (int q = 0; q < training_data_probability[p].size(); q++) {
					for (int cl = 0; cl < NUM_OF_CLASSES; cl++) {
						cout << training_data_probability[p][q].class_postarior[cl] << " ";
					}
					
				}
				cout << endl;
			}

			cout << "------------++++++++++++++++++++++++++++++++Probability estimate test data++++++++++++++++--------------------------" << endl;
			for (int p = 0; p < test_data_probability.size(); p++) {
				for (int q = 0; q < test_data_probability[p].size(); q++) {
					for (int cl = 0; cl < NUM_OF_CLASSES; cl++) {
						cout << test_data_probability[p][q].class_postarior[cl] << " ";
					}
					
				}
				cout << endl;
			}

			cout << "------------++++++++++++++++++++++++++++++++Path info training data++++++++++++++++--------------------------" << endl;
			for (int p = 0; p < training_data_path_visited_nodes.size(); p++) {
				for (int q = 0; q < training_data_path_visited_nodes[p].size(); q++) {
					for (int tr = 0; tr < training_data_path_visited_nodes[p][q].visited_nodes.size(); tr++) {
						for (int nd = 0; nd < training_data_path_visited_nodes[p][q].visited_nodes[tr].size(); nd++) {
							cout << training_data_path_visited_nodes[p][q].visited_nodes[tr][nd];
						}
						cout << " ";
					}
					cout << " ";
				}
				cout << endl;
			}

			cout << "------------++++++++++++++++++++++++++++++++Path info test data++++++++++++++++--------------------------" << endl;
			for (int p = 0; p < test_data_path_visited_nodes.size(); p++) {
				for (int q = 0; q < test_data_path_visited_nodes[p].size(); q++) {
					for (int tr = 0; tr < test_data_path_visited_nodes[p][q].visited_nodes.size(); tr++) {
						for (int nd = 0; nd < test_data_path_visited_nodes[p][q].visited_nodes[tr].size(); nd++) {
							cout << test_data_path_visited_nodes[p][q].visited_nodes[tr][nd];
						}
						cout << " ";
					}
					cout << " ";
				}
				cout << endl;
			}

			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		}
		/*else {
			cout << "Writing intensity:" << endl;
			for (int q = 0; q < test_data[0].size(); q++) {
				cout << test_data[0][q] << " ";
			}
			cout << endl << endl << endl;
			for (int q = 0; q < test_data[1].size(); q++) {
				cout << test_data[1][q] << " ";
			}
			cout << endl << endl << endl;
			cout << "------------------------------------------------" << endl;
			for (int q = 0; q < test_data[2].size(); q++) {
				cout << test_data[2][q] << " ";
			}
			cout << endl << endl << endl;
			cout << "Writing probability estimates" << endl;
			for (int q = 0; q < test_data_probability[0].size(); q++) {
				for (int cl = 0; cl < NUM_OF_CLASSES; cl++) {
					cout << test_data_probability[0][q].class_postarior[cl] << " ";
				}

			}
			cout << endl << endl << endl;
			cout << "------------------------------------------------" << endl;
			for (int q = 0; q < test_data_probability[1].size(); q++) {
				for (int cl = 0; cl < NUM_OF_CLASSES; cl++) {
					cout << test_data_probability[1][q].class_postarior[cl] << " ";
				}

			}
			cout << "------------------------------------------------" << endl;
			for (int q = 0; q < test_data_probability[2].size(); q++) {
				for (int cl = 0; cl < NUM_OF_CLASSES; cl++) {
					cout << test_data_probability[2][q].class_postarior[cl] << " ";
				}

			}
			cout << endl << endl << endl;
		}*/
		
	}

	///////////////////////////////////Predicting considering results of previous layers/////////////////////////////////////////////////
	/*
	for (int i = 0; i < MAX_FOREST_DEPTH; i++) {
		int correct_pred_count = 0;
		cout << "------------------------------++++++++++++++++++++Greedily chosing most confident prediction++++++++++++++++-----------------------------" << endl;
		for (int t = 0; t < NUM_OF_TEST_SAMPLES; t++) {
			//cout << predictions_in_diff_layers[i][t].max_confidence <<" "<<predictions_in_diff_layers[i][t].predicted_class << endl;
			double max_conf = -99;
			int pred_class = 0;
			for (int j = i; j >= 0; j--) {
				if (predictions_in_diff_layers[j][t].max_confidence > max_conf) {
					max_conf = predictions_in_diff_layers[j][t].max_confidence;
					pred_class = predictions_in_diff_layers[j][t].predicted_class;
				}
			}
			if (pred_class == predictions_in_diff_layers[i][t].actual_class) { correct_pred_count++; }
		}

		cout << "At layer " << i << ": " << "accuracy" << " : " << (double)correct_pred_count / (double)NUM_OF_TEST_SAMPLES * 100<<endl;
	}


	for (int i = 0; i < MAX_FOREST_DEPTH; i++) {
		int correct_pred_count = 0;
		cout << "------------------------------++++++++++++++++++++Averaging confidence from diff layers++++++++++++++++-----------------------------" << endl;
		for (int t = 0; t < NUM_OF_TEST_SAMPLES; t++) {
			//cout << predictions_in_diff_layers[i][t].max_confidence <<" "<<predictions_in_diff_layers[i][t].predicted_class << endl;
			double confidence_sum[NUM_OF_CLASSES];
			double max_conf = -99;
			int pred_class = 0;
			for (int v = 0; v < NUM_OF_CLASSES; v++) {
				confidence_sum[v] = 0;
				for (int j = i; j >= 0; j--) {
					confidence_sum[v] += predictions_in_diff_layers[j][t].confidences[v];
				}

				if (confidence_sum[v] > max_conf) {
					max_conf = confidence_sum[v];
					pred_class = v;
				}
			}

			if (pred_class == predictions_in_diff_layers[i][t].actual_class) { correct_pred_count++; }
		}

		cout << "At layer " << i << ": " << "accuracy" << " : " << (double)correct_pred_count / (double)NUM_OF_TEST_SAMPLES * 100 << endl;
	}

	*/
	/*
	Building the Forest with NUM_OF_TREE
	*/




	//return 0;


	//////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	Mat vect;
	vect = Mat::zeros(IMAGE_DATA_DIMENSION, IMAGE_DATA_DIMENSION, CV_8UC1);
	for (int i = 0; i < 2; i++) {
	int counter = 0;
	for (int r = 0; r < IMAGE_DATA_DIMENSION; r++) {
	for (int c = 0; c < IMAGE_DATA_DIMENSION; c++) {
	vect.at<uchar>(r, c) = training_data[i][counter];
	//cout << (int)vect.at<uchar>(r, c)<<" ";
	counter++;
	}
	//cout << endl;
	}
	namedWindow("Display window_" + i, WINDOW_AUTOSIZE);// Create a window for display.
	moveWindow("Display window_" + i, 50, 100);
	imshow("Display window_" + i, vect);
	}


	*/
	/////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	cout << "Total unique response: " << global_response.size()<<endl;

	for (map<int, int>::iterator hist = global_response.begin(); hist != global_response.end(); ++hist) {
	if (hist->first > max_response) {
	max_response = hist->first;
	}
	if (hist->first < min_response) {
	min_response = hist->first;
	}
	cout << hist->first << ": " << hist->second << endl;

	}

	cout << "Max response: " << max_response << " " << "Min response: " << min_response << endl;
	*/
	//Now displaying first two images from data sample
	/*
	Mat vect = Mat::zeros(28, 28, CV_8UC1);

	// Loop over vectors and add the data
	int counter = 1;
	cout << training_data.size() << endl;

	for (int r = 0; r < 28; r++) {
	for (int c = 0; c < 28; c++) {
	vect.at<uchar>(r, c) = training_data[10][counter];
	//cout << (int)vect.at<uchar>(r, c)<<" ";
	counter++;
	}
	//cout << endl;
	}
	cout << counter << endl;
	namedWindow("Display window", 0.5);// Create a window for display.
	imshow("Display window", vect);
	//code for reading image files
	String path("./data/*jpg");
	vector<String> fn;
	vector<int>training_data_index;
	int sample_index = 0;
	glob(path, fn, true);
	//Mat img = imread(path,CV_LOAD_IMAGE_COLOR);
	//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	//imshow("Display window", img);
	*/

	/*
	for (size_t k = 0; k < fn.size(); k++) {
	Mat img = imread(fn[k], CV_LOAD_IMAGE_COLOR);
	if (!img.data) { continue; };
	training_data.push_back(img);
	training_data_index.push_back(sample_index);
	cout<<sample_index++;
	//training data labels also has to be loaded
	namedWindow("Display window"+k, 0.5);// Create a window for display.
	imshow("Display window"+k, training_data.at(k));                   // Show our image inside it.
	Mat grey;
	cvtColor(training_data.at(k), grey, CV_BGR2GRAY);
	Scalar intensity = (training_data.at(k).at<uchar>(50, 50));
	cout << (int)training_data.at(k).at<uchar>(50, 50) << endl;
	cout << intensity.val[0] << " "<<intensity.val[1]<< " "<<intensity.val[2]<<" "<<(int)grey.at<uchar>(50,50)<<endl;
	}



	Mat mytestimg = training_data.at(0);
	//cout<<mytestimg.at(Point(5,5));
	populateFilters();
	Node *root = new Node();
	root = BuildTree(root,1, training_data_index);
	getPrediction(root, 12);
	getPrediction(root, 10);
	//getPrediction(root, 8);
	*/
	//getchar();
	//read from MNIST

	//vector<vector<double>> ar;
	//ReadMNIST(5, 784, ar);
	//Mat mat_img;
	//cout << ar.size() <<" "<< ar.at(3).size() << endl;
	/*for (size_t i = 0; i < ar.size(); i++)
	{
	if (i == 2)break;
	for (size_t j = 0; j < ar.at(i).size(); j++)
	{
	mat_img.at<double>(i, j) = ar[i][j];
	}

	}*/
	//getchar();
	//waitKey(0);
	long long stop = clock();
	cout << "Total execution time: " << stop - start << endl;
	return 0;
}