#ifndef PREDICT_H_
#define PREDICT_H_

#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>

typedef struct{
    float clk;
    float nclk;
    long idx;
} clkinfo;

class Predict{
    public:
    Predict(Load_Data* load_data, int total_num_proc, int my_rank) 
            : data(load_data), nproc(total_num_proc), rank(my_rank){
        pctr = 0.0;
        MAX_ARRAY_SIZE = 2000;
        g_all_non_clk = new float[MAX_ARRAY_SIZE];
        g_all_clk = new float[MAX_ARRAY_SIZE];
        g_nclk = new float[MAX_ARRAY_SIZE];
        g_clk = new float[MAX_ARRAY_SIZE];
    }
    ~Predict(){
        delete[] g_all_non_clk;
        delete[] g_all_clk;
        delete[] g_nclk;
        delete[] g_clk;
    }

    void predict(float* glo_w, float** glo_v){
        int index = 0; float value = 0.0; float pctr = 0.0;
        for(int i = 0; i < data->fea_matrix.size(); i++) {
	        float wx = 0.0;
            for(int j = 0; j < data->fea_matrix[i].size(); j++) {
                index = data->fea_matrix[i][j].idx;
                value = data->fea_matrix[i][j].val;
                wx += glo_w[index] * value;
            }
            for(int k = 0; k < data->factor; k++){
                float vxvx = 0.0, vvxx = 0.0;
                for(int col = 0; col < data->fea_matrix[i].size(); col++){
                    index = data->fea_matrix[i][col].idx;
                    value = data->fea_matrix[i][col].val;
                    vxvx += glo_v[k][index] * value;
                    vvxx += glo_v[k][index] * glo_v[k][index] * value * value;
                }
                vxvx *= vxvx;
                vxvx -= vvxx;
                wx += vxvx * 1.0 / 2.0;
            }
            if(wx < -30){
                pctr = 1e-6;
            }
            else if(wx > 30){
                pctr = 1.0;
            }
            else{
                double ex = pow(2.718281828, wx);
                pctr = ex / (1.0 + ex);
            }
            int id = int(pctr*MAX_ARRAY_SIZE);
            clkinfo clickinfo;
            clickinfo.clk = data->label[i];
            clickinfo.nclk = 1 - data->label[i];
            clickinfo.idx = id;
            result_list.push_back(clickinfo);
        }
    }

    void merge_clk(){//merge local node`s clk
        memset(g_nclk, 0.0, MAX_ARRAY_SIZE * sizeof(float));
        memset(g_clk, 0.0, MAX_ARRAY_SIZE * sizeof(float));
        int cnt = result_list.size();
        for(int i = 0; i < cnt; i++){
            long index = result_list[i].idx;
            g_nclk[index] += result_list[i].nclk;
            g_clk[index] += result_list[i].clk;
        }
    }

    int auc_cal(float* all_clk, float* all_nclk, double& auc_res){
            double clk_sum = 0.0;
            double nclk_sum = 0.0;
            double old_clk_sum = 0.0;
            double clksum_multi_nclksum = 0.0;
            auc_res = 0.0;
            for(int i = 0; i < MAX_ARRAY_SIZE; i++){
                    old_clk_sum = clk_sum;
                    clk_sum += all_clk[i];
                    nclk_sum += all_nclk[i];
                    auc += (old_clk_sum + clk_sum) * all_nclk[i] / 2;
            }
            clksum_multi_nclksum = clk_sum * nclk_sum;
            auc_res = auc/(clksum_multi_nclksum);
    }

    int mpi_auc(int nprocs, int rank, double& auc){
        MPI_Status status;
        if(rank != MASTER_ID){
            MPI_Send(g_nclk, MAX_ARRAY_SIZE, MPI_FLOAT, MASTER_ID, 199, MPI_COMM_WORLD);
            MPI_Send(g_clk, MAX_ARRAY_SIZE, MPI_FLOAT, MASTER_ID, 1999, MPI_COMM_WORLD);
        }
        else if(rank == MASTER_ID){
            for(int i = 0; i < MAX_ARRAY_SIZE; i++){
                g_all_non_clk[i] = g_nclk[i];
                g_all_clk[i] = g_clk[i];
            }
            for(int i = 1; i < nprocs; i++){
                MPI_Recv(g_nclk, MAX_ARRAY_SIZE, MPI_FLOAT, i, 199, MPI_COMM_WORLD, &status);
                MPI_Recv(g_clk, MAX_ARRAY_SIZE, MPI_FLOAT, i, 1999, MPI_COMM_WORLD, &status);
                for(int i = 0; i < MAX_ARRAY_SIZE; i++){
                    g_all_non_clk[i] += g_nclk[i];
                    g_all_clk[i] += g_clk[i];
                }
            }
            auc_cal(g_all_non_clk, g_all_clk, auc);
        }
    }

    //void run(std::vector<float> w){
    void run(float* w, float** v){
        predict(w, v);

        merge_clk();
        mpi_auc(nproc, rank, auc);

        if(MASTER_ID == rank){
            printf("AUC = %lf\n", auc);
        }
    }

    private:
    Load_Data* data;
    std::vector<clkinfo> result_list;
    int MAX_ARRAY_SIZE;
    double auc = 0.0;
    float* g_all_non_clk;
    float* g_all_clk;
    float* g_nclk;
    float* g_clk;
    float g_total_clk;
    float g_total_nclk;

    float pctr;
    int nproc; // total num of process in MPI comm world
    int rank; // my process rank in MPT comm world
};
#endif
