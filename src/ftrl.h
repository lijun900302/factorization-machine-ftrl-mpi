#ifndef FTRL_H
#define FTRL_H
#include "load_data.h"
#include "predict.h"
#include "mpi.h"
#include <math.h>

class FTRL{
    public:
        FTRL(Load_Data* load_data, Predict* predict, int total_num_proc, int my_rank) 
                        : data(load_data), pred(predict), num_proc(total_num_proc), rank(my_rank){
                                init();
        }
        ~FTRL(){}

        void init(){
            v_dim = data->glo_fea_dim*data->factor;
            loc_w = new float[data->glo_fea_dim]();
            loc_g = new float[data->glo_fea_dim]();
            glo_g = new float[data->glo_fea_dim]();
            loc_sigma = new float[data->glo_fea_dim]();
            loc_n = new float[data->glo_fea_dim]();
            loc_z = new float[data->glo_fea_dim]();

            loc_v_onedim = new float[v_dim]();
            loc_v=new float*[data->factor];
            for(int i = 0; i < data->factor; i++){
                loc_v[i]=&loc_v_onedim[i*data->glo_fea_dim];
                for(int j = 0; j < data->glo_fea_dim; j++){
                    loc_v[i][j] = 0.01;
                }
            }
            loc_g_v_onedim = new float[v_dim];
            loc_g_v = new float*[data->factor];
            for(int i = 0; i < data->factor; i++){
                loc_g_v[i] = &loc_g_v_onedim[i*data->glo_fea_dim];
            }
            glo_g_v_onedim = new float[v_dim];
            glo_g_v = new float*[data->factor];
            for(int i = 0; i < data->factor; i++){
                glo_g_v[i] = &glo_g_v_onedim[i*data->glo_fea_dim];
            }
            loc_sigma_v_onedim = new float[v_dim]();
            loc_sigma_v = new float*[data->factor];
            for(int i = 0; i < data->factor; i++){
                loc_sigma_v[i] = &loc_sigma_v_onedim[i*data->glo_fea_dim];
            }
            loc_n_v_onedim = new float[v_dim]();
            loc_n_v = new float*[data->factor];
            for(int i = 0; i < data->factor; i++){
                loc_n_v[i]=&loc_n_v_onedim[i*data->glo_fea_dim];
            }
            loc_z_v_onedim = new float[v_dim]();
            loc_z_v = new float*[data->factor];
            for (int i = 0; i < data->factor; i++){
                loc_z_v[i] = &loc_z_v_onedim[i*data->glo_fea_dim];
            }
        }

        float sigmoid(float x){
            if(x < -30) return 1e-6;
            else if(x > 30) return 1.0;
            else{
                double ex = pow(2.718281828, x);
                return ex / (1.0 + ex);
            }
        }

        void update_w(){// only for master node
            for(int col = 0; col < data->glo_fea_dim; col++){
                loc_sigma[col] = ( sqrt (loc_n[col] + glo_g[col] * glo_g[col]) - sqrt(loc_n[col]) ) / alpha;
                loc_n[col] += glo_g[col] * glo_g[col];
                loc_z[col] += glo_g[col] - loc_sigma[col] * loc_w[col];
                if(abs(loc_z[col]) <= lambda1){
                    loc_w[col] = 0.0;
                }
                else{
                    float tmpr= 0.0;
                    if(loc_z[col] >= 0) tmpr = loc_z[col] - lambda1;
                    else tmpr = loc_z[col] + lambda1;
                    float tmpl = -1 * ( ( beta + sqrt(loc_n[col]) ) / alpha  + lambda2);
                    loc_w[col] = tmpr / tmpl;
                }
            }//end for
        }

        void update_v(){// only for master node
            for(int k = 0; k < data->factor; k++){
                for(int col = 0; col < data->glo_fea_dim; col++){
                    loc_sigma_v[k][col] = ( sqrt (loc_n_v[k][col] + glo_g_v[k][col] * glo_g_v[k][col]) - sqrt(loc_n_v[k][col]) ) / alpha;
                    loc_n_v[k][col] += glo_g_v[k][col] * glo_g_v[k][col];
                    loc_z_v[k][col] += glo_g_v[k][col] - loc_sigma_v[k][col] * loc_v[k][col];
                    //std::cout<<"loc_z_v = "<<loc_z_v[k][col]<<std::endl;
                    if(abs(loc_z_v[k][col]) <= lambda1){
                        loc_v[k][col] = 0.0;
                    }
                    else{
                        float tmpr= 0.0;
                        if(loc_z_v[k][col] >= 0) tmpr = loc_z_v[k][col] - lambda1;
                        else tmpr = loc_z_v[k][col] + lambda1;
                        float tmpl = -1 * ( ( beta + sqrt(loc_n_v[k][col]) ) / alpha  + lambda2);
                        loc_v[k][col] = tmpr / tmpl;
                    }
                }//end for
            }//end for
        }

        void batch_gradient_calculate(int &row){
            int index = 0; float value = 0.0; float pctr = 0;
            for(int line = 0; line < batch_size; line++){
                float wx = bias;
                int ins_seg_num = data->fea_matrix[row].size();
                std::vector<float> vx_sum(data->factor, 0.0);
                float vxvx = 0.0, vvxx = 0.0;
                for(int col = 0; col < ins_seg_num; col++){//for one instance
                    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    wx += loc_w[index] * value;
                    for(int k = 0; k < data->factor; k++){
                        int loc_v_temp = loc_v[k][index];
                        vx_sum[k] += loc_v_temp * value;
                        vvxx += loc_v_temp * loc_v_temp * value * value;
                    }
                }
                for(int k = 0; k < data->factor; k++){
                    vxvx += vx_sum[k] * vx_sum[k]; 
                }
                vxvx -= vvxx;
                wx += vxvx * 1.0 / 2.0;
                pctr = sigmoid(wx);
                float delta = pctr - data->label[row];

                for(int col = 0; col < ins_seg_num; col++){
                    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    loc_g[index] += delta * value;
                    float vx = 0.0;
                    for(int k = 0; k < data->factor; k++){
                        vx = loc_v[k][index] * value;
                        loc_g_v[k][index] += delta * (vx_sum[k] - vx) * value;
                    }
                }
                //print2dim(loc_g_v, data->factor, data->glo_fea_dim);
                row++;
            }//end for
        }
        void print2dim(float** a, int m, int n){
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    if(a[i][j] != 0) std::cout<<a[i][j]<<" ";
                }
                std::cout<<std::endl;
            }
        } 
        void save_model(int epoch){
            char buffer[1024];
            snprintf(buffer, 1024, "%d", epoch);
            std::string filename = buffer;
            std::ofstream md;
            md.open("./model/model_epoch" + filename + ".txt");
            if(!md.is_open()){
                std::cout<<"open file error: "<< std::endl;
            }
            float wi;
            for(int j = 0; j < data->glo_fea_dim; j++){
                wi = loc_w[j];
                md<< j << "\t" <<wi<<std::endl;
            }
            md.close();
        }

        void ftrl(){
            int batch_num = data->fea_matrix.size() / batch_size, batch_num_min = 0;
            MPI_Allreduce(&batch_num, &batch_num_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            std::cout<<"total epochs = "<<epochs<<" batch_num_min = "<<batch_num_min<<std::endl;
            for(int epoch = 0; epoch < epochs; epoch++){
                int row = 0, batches = 0;
                std::cout<<"epoch "<<epoch<<" ";
                pred->run(loc_w, loc_v);
                if(rank == 0 && (epoch+1) % 20 == 0) save_model(epoch);
                while(row < data->fea_matrix.size()){
                    if( (batches == batch_num_min - 1) ) break;
                    batch_gradient_calculate(row);
                    if(row % 50000 == 0) std::cout<<"row = "<<row<<std::endl;
                    for(int col = 0; col < data->glo_fea_dim; col++){
                        loc_g[col] /= batch_size;
                    }
                    for(int j = 0; j < data->factor * data->glo_fea_dim; j++){
                        loc_g_v_onedim[j] /= batch_size;
                    }

                    if(rank != 0){//slave nodes send gradient to master node;
                        MPI_Send(loc_g, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
                        MPI_Send(loc_g_v_onedim, data->factor * data->glo_fea_dim, MPI_FLOAT, 0, 399, MPI_COMM_WORLD);
                    }
                    else if(rank == 0){//rank 0 is master node
                        for(int j = 0; j < data->glo_fea_dim; j++){//store local gradient to glo_g;
                            glo_g[j] = loc_g[j];
                        }
                        for(int j = 0; j < data->factor * data->glo_fea_dim; j++){
                            glo_g_v_onedim[j] = loc_g_v_onedim[j];
                        }

                        for(int r = 1; r < num_proc; r++){//receive other node`s gradient and store to glo_g;
                            MPI_Recv(loc_g, data->glo_fea_dim, MPI_FLOAT, r, 99, MPI_COMM_WORLD, &status);
                            for(int j = 0; j < data->glo_fea_dim; j++){
                                glo_g[j] += loc_g[j];
                            }
                            MPI_Recv(loc_g_v_onedim, data->factor * data->glo_fea_dim, MPI_FLOAT, r, 399, MPI_COMM_WORLD, &status);
                            for(int j = 0; j < data->factor * data->glo_fea_dim; j++){
                                glo_g_v_onedim[j] += loc_g_v_onedim[j];
                            }
                        }
                        for(int j = 0; j < data->glo_fea_dim; j++){
                            glo_g[j] /= num_proc;
                        }
                        for(int j = 0; j < data->factor * data->glo_fea_dim; j++){
                            glo_g_v_onedim[j] /= num_proc;
                        }

                        update_w();
                        update_v();
                    }
                    //sync w of all nodes in cluster
                    if(rank == 0){
                        for(int r = 1; r < num_proc; r++){
                            MPI_Send(loc_w, data->glo_fea_dim, MPI_FLOAT, r, 999, MPI_COMM_WORLD);
                            MPI_Send(loc_v_onedim, data->factor * data->glo_fea_dim, MPI_FLOAT, r, 3999, MPI_COMM_WORLD);
                        }
                    }
                    else if(rank != 0){
                        MPI_Recv(loc_w, data->glo_fea_dim, MPI_FLOAT, 0, 999, MPI_COMM_WORLD, &status);
                        MPI_Recv(loc_v_onedim, data->factor * data->glo_fea_dim, MPI_FLOAT, 0, 3999, MPI_COMM_WORLD, &status);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);//will it make the procedure slowly? is it necessary?
                    batches++;
                }//end row while
            }//end epoch for
        }//end ftrl

    public:
        int v_dim;

        float* loc_w;
        float* loc_v_onedim;
        float** loc_v;
        int epochs;
        int batch_size;
        float bias;
        float alpha;
        float beta;
        float lambda1;
        float lambda2;
    private:
        MPI_Status status;

        Load_Data* data;
        Predict* pred;
        
        float* loc_g;
        float* glo_g;
        float* loc_z;
        float* loc_sigma;
        float* loc_n;

        float* loc_g_v_onedim;
        float** loc_g_v;
        float* glo_g_v_onedim;
        float** glo_g_v;
        float* loc_sigma_v_onedim;
        float** loc_sigma_v;
        float* loc_n_v_onedim;
        float** loc_n_v;
        float* loc_z_v_onedim;
        float** loc_z_v;


        int num_proc;
        int rank;
};
#endif
