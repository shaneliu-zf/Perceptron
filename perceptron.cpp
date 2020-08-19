#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <functional>

using namespace std;

namespace activation_func{
    const function<double(double)>Identity=[](double x){return x;};
    const function<double(double)>Sigmoid=[](double x){return 1/(1-exp(-x));};
    const function<double(double)>Step=[](double x){return x>0;};
    const function<double(double)>ReLU=[](double x){return x>0?x:0;};
}

class Perceptron{
private:

    double eta;
    double bias;
    vector<double>w;
    function<double(double)>f;

public:

    Perceptron(){
        eta=0.007;
        f=activation_func::Identity;
        bias=1;
    }

    void setEta(double _eta){
        eta=_eta;
    }

    void setN(int N){
        w.resize(N);
        for(int i=0;i<N;i++)w[i]=1;
    }

    void setF(function<double(double)>_f){
        f=_f;
    }

    double Y(vector<double>x)const{
        return f(net(x));
    }

    double net(vector<double>x)const{
        double sigma=0;
        for(int i=0;i<x.size();i++)sigma+=x[i]*w[i];
        sigma+=bias;
        return sigma;
    }
    void show(){
        for(int i=0;i<w.size();i++){
            cout<<"w["<<i<<"]:"<<w[i];
            cout<<endl;
        }
        cout<<"bias:"<<bias<<endl;
    }
    void update(vector<double>x,double T,bool check=false){
        double delta=T-Y(x);
        if(check){
            cout<<"T:"<<T<<" Y:"<<Y(x)<<endl;
            cout<<"delta:"<<delta<<endl;
            for(int i=0;i<w.size();i++){
                cout<<"x["<<i<<"]:"<<x[i]<<" w["<<i<<"]:"<<w[i]<<endl;
            }
            cout<<"bias:"<<bias<<endl;
            cout<<endl;
        }

        for(int i=0;i<w.size();i++){
            w[i]+=eta*delta*x[i];
        }
        bias+=eta*delta;
    }
};

int main(){
    srand(time(NULL));
    Perceptron perceptron;
    vector<double>x;

    /////// Setting ///////

    perceptron.setEta(0.005);//設定學習速率
    x.resize(2);//設定輸入的個數
    perceptron.setF(activation_func::Step);//設定激活函式
    int test=10000;//設定一輪學習的測試比數
    int x_mod=2;//x的隨機範圍0~x_mod-1
    function<double(vector<double>)>T=[](vector<double>x){
        return x[0] && x[1];
    };//正確輸出函式

    /////// Setting ///////

    perceptron.setN(x.size());
    for(int round=1,correct=0;;correct=0,round++){

        for(int t=1;t<=test;t++){
            for(int i=0;i<x.size();i++){
                x[i]=rand()%x_mod;
            }
            perceptron.update(x,T(x));
        }

        for(int t=1;t<=test;t++){
            for(int i=0;i<x.size();i++){
                x[i]=rand()%x_mod;
            }
            double delta=T(x)-perceptron.Y(x);
            if((delta>=0&&delta<0.0001)||(delta<0 && delta>-0.0001)){
                correct++;
            }
        }
        cout<<"round["<<round<<"]correct:"<<correct*1.0/test*100<<"%"<<endl;
        if(correct*1.0/test==1){
            cout<<"Finished learning!"<<endl;
            perceptron.show();
            cout<<endl;
            break;
        }

    }

    while(true){
        cout<<"input test:";
        for(int i=0;i<x.size();i++){
            cin>>x[i];
        }
        cout<<perceptron.Y(x)<<endl;
    }

    return 0;
}
