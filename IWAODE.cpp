/*
 * File:   newmain.cpp
 * Author: think
 *
 * Created on 2015年12月8日, 上午9:59
 */


#include "IWAODE.h"
#include "utils.h"

IWAODE::IWAODE(char* const *& argv, char* const * end)
{
    name_ = "AODE written by Naomi";
    trainingIsFinished_ = false;
}

 IWAODE::~IWAODE(void)
{
}

void
IWAODE::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void
IWAODE::reset(InstanceStream &is)
{
    xxyDist_.reset(is);
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    HXXC = std::vector<std::vector<double> >(this->noCatAtts_,std::vector<double>(this->noCatAtts_,0));
    HXC = std::vector<double>(this->noCatAtts_,0);
    HC  = 0;

    trainingIsFinished_ = false;
}

void
IWAODE::initialisePass()
{

}

void
IWAODE::train(const instance &inst)
{
    xxyDist_.update(inst);
}

/// true iff no more passes are required. updated by finalisePass()

bool
IWAODE::trainingIsFinished()
{
    return trainingIsFinished_;
}

void
IWAODE::finalisePass()
{
    for(int x1 = 0;x1<this->noCatAtts_;x1++){
        for(int x2 = 0;x2<this->noCatAtts_;x2++){
            //printf("HXXY:");
            if(x1 == x2){
                continue;
            }
            double hxxy =0;
            for(int x1v = 0;x1v<this->instanceStream_->getNoValues(x1);x1v++){
                for(int x2v = 0;x2v<this->instanceStream_->getNoValues(x2);x2v++){
                    for(int y=0;y < this->noClasses_;y++){
                        hxxy += this->xxyDist_.jointP(x1,x1v,x2,x2v,y) * log(xxyDist_.p(x1,x1v,x2,x2v,y));
                    }
                }
            }
            hxxy = -hxxy;
            HXXC[x1][x2] = hxxy;
            //printf("%lf\t",HXXC[x1][x2] );
            assert(HXXC[x1][x2] >= 0);
        }
        //printf("\n");
    }

    for(int x1 = 0;x1<this->noCatAtts_;x1++){
        double hxy =0;
        for(int x1v = 0;x1v<this->instanceStream_->getNoValues(x1);x1v++){
             for(int y=0;y < this->noClasses_;y++){
                 hxy += this->xxyDist_.xyCounts.jointP(x1,x1v,y) * log(1.0/(this->xxyDist_.xyCounts.p(x1,x1v,y)));
             }
        }
        HXC[x1] = hxy;
        assert(HXC[x1] >= 0);
    }
    HC = 0;
    for(int y=0;y < this->noClasses_;y++){
        HC +=this->xxyDist_.xyCounts.p(y) * log(1.0/ (this->xxyDist_.xyCounts.p(y)));
    }
    assert(HC >= 0);

    trainingIsFinished_ = true;
}

void
IWAODE::classify(const instance &inst, std::vector<double> &classDist)
{

    unsigned int noCatAtts=xxyDist_.getNoCatAtts();
    //printf("实例为：");

    //for(int i=0;i<noCatAtts;i++)
    //    printf("%d ",inst.getCatVal(i));
    //printf("\n");
    //HXXC[X1][X2] = H(X1 | X2,C)

    std::vector<std::vector<double> > HxxC = std::vector<std::vector<double> >(this->noCatAtts_,std::vector<double>(this->noCatAtts_,0));
    std::vector<double>  HxC = std::vector<double>(this->noCatAtts_,0);

    for(int x1 = 0;x1<this->noCatAtts_;x1++){
        for(int x2 = 0;x2<this->noCatAtts_;x2++){
            if(x1 == x2){
                continue;
            }
            double hxxy =0;
            for(int y=0;y < this->noClasses_;y++){
                hxxy += this->xxyDist_.jointP(x1,inst.getCatVal(x1),x2,inst.getCatVal(x2),y) * log(xxyDist_.p(x1,inst.getCatVal(x1),x2,inst.getCatVal(x2),y));
            }
            hxxy = -hxxy;
            HxxC[x1][x2] = hxxy;
            assert(HxxC[x1][x2] >= 0);
        }
    }

    for(int x1 = 0;x1<this->noCatAtts_;x1++){
        double hxy =0;
        for(int y=0;y < this->noClasses_;y++){
            hxy += this->xxyDist_.xyCounts.jointP(x1,inst.getCatVal(x1),y) * log(1.0/(this->xxyDist_.xyCounts.p(x1,inst.getCatVal(x1),y)));
        }
        HxC[x1] = hxy;
        assert(HxC[x1] >= 0);
    }





    unsigned int noClasses=xxyDist_.getNoClasses();
    double p=0.0;
    double result=0.0;
    double tmp=0.0;
    std::vector<double> Ws = std::vector<double>(this->noCatAtts_,0);
    std::vector<double> ws = std::vector<double>(this->noCatAtts_,0);
    for(int father=0;father<noCatAtts;father++)
    {
        double W = 0;
        double w = 0;
        W += HC;
        W += HXC[father];
        w += HC;
        w += HxC[father];
        for(int att=0;att<noCatAtts;att++){
            if(att!=father){
                W += HXXC[att][father];
                w += HxxC[att][father];
            }
        }
        Ws[father] = W;
        ws[father] = w;
    }
    normalise(Ws);
    normalise(ws);
    for(int father=0;father<noCatAtts;father++)
    {
        Ws[father] = 1 - Ws[father];
        ws[father] = 1 - ws[father];
    }
    normalise(Ws);
    normalise(ws);



    for(int classValue=0;classValue<noClasses;classValue++)
    {
        result=0.0;//这是记录每一个类取值的结果
        for(int father=0;father<noCatAtts;father++)//皇帝轮流做
        {
            double W = Ws[father];
            double w = ws[father];
            int fatherValue=inst.getCatVal(father);
            double fatherP= xxyDist_.xyCounts.jointP(father,fatherValue, classValue);//不用每次都算
            //printf("fatherP=%f\n",fatherP);
            p=fatherP*(std::numeric_limits<double>::max() /3.0);//计算了P(C=i,Xi=xi)的概率

            for(int att=0;att<noCatAtts;att++)
                if(att!=father)//一定要绕开att等于father的情况，因为xxyDist_里的counts_根本就没有count(x2,x2,x1)的项，比如说取count(x2=x21,x2=x21,C=1) count_[x1][v1 * x1 + x2][v2 * noOfClasses_ + y]  正常来说第二个属性不是0，就是1，应该比2小，结果你又是2，就会取到x2=x22那里去，就越界了
                {
                    tmp=xxyDist_.p(att, inst.getCatVal(att), father, fatherValue, classValue); //P(Xson=xson|Xfather=xfather,C=classValue) 有直接调用的函数就用直接的函数，不要用两个概率相除 M估计有所不同 与petal系统自带的AODE是两个概率相除，在0-1损失上略有出入
                    //printf("tmp=%f\n",tmp);
                    p*=tmp;
                    p *= (W + w);
                }

                result+=p;
        }
        //printf("result=%f\n",result);
        classDist[classValue]=(double)result/noCatAtts;
    }
    normalise(classDist);
    //printf("AODE的结果为：\n");

    //for(int i=0;i<classDist.size();i++)
    //        printf("%d: %f ",i,classDist[i]);
    // printf("\n");

}
