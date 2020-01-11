/*
 * File:   IWAODE.h
 * Author: think
 *
 * Created on 2015年12月9日, 上午11:05
 */
#include "incrementalLearner.h"
#include "xxxxyDist.h"
#include "stdio.h"
#ifndef NAOMIWGLAODE_H
#define	NAOMIWGLAODE_H

class IWAODE : public IncrementalLearner
{
public:
    IWAODE(char* const *& argv, char* const * end);

    virtual ~IWAODE(void);

     void reset(InstanceStream &is); ///< reset the learner prior to training

       bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()

    /**
     * Inisialises the pass indicated by the parametre.
     *
     * @param pass  Current pass.
     */
    void initialisePass();
    /**
     * Train an aode with instance inst.
     *
     * @param inst Training instance
     */
    void train(const instance &inst);

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param inst The instance to be classified
     * @param classDist Predicted class probability distribution
     */
   virtual void classify(const instance &inst, std::vector<double> &classDist);
    /**
     * Calculates the weight for waode
     */
    void finalisePass();

    void getCapabilities(capabilities &c);

   virtual void printClassifier()
    {
        printf("     纯纯的AODE分类 written by Naomi:\n");
    }

private://protected:只允许子类及本类的成员函数访问 private:只允许本类的成员函数访问
    bool trainingIsFinished_; ///< true iff the learner is trained
    unsigned int noCatAtts_;                                   ///< the number of categorical attributes.
    unsigned int noClasses_;                                   ///< the number of classes
    std::vector<std::vector<double> > HXXC;
    std::vector<double>  HXC ;
    double HC;

    xxyDist xxyDist_; ///< the xxy distribution that aode learns from the instance stream and uses for classification
    InstanceStream* instanceStream_;

};

#endif	/* NAOMIWGLAODE_H */

