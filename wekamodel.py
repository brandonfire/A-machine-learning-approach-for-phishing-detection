#!/usr/bin/python
# Weka model for training arff data
# Author: Chengbin Hu
# Python: 2.7
 

from __future__ import absolute_import
from __future__ import division
from weka.core.converters import Loader
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.classifiers import PredictionOutput, KernelClassifier, Kernel
from weka.classifiers import Evaluation
from weka.classifiers import FilteredClassifier
from weka.filters import Filter
from weka.core.classes import Random
import weka.plot.classifiers as plcls




def wekamodel():
    jvm.start()
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file("./Training.arff")
    data.class_is_last()
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "1-3"])
    cls = Classifier(classname="weka.classifiers.functions.MultilayerPerceptron")
    print cls.getHiddenLayers()
    fc = FilteredClassifier()
    fc.filter = remove
    fc.classifier = cls
    evl = Evaluation(data)
    evl.crossvalidate_model(fc, data, 10, Random(1))
    print evl.percent_correct
    print evl.summary()
    print evl.class_details()
    plcls.plot_roc(evl, class_index=[0, 1], wait=True)
    jvm.stop()

if __name__ == "__main__":
    wekamodel()
