
# coding: utf-8

# In[18]:


import essentia
import essentia.standard as es
import os
import json
import numpy as np
from sklearn import preprocessing
import csv, json, sys



#'dan','hip','jaz','pop','rhy','roc','spe'
genreDir = './genre/'
genreList = ['cla','hip','jaz','pop', 'rhy', 'roc', 'spe']
songClass_dict = dict()

for genre in genreList:
    songDir = genreDir+genre
    print(songDir)    
      
    for subdir, dirs, files in os.walk(songDir):
        feature_list = []
        
        for file in files:

            audio = str(songDir+"/"+file)
        #    print(audio)
            
        # Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
            features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'], 
                                                          rhythmStats=['mean', 'stdev'], 
                                                          tonalStats=['mean', 'stdev'])(audio)    
            feature_dict = {}
            

            for feature in features.descriptorNames():
                if feature.find('lowlevel') != -1 or feature.find('rhythm') != -1 or feature.find('tonal') != -1:       
                    if type(features[feature]) != str and type(features[feature]) != np.ndarray:
                        if feature.find('mean') != -1 or feature.find('stdev') != -1:            
                            feature_dict[feature] = features[feature] 
                            
                            
            #preprocessing step.Standerdising the features
            
            data_array = []       
            
            for value in feature_dict.values():
                data_array.append(value)    
            
            data_array = preprocessing.scale(data_array)            
            feature_list.append(data_array)
             

        songClass_dict[genre] = feature_list
        

    



# In[17]:


print(songClass_dict)

def create_train_and_test_sets(dataset, class_names, percentage_training_data, 
                               max_input_tags_for_testing):
    
    
    training_set = dict()
    testing_set = dict()

    # Get 'n_training_sounds_per_class' sounds per class 
    for genre_name, features in dataset.items():
        n_training_features_per_genre = int(len(features) * percentage_training_data)
        print(n_training_features_per_genre)
        features_from_genre = features[:] # Copy the list so when we later shuffle it does not affect the original data 
        
        training_set[genre_name] = features_from_genre[:n_training_features_per_genre] # First sounds for training
        testing_set[genre_name] = features_from_genre[n_training_features_per_genre:] # Following sounds for testing
     
        # Save a trimmed version of input tags for testing sounds
  #      for sound in testing_set[genre_name]:
   #         sound['tags'] = random.sample(sound['tags'], min(max_input_tags_for_testing, len(sound['tags'])))

    print('Created training and testing sets with the following number of sounds:\n\tTrain\tTest')
    for genre_name in class_names:
        training_sounds = training_set[genre_name]
        testing_sounds = testing_set[genre_name]
        print('\t%i\t%i\t%s' % (len(training_sounds), len(testing_sounds), genre_name))
    return training_set, testing_set


PERCENTAGE_OF_TRAINING_DATA = 0.75 # Percentage of sounds that will be used for training (others are for testing)
MAX_INPUT_TAGS_FOR_TESTING = 110 # Use a big number to "omit" this parameter and use as many tags as originally are in the sound

training_set, testing_set = create_train_and_test_sets(
    dataset=songClass_dict, 
    class_names=genreList,
    percentage_training_data=PERCENTAGE_OF_TRAINING_DATA,
    max_input_tags_for_testing=MAX_INPUT_TAGS_FOR_TESTING,
)

    
  


# In[78]:


#def build_tag_feature_vector(sound):
 #   tag_features = utils.get_feature_vector_from_tags(sound['tags'], prototype_feature_vector)
 #   return np.concatenate([[], tag_features])

def train_classifier(training_set, classifier_type, class_names, dataset_name, feature_vector_func, 
                     feature_vector_dimension_labels=None, tree_max_depth=5):
    
    # Prepare data for fitting classifier (as sklearn classifiers require)
    classes_vector = list()
    feature_vectors = list()
    for genre_name, features in training_set.items():
        for count, feature in enumerate(features):
            # Use index of class name in class_names as numerical value (classifier internally represents 
            # class label as number)
            classes_vector.append(class_names.index(genre_name))
            feature_vector = feature_vector_func(sound)
            feature_vectors.append(feature_vector)

    # Create and fit classifier
    print('Training classifier (%s) with %i sounds...' % (CLASSIFIER_TYPE, len(feature_vectors)))
    if classifier_type == 'svm':
        classifier = svm.LinearSVC()
        classifier.fit(feature_vectors, classes_vector)
    elif classifier_type == 'tree':
        classifier = tree.DecisionTreeClassifier(max_depth=tree_max_depth)
        classifier.fit(feature_vectors, classes_vector)
        # Plot classifier decision rules
        # WARNING: do not run this if tree is too big, might freeze
        out_filename = '%s_tree_%i.png' % (dataset_name, random.randint(1000,9999))
        utils.export_tree_as_graph(
            classifier, feature_vector_dimension_labels, class_names=class_names, filename=out_filename)
        display(HTML('<h4>Learned tree:</h4><img src="%s"/>' % out_filename))
    else:
        raise Exception('Bad classifier type!!!')
    
    print('done!')
    return classifier

CLASSIFIER_TYPE = 'tree' # Use 'svm' or 'tree'

classifier = train_classifier(
    training_set=training_set,
    classifier_type=CLASSIFIER_TYPE, 
    class_names=genreList, 
    dataset_name= songClass_dict,
    feature_vector_func=build_tag_feature_vector,
    feature_vector_dimension_labels=prototype_feature_vector,  # This is used to show the class names in the tree image
)


# In[ ]:


def evaluate_classifier(testing_set, classifier, class_names, feature_vector_func, show_confussing_matrix=True):
    # Test with testing set
    print('Evaluating with %i instances...' % sum([len(sounds) for sounds in testing_set.values()]))
    predicted_data = list()
    for class_name, sounds in testing_set.items():
        for count, sound in enumerate(sounds):
            feature_vector = feature_vector_func(sound)
            predicted_class_name = class_names[classifier.predict([feature_vector])[0]]
            predicted_data.append((sound['id'], class_name, predicted_class_name))     
    print('done!')

    # Compute overall accuracy
    good_predictions = len([1 for sid, cname, pname in predicted_data if cname == pname])
    wrong_predictions = len([1 for sid, cname, pname in predicted_data if cname != pname])
    print('%i correct predictions' % good_predictions)
    print('%i wrong predictions' % wrong_predictions)
    accuracy = float(good_predictions)/(good_predictions + wrong_predictions)
    print('Overall accuracy %.2f%%' % (100 * accuracy))
    
    if show_confussing_matrix:
        # Compute confussion matrix (further analysis)
        matrix = list()
        for class_name in class_names:
            predicted_classes = list()
            for sid, cname, pname in predicted_data:
                if cname == class_name:
                    predicted_classes.append(pname)
            matrix.append([predicted_classes.count(target_class) for target_class in class_names])

        # Plot confussion matrix
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(matrix, cmap=plt.cm.Blues, interpolation='nearest')

        for x in xrange(len(matrix)):
            for y in xrange(len(matrix)):
                ax.annotate(str(matrix[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')

        shortened_class_names = [item[0:10] for item in class_names]
        plt.xticks(range(len(class_names)), shortened_class_names, rotation=90)
        plt.yticks(range(len(class_names)), shortened_class_names)
        plt.xlabel('Predicted classes')
        plt.ylabel('Groundtruth classes')

        print('Confussion matrix')
        plt.show()
    
    return accuracy
    
evaluate_classifier(
    testing_set=testing_set,
    classifier=classifier,
    class_names=genreList,
    feature_vector_func=build_tag_feature_vector,
)

