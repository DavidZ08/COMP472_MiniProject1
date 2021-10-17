import sklearn.datasets as skd
import sklearn.model_selection as skm
import sklearn.naive_bayes as sknb
import sklearn.metrics as skme
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer


def plotting(dirpath):
    # This function goes through the BBC directory, and count each file for each class before plotting a bar graph.
    # Uses os and matplotlib.pyplot.
    category_list = list()
    category_frequency_list = list()
    for base, dirs, files in os.walk(dirpath):
        for directory in dirs:
            category_list.append(os.path.join("", directory))  # Appending each class name to the list
        category_frequency_list.append(len(files))  # Appending the number of files in each class, to a different list
    category_frequency_list.pop(0)  # first value of the array was 0, needed to be removed for plotting
    plt.bar(category_list, category_frequency_list)
    plt.savefig("BBC-distribution.pdf")
    return category_list, category_frequency_list


def load_corpus(category_list, dirpath):
    # This function loads the corpus. Uses sklearn.datasets
    data_set = skd.load_files(dirpath, categories=category_list, encoding="latin1", load_content=True)
    return data_set


def preprocessing(data_set):
    # This function uses a CountVectorizer to produce a term-document matrix from the dataset (data_set.data),
    # also returns the vocabulary. Uses sklearn.datasets.
    vectorizer = CountVectorizer()
    term_document_matrix = vectorizer.fit_transform(data_set.data)
    return term_document_matrix.toarray(), vectorizer.get_feature_names_out()


def dataset_split(data_set, t_d_matrix):
    # This function splits the term-document matrix and the targets (the classes) into training sets (80%)
    # and validation sets (20%). Uses sklearn.model_selection.
    x_train, x_test, y_train, y_test = skm.train_test_split(t_d_matrix, data_set.target, train_size=0.8,
                                                            random_state=None)
    return x_train, x_test, y_test, y_train


def train_nb_classifier(x_train, y_train):
    # This function trains the Naive-Bayes classifier with default values. Uses sklearn.naive_bayes
    nb_classifier = sknb.MultinomialNB()
    nb_classifier.fit(X=x_train, y=y_train)
    return nb_classifier


def train_nb_classifier_smoothing(x_train, y_train, smoothing):
    # This function trains the Naive-Bayes classifier with a custom value for smoothing. Uses sklearn.naive_bayes
    nb_classifier = sknb.MultinomialNB(alpha=smoothing)
    nb_classifier.fit(X=x_train, y=y_train)
    return nb_classifier


def test_nb_classifier(x_test, nb_classifier):
    # This function tests the Naive-Bayes classifier with a test input (from the validation set) and returns an
    # output (the prediction).
    prediction = nb_classifier.predict(x_test)
    return prediction


def compute_confusion_matrix(prediction, y_test):
    # This function computes the confusion matrix from the prediction output and test output (true values).
    # Uses sklearn.metrics.
    confusion_matrix = skme.confusion_matrix(y_true=y_test, y_pred=prediction)
    return confusion_matrix


def compute_classificationreport(prediction, y_test, data_set):
    # This function computes the classification report from the prediction output, test output and the dataset targets.
    # Uses sklearn.metrics
    classification_report = skme.classification_report(y_true=y_test, y_pred=prediction,
                                                       target_names=data_set["target_names"])
    return classification_report


def compute_f1_values(prediction, y_test):
    # This function computes the f1 values from the prediction output and test output.
    # Uses sklearn.metrics
    accuracy = skme.accuracy_score(y_true=y_test, y_pred=prediction)
    macroavg_f1 = skme.f1_score(y_true=y_test, y_pred=prediction, average="macro")
    weightedavg_f1 = skme.f1_score(y_true=y_test, y_pred=prediction, average="weighted")
    return accuracy, macroavg_f1, weightedavg_f1


def compute_prior_averages(nb_classifier):
    # This function computes the prior averages for each class.
    log_prior_classes = nb_classifier.class_log_prior_
    return log_prior_classes


def compute_vocab_size(nb_classifier):
    # This function computes the vocabulary size.
    return nb_classifier.n_features_in_


def compute_word_tokens_per_class(nb_classifier, category_list):
    # This function computes the word tokens per class.
    # It takes the the feature count and sums it up for each class, before appending it to a list.
    words_per_class_list = list()
    for i in range(len(category_list)):
        words_per_class_list.append(sum(nb_classifier.feature_count_[i]))
    return words_per_class_list


def compute_word_tokens_entire_corpus(words_per_class_list):
    # This function computes the word tokens in the entire corpus.
    # It takes the sum of word tokens for each class and adds them together.
    words_corpus = 0
    for i in range(len(words_per_class_list)):
        words_corpus += words_per_class_list[i]
    return words_corpus


def compute_frequency_0_per_class(t_d_matrix, features, category_list):
    # This function computes the number and percentage of words with a frequency of 0 in each class
    # It returns two lists, one list for the numbers for each class, one list for the percentages for each class.
    number_class = 0
    frequency_0_per_class_list_numbers = list()
    frequency_0_per_class_list_percentage = list()
    for i in range(len(category_list)):
        temp_array = t_d_matrix[:, i]  # This creates a temporary array for each column of the term_document matrix.
        temp_array2 = np.where(temp_array == 0)  # This creates an array of indexes that indicate where there elements.
        # equal to zero.
        number_class = len(temp_array2)  # The length of the previously mentioned array is the frequency of zeroes in
        # the first temporary array.
        frequency_0_per_class_list_numbers.append(number_class)
        frequency_0_per_class_list_percentage.append((number_class/len(features)))
    return frequency_0_per_class_list_numbers, frequency_0_per_class_list_percentage


def compute_frequency_0_corpus(frequency_0_per_class_list_numbers, features):
    # This function computes the number and percentage of words with a frequency of 0 in the entire corpus.
    # It sums the number of words with a frequency of zero in each class and sums them before return the number of words
    # with a frequency of 0 in the corpus, and the percentage associated with it.
    frequency_0_corpus_number = sum(frequency_0_per_class_list_numbers)
    frequency_0_corpus_percentage = frequency_0_corpus_number/len(features)
    return frequency_0_corpus_number, frequency_0_corpus_percentage


def favorite_words(nb_classifier, features):
    # This function appends the favorite words into a list, and their log-prob into another.
    # The words are "00" and "000".
    words_string = list()
    words_log_prob = list()
    words_string.append(features[0])
    words_string.append(features[1])
    words_log_prob.append(nb_classifier.feature_log_prob_[0][0])
    words_log_prob.append(nb_classifier.feature_log_prob_[0][1])
    return words_string, words_log_prob


def calculate_stats_and_save(category_list, data_set, features, file, term_document_matrix, x_test, y_test, try_id,
                             nb_classifier):
    # This function assembles all previous functions, executes them, and appends the result in a .txt file.
    # The try_id separates each iteration of this function.
    # This function can be used with different Naive-Bayes classifiers.
    prediction = test_nb_classifier(x_test, nb_classifier)
    confusion_matrix = compute_confusion_matrix(prediction, y_test)
    classification_report = compute_classificationreport(prediction, y_test, data_set)
    accuracy, macroavg_f1, weightedavg_f1 = compute_f1_values(prediction, y_test)
    priors = compute_prior_averages(nb_classifier)
    vocab_size = compute_vocab_size(nb_classifier)
    word_tokens_per_class = compute_word_tokens_per_class(nb_classifier, category_list)
    word_tokens_corpus = compute_word_tokens_entire_corpus(word_tokens_per_class)
    frequency_0_per_class_list_numbers, frequency_0_per_class_list_percentage = compute_frequency_0_per_class(
        term_document_matrix, features, category_list)
    frequency_0_corpus_numbers, frequency_0_corpus_percentage = compute_frequency_0_corpus(
        frequency_0_per_class_list_numbers, features)
    words_string, words_log_prob = favorite_words(nb_classifier, features)

    # Writing to the .txt file
    file.write(try_id + "\n")
    file.write("Confusion matrix: \n" + np.array2string(confusion_matrix) + "\n")
    file.write("\nClassification report: \n")
    file.write(classification_report + "\n")
    file.write("Accuracy, marco average f1, weighted average f1: \n")
    file.write(str(accuracy) + ", " + str(macroavg_f1) + ", " + str(weightedavg_f1) + "\n")
    file.write("Priors: \n")
    file.write(", ".join(str(p) for p in priors) + "\n")
    file.write("Vocabulary size: \n")
    file.write(str(vocab_size) + "\n")
    file.write("Word tokens per class: \n")
    file.write(", ".join(str(p) for p in word_tokens_per_class) + "\n")
    file.write("Word tokens in corpus \n")
    file.write(str(word_tokens_corpus) + "\n")
    file.write("The number of words with frequency of 0 in each class: \n")
    file.write(", ".join(str(p) for p in frequency_0_per_class_list_numbers) + "\n")
    file.write("The percentage of words with frequency of 0 in each class: \n")
    file.write(", ".join(str(p) for p in frequency_0_per_class_list_percentage) + "\n")
    file.write("The number of words with frequency of 0 in the corpus: \n")
    file.write(str(frequency_0_corpus_numbers) + "\n")
    file.write("The percentage of words with frequency of 0 in the corpus: \n")
    file.write(str(frequency_0_corpus_percentage) + "\n")
    file.write("My two favorite words in the vocabulary: \n")
    file.write(", ".join(str(p) for p in words_string) + "\n")
    file.write("Their log-prob: \n")
    file.write(", ".join(str(p) for p in words_log_prob) + "\n")


def main():
    # This part contains the plotting, loading the dataset, producing the term-document matrix and splitting the dataset
    dirpath = "C:\\Users\\Davmo\\PycharmProjects\\Miniproject1\\BBC"
    category_list, category_frequency_list = plotting(dirpath)
    data_set = load_corpus(category_list, dirpath)
    term_document_matrix, features = preprocessing(data_set)
    x_train, x_test, y_test, y_train = dataset_split(data_set, term_document_matrix)

    # This part contains the training of different Naive-Bayes classifiers, with and without smoothing values.
    nb_classifier_1 = train_nb_classifier(x_train, y_train)
    nb_classifier_2 = train_nb_classifier(x_train, y_train)
    nb_classifier_3 = train_nb_classifier_smoothing(x_train, y_train, 0.0001)
    nb_classifier_4 = train_nb_classifier_smoothing(x_train, y_train, 0.9)

    # This part contains the computing of statistics for each Naive-Bayes classifier and saving them to a .txt file.
    file = open("bbc-performance.txt", "a")
    file.write("The classes are in this order: " + ", ".join(str(c) for c in category_list) + "\n")
    calculate_stats_and_save(category_list, data_set, features, file, term_document_matrix, x_test, y_test,
                             "***************************MultinomialNB default values, try 1", nb_classifier_1)
    calculate_stats_and_save(category_list, data_set, features, file, term_document_matrix, x_test, y_test,
                             "***************************MultinomialNB default values, try 2", nb_classifier_2)
    calculate_stats_and_save(category_list, data_set, features, file, term_document_matrix, x_test, y_test,
                             "***************************MultinomialNB smoothing = 0.0001", nb_classifier_3)
    calculate_stats_and_save(category_list, data_set, features, file, term_document_matrix, x_test, y_test,
                             "***************************MultinomialNB smoothing = 0.9", nb_classifier_4)
    file.close()


if __name__ == '__main__':
    main()
