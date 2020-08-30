import math
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

count_ham_files = len(os.listdir("train/ham/"))
count_spam_files = len(os.listdir("train/spam/"))
total_no_of_files = count_ham_files + count_spam_files
tokenizer = RegexpTokenizer("[a-zA-Z]+")
x = [os.path.join(r, file) for r, d, f in os.walk("train\\spam") for file in f]

    # print(x)
y = [os.path.join(r, file) for r, d, f in os.walk("train\\ham") for file in f]

    # print(y)

training_files = y + x

stemmer = SnowballStemmer("english")

p = [os.path.join(r, file) for r, d, f in os.walk("test\\spam") for file in f]

# print(x)
q = [os.path.join(r, file) for r, d, f in os.walk("test\\ham") for file in f]
ham_test = q
spam_test = p

ham_files = y
spam_files = x


def naive_bayes_with_stop_words():



    total_no_of_words_in_train = 0
    bag_of_words = {}



   # new_files = os.listdir("train/*/*"+".txt")
   # print(new_files)
    getting_loaded = {}
   # print(training_files)


   ##print(art)

   # print("\n\n")


  #  files = filter(os.path.isfile, os.listdir("train/spam"))  # files only
   # files = [f for f in os.listdir("train/spam") if os.path.isfile(f)]
    #destdir = 'train/'




    for file in training_files:
        with open(file, 'r', encoding="latin1") as fp:
            for line in fp:
                tokens = tokenizer.tokenize(line)
                total_no_of_words_in_train = total_no_of_words_in_train + len(tokens)
                stemmed_words = [stemmer.stem(t) for t in tokens]
                for word in stemmed_words:
                        if word.lower() in bag_of_words:
                            bag_of_words[word.lower()] += 1
                        else:
                            bag_of_words[word.lower()] = 1

    no_of_unique_words_in_train = len(bag_of_words)
    #print(no_of_unique_words_in_train)


    #print(ham_files)
    no_of_unique_ham_words_in_ham = {}
    total_ham_words = 0

    for file in ham_files:
        with open(file, 'r', encoding="latin1") as fp:
            for line in fp:
                tokens = tokenizer.tokenize(line)
                total_ham_words += len(tokens)
                stemmed_words = [stemmer.stem(t) for t in tokens]
                for word in stemmed_words:

                        if word.lower() in no_of_unique_ham_words_in_ham:
                            no_of_unique_ham_words_in_ham[word.lower()] += 1
                        else:
                            no_of_unique_ham_words_in_ham[word.lower()] = 1


    no_of_unique_spam_words_in_spam = {}
    total_spam_words = 0

    for file in spam_files:
        with open(file, 'r', encoding="latin1") as fp:
            for line in fp:
                # tokenizing
                tokens = tokenizer.tokenize(line)
                total_spam_words += len(tokens)
                stemmed_words = [stemmer.stem(t) for t in tokens]
                for word in stemmed_words:

                        if word.lower() in no_of_unique_spam_words_in_spam:
                            no_of_unique_spam_words_in_spam[word.lower()] += 1
                        else:
                            no_of_unique_spam_words_in_spam[word.lower()] = 1


    probability_of_ham_class = math.log10(count_ham_files / total_no_of_files)
    probability_of_spam_class = math.log10(count_spam_files / total_no_of_files)

    path_for_ham_train = y
    correct_guesses_during_train = 0
    train_file_count = len(path_for_ham_train)

    for file in path_for_ham_train:
        probability_of_ham = 0
        probability_of_spam = 0
        with open(file, 'r') as fp:
            for line in fp:
                tokens = tokenizer.tokenize(line)
                stemmed_words = [stemmer.stem(str(t)) for t in tokens]
                for word in stemmed_words:
                    if word in no_of_unique_ham_words_in_ham:
                        probability_of_ham = probability_of_ham + math.log10((no_of_unique_ham_words_in_ham[word] + 1) / (total_ham_words + no_of_unique_words_in_train))
                    else:
                        probability_of_ham = probability_of_ham + math.log10((1) / (total_ham_words + no_of_unique_words_in_train))

                    if word in no_of_unique_spam_words_in_spam:
                        probability_of_spam = probability_of_spam + math.log10((no_of_unique_spam_words_in_spam[word] + 1) / (total_spam_words + no_of_unique_words_in_train))
                    else:
                        probability_of_spam = probability_of_spam + math.log10((1) / (total_spam_words + no_of_unique_words_in_train))

            probability_of_ham = probability_of_ham_class + probability_of_ham
            probability_of_spam = probability_of_spam_class + probability_of_spam

        if (probability_of_ham > probability_of_spam):
            correct_guesses_during_train = correct_guesses_during_train + 1

    path_for_spam_train = x
    train_file_count += len(path_for_spam_train)

    for file in path_for_spam_train:
        probability_of_ham = 0
        probability_of_spam = 0
        with open(file, 'r', encoding='latin1') as fp:
            # Reading each line in the file
            for line in fp:
                # tokenizing
                tokens = tokenizer.tokenize(line)
                stemmed_words = [stemmer.stem(str(t)) for t in tokens]
                for word in stemmed_words:
                    if word in no_of_unique_ham_words_in_ham:
                        probability_of_ham = probability_of_ham + math.log10((no_of_unique_ham_words_in_ham[word] + 1) / (total_ham_words + no_of_unique_words_in_train))
                    else:
                        probability_of_ham = probability_of_ham + math.log10((1) / (total_ham_words + no_of_unique_words_in_train))

                    if word in no_of_unique_spam_words_in_spam:
                        probability_of_spam = probability_of_spam + math.log10(
                            (no_of_unique_spam_words_in_spam[word] + 1) / (total_spam_words + no_of_unique_words_in_train))
                    else:
                        probability_of_spam = probability_of_spam + math.log10((1) / (total_spam_words + no_of_unique_words_in_train))

            probability_of_ham = probability_of_ham_class + probability_of_ham
            probability_of_spam = probability_of_spam_class + probability_of_spam

        if probability_of_ham < probability_of_spam:
            correct_guesses_during_train = correct_guesses_during_train + 1

    training_accuracy = 0
    training_accuracy = (correct_guesses_during_train / train_file_count) * 100


   # print(ham_test)
    correct_guesses_during_test = 0
    test_file_count = len(ham_test)

    for file in ham_test:
        probability_of_ham = 0
        probability_of_spam = 0
        with open(file, 'r', encoding='latin1') as fp:
            for line in fp:
                # tokenizing
                tokens = tokenizer.tokenize(line)
                stemmed_words = [stemmer.stem(str(t)) for t in tokens]
                for word in stemmed_words:
                    if word in no_of_unique_ham_words_in_ham:
                        probability_of_ham = probability_of_ham + math.log10((no_of_unique_ham_words_in_ham[word] + 1) / (total_ham_words + no_of_unique_words_in_train))
                    else:
                        probability_of_ham = probability_of_ham + math.log10((1) / (total_ham_words + no_of_unique_words_in_train))

                    if word in no_of_unique_spam_words_in_spam:
                        probability_of_spam = probability_of_spam + math.log10((no_of_unique_spam_words_in_spam[word] + 1) / (total_spam_words + no_of_unique_words_in_train))
                    else:
                        probability_of_spam = probability_of_spam + math.log10((1) / (total_spam_words + no_of_unique_words_in_train))


        if (probability_of_ham > probability_of_spam):
            correct_guesses_during_test = correct_guesses_during_test + 1


    test_file_count += len(spam_test)

    for file in spam_test:
        probability_of_ham = 0
        probability_of_spam = 0
        with open(file, 'r',encoding="latin1") as fp:
            # Reading each line in the file
            for line in fp:
                # tokenizing
                tokens = tokenizer.tokenize(line)
                stemmed_words = [stemmer.stem(str(t)) for t in tokens]
                for word in stemmed_words:
                    if word in no_of_unique_ham_words_in_ham:
                        probability_of_ham = probability_of_ham + math.log10(
                            (no_of_unique_ham_words_in_ham[word] + 1) / (total_ham_words + no_of_unique_words_in_train))
                    else:
                        probability_of_ham = probability_of_ham + math.log10((1) / (total_ham_words + no_of_unique_words_in_train))

                    if word in no_of_unique_spam_words_in_spam:
                        probability_of_spam = probability_of_spam + math.log10((no_of_unique_spam_words_in_spam[word] + 1) / (total_spam_words + no_of_unique_words_in_train))
                    else:
                        probability_of_spam = probability_of_spam + math.log10( (1) / (total_spam_words + no_of_unique_words_in_train))


        if probability_of_ham < probability_of_spam:
            correct_guesses_during_test = correct_guesses_during_test + 1

    test_accuracy_with_stopwords = 0

    test_accuracy_with_stopwords = (correct_guesses_during_test / test_file_count) * 100

    return training_accuracy, test_accuracy_with_stopwords

# Naive bayes after filtering out stop words
def naive_bayes_without_stop_words():



    stop_words = set(stopwords.words("english"))

    total_no_of_words_in_train = 0
    bag_of_words = {}


    for file in training_files:
        with open(file, 'r', encoding='latin1') as fp:
            for line in fp:
                tokens = tokenizer.tokenize(line)

                stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                total_no_of_words_in_train = total_no_of_words_in_train + len(stemmed_words)
                for word in stemmed_words:

                        if word.lower() in bag_of_words:
                            bag_of_words[word.lower()] += 1
                        else:
                            bag_of_words[word.lower()] = 1

    no_of_unique_words_in_train = len(bag_of_words)


    no_of_unique_ham_words_in_ham = {}
    total_ham_words = 0

    for file in ham_files:
        with open(file, 'r', encoding='latin1') as fp:
            # Reading each line of file
            for line in fp:
                # tokenizing
                tokens = tokenizer.tokenize(line)

                stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                total_ham_words += len(stemmed_words)
                # Once words are stemmed, selecting only the unique words from entire bag of words
                for word in stemmed_words:

                        if word.lower() in no_of_unique_ham_words_in_ham:
                            no_of_unique_ham_words_in_ham[word.lower()] += 1
                        else:
                            no_of_unique_ham_words_in_ham[word.lower()] = 1

    no_of_unique_spam_words_in_spam = {}
    total_spam_words = 0

    for file in spam_files:
        with open(file, 'r', encoding='latin1') as fp:
            for line in fp:
                tokens = tokenizer.tokenize(line)

                stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                total_spam_words += len(stemmed_words)
                for word in stemmed_words:

                        if word.lower() in no_of_unique_spam_words_in_spam:
                            no_of_unique_spam_words_in_spam[word.lower()] += 1
                        else:
                            no_of_unique_spam_words_in_spam[word.lower()] = 1

    probability_of_ham_class = math.log10(count_ham_files / total_no_of_files)
    probability_of_spam_class = math.log10(count_spam_files / total_no_of_files)


    correct_guesses = 0
    test_file_count = len(ham_test)

    for file in ham_test:
        probability_of_ham = 0
        probability_of_spam = 0
        with open(file, 'r', encoding='latin1') as fp:
            for line in fp:
                # tokenizing
                tokens = tokenizer.tokenize(line)
                # Filtering out stopwords along with stemming
                stemmed = [stemmer.stem(str(t)) for t in tokens if not t in stop_words]
                for word in stemmed:
                    if word in no_of_unique_ham_words_in_ham:
                        probability_of_ham = probability_of_ham + math.log10((no_of_unique_ham_words_in_ham[word] + 1) / (total_ham_words + no_of_unique_words_in_train))
                    else:
                        probability_of_ham = probability_of_ham + math.log10((1) / (total_ham_words + no_of_unique_words_in_train))

                    if word in no_of_unique_spam_words_in_spam:
                        probability_of_spam = probability_of_spam + math.log10((no_of_unique_spam_words_in_spam[word] + 1) / (total_spam_words + no_of_unique_words_in_train))
                    else:
                        probability_of_spam = probability_of_spam + math.log10((1) / (total_spam_words + no_of_unique_words_in_train))

            probability_of_ham = probability_of_ham_class + probability_of_ham
            probability_of_spam = probability_of_spam_class + probability_of_spam

        if (probability_of_ham > probability_of_spam):
            correct_guesses = correct_guesses + 1


    test_file_count += len(spam_test)

    for file in spam_test:
        probability_of_ham = 0
        probability_of_spam = 0
        with open(file, 'r', encoding="latin1") as fp:
            for line in fp:
                tokens = tokenizer.tokenize(line)
                stemmed = [stemmer.stem(str(t)) for t in tokens if not t in stop_words]
                for word in stemmed:
                    if word in no_of_unique_ham_words_in_ham:
                        probability_of_ham = probability_of_ham + math.log10(
                            (no_of_unique_ham_words_in_ham[word] + 1) / (total_ham_words + no_of_unique_words_in_train))
                    else:
                        probability_of_ham = probability_of_ham + math.log10((1) / (total_ham_words + no_of_unique_words_in_train))

                    if word in no_of_unique_spam_words_in_spam:
                        probability_of_spam = probability_of_spam + math.log10((no_of_unique_spam_words_in_spam[word] + 1) / (total_spam_words + no_of_unique_words_in_train))
                    else:
                        probability_of_spam = probability_of_spam + math.log10((1) / (total_spam_words + no_of_unique_words_in_train))

            probability_of_ham = probability_of_ham_class + probability_of_ham
            probability_of_spam = probability_of_spam_class + probability_of_spam

        if probability_of_ham < probability_of_spam:
            correct_guesses = correct_guesses + 1

    test_accuracy_without_stopwords = 0
    test_accuracy_without_stopwords = (correct_guesses / test_file_count) * 100

    return test_accuracy_without_stopwords


training_accuracy, test_accuracy_with_stopwords = naive_bayes_with_stop_words()

test_accuracy_without_stopwords = naive_bayes_without_stop_words()

print("\n Training    Test_with_stopwords    Test_without_stopwords")
print(' %.2f              %.2f	            	    %.2f' % (training_accuracy, test_accuracy_with_stopwords, test_accuracy_without_stopwords))

