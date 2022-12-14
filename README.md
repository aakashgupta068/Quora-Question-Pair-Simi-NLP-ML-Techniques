# Quora-Question-Pair Similarity-NLP-Machine Learning-Techniques
The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. Suppose you ask a question on Quora, and get some suggested answers. These answers corrosponds to questions which are similar to the one you have asked.

#### My Opinion:
- If you get suggested wrong answers, then it is more damaging to the end-user, than compared to the situation where Quora Model is unable to suggest available answers to already asked similar questions. 

#### Hence, we should penalise the model more if it misclassifies a Question as Duplicate, i.e., False Positives as compared to miss-classified as Not-Duplicate

# Solution & Methodology
<div style="background-color:#F8F6F0;font-size:17px;font-family:Georgia;border-style: solid;border-color: #EDE6D6;border-width:1px;padding:20px;margin: 0px;color:#254E58;overflow:hidden">
    <p style="color:blue"><b>1) I have extracted Features for Similarity Measures Like</b></p> 
    <ul style="font-size:14px">
        <li>Proportion of Common Words, Tokens & stop words between Question Pairs</li>
        <li>TF-IDF vectors of the Question Pairs and cosine similarity between them</li>
        <li>Longest common substring</li>
        <li>Fuzzy Similarity Ratios based on Levenstein Distance</li>
    </ul>
</div>

<div style="background-color:#F8F6F0;font-size:17px;font-family:Georgia;border-style: solid;border-color: #EDE6D6;border-width:1px;padding:20px;margin: 0px;color:#254E58;overflow:hidden">
    <p style="color:blue"><b>2) I have clustered these ~5.3 Lakh Questions into 9 Buckets, and extracted a Feature - Do Both Questions belong in same cluster ? (0 or 1)</b></p> 
    <p>Check out the Cluster Visuals in 2-D, and the WordClouds generated from these Cluster</p>
</div>


<div style="background-color:#F8F6F0;font-size:17px;font-family:Georgia;border-style: solid;border-color: #EDE6D6;border-width:1px;padding:20px;margin: 0px;color:#254E58;overflow:hidden">
    <p style="color:blue"><b>3) Total 29 Features were generated and fit into Logistic Regression, Naive Bayes, KNN(k=5) & Random Forest Classifier.</b></p> 
    <ul style="font-size:14px">
        <li>RF gave the highest Accuracy 0.82, followed by Log-Reg & KNN (~0.77). Log-loss from RF was the lowest. So, in final model prediction, we have used RF.</li>
        <li>Not used Logistic Regression, since Log-Odds may not be Linearly Related to Independent variables, and we may not be able to create a Odd-Ratio Table</li>
    </ul>
</div>

<div style="background-color:#F8F6F0;font-size:17px;font-family:Georgia;border-style: solid;border-color: #EDE6D6;border-width:1px;padding:20px;margin: 0px;color:#254E58;overflow:hidden">
    <p style="color:blue"><b>4) Log-Loss from RF was the lowest (0.17) in Train Data, which spiked to 0.72 in Test Submission. This means, distribution of data changed drastically from Train to Test</b></p> 
</div>

# Key-Insights

- Percentage of Questions with '?' (not necessarily ending with): 99.864 %
- Number of Math based questions:  0.138647 %

**Questions with multiple parts**: 5.542 %

**Examples of Questions with subparts** :
- 'How safe is it to take 90 mg of codeine? What are the health concerns?', 
- 'Which is the best 50" tv to buy in India? Very confused with sony Samsung or lg? And confused with 4k n full hd.!'

**Personal or Opinion-based questions**: 12.038 %

**Examples of some Personal Or Opinion-based questions**:  
- 'Who do you think is the biggest actor in Bollywood?', 
- 'What things you can do on the internet to make money?'

# Contents

Table of Content
<br><a href="#1">1. Data Definations</a>
<br><a href="#2">2. Is data imbalanced</a>
<br><a href="#3">3. Checking Missing values</a>
<br><a href="#4">4. Check number of unique questions and duplicates in whole corpus</a>
<br><a href="#5">5. Distribution of repeated Questions</a>
<br><a href="#6">6. Text Pre-processing</a>
<br><a href="#7">7. Clustering of Questions</a>
<br><a href="#8">8. Word-Clouds of different Questions Cluster</a>
<br><a href="#9">9. Basic Feature Extraction</a>
<br><a href="#10">10. Advanced Text Feature Extraction</a>
<br><a href="#11">11. Fuzzy Text Similarity Feature Extraction</a></a>
<br><a href="#12">12. Finding Cosine Similarity from TF-IDF Text Vectors</a>
<br><a href="#13">13. Model Fitting</a>
<br><a href="#14">14. Logistic Model Performance</a></a>
<br><a href="#15">15. Different Classification Model Performance Comparison</a>
<br><a href="#16">16. Varying threshold to minimise False Duplicates</a>
<br><a href="#17">17. K-Fold Cross Validation</a></a>
<br><a href="#18">18. Confidence Interval & p-value of Coefficients of Logistic Regression</a></a></a>
<br><a href="#19">19. Importing Test data</a></a>
<br><a href="#20">20. Missing values in Test Dataset</a></a>
<br><a href="#21">21. Analysis of Unique & Repeated Questions in Test Set</a></a>
<br><a href="#22">22. Test Data: Pre-processing of Text</a></a>
<br><a href="#23">23. Clustering of Questions</a></a>
<br><a href="#24">24. Basic Text-Feature Engineering</a></a>
<br><a href="#25">25. Advanced Text-Feature Engineering in Test Data</a></a>
<br><a href="#26">26. Fuzzy Text-Similarity Ratios Extracted from Test Data</a></a>
<br><a href="#27">27. Cosine Similarity Extracted for Test data</a>
<br><a href="#28">28. Training Random Forrest Classifier Model on entire Train dataset</a>
<br><a href="#29">29. Prediction on Test set & Submission</a>

## <a id="1">1. Dataset Description</a>

Data fields
* id - the id of a training set question pair
* qid1, qid2 - unique ids of each question (only available in train.csv)
* question1, question2 - the full text of each question
* is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise

## <a id="2">2. Is data imbalanced ?</a>

<iframe src="https://www.kaggle.com/embed/newbieag068/quora-questions-eda-tfidf-similarity?cellIds=14&kernelSessionId=113799250" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="Quora Questions EDA TfIdf Similarity"></iframe>

<img src="https://www.kaggle.com/code/newbieag068/quora-questions-eda-tfidf-similarity?scriptVersionId=113799250&cellId=13.png" width="400px" height="200px">

We have 63% of Non-duplicates, 37% of duplicates, signifiy slighlty imbalanced dataset


## <a id="3">3. Checking Missing values</a>
Question2 with qid2= 174364 is the only missing among Question2 with no other occurances
Question1 with qid1= 493340 is the only missing among Question1 with no other occurances

**Conclusion**
* Since, we can't find any occurances of these missing questions, bettter to drop these rows
* Only 3 rows have a missing question overall (out of 404290 Question Pairs). Hence we can delete these 3 missing Question Pairs
Dropping these 3 rows with missing Question Pairs

## <a id="4">4. Check number of unique questions and duplicates in whole corpus</a>

## <a id="5">5. Distribution of repeated Questions</a>

## <a id="6">6. Text Pre-processing</a>

## <a id="7">7. Clustering of Questions</a>

## <a id="7">7. Clustering of Questions</a>

**Conclusion**
Here, we see that, at Cluster number=5, there is a slight dip, post which Sum of Sq Distance dips linearly.
So, we take optimum cluster number as 5 & map the cluster to the respective question

## <a id="8">8. Major words in each word cloud</a>
Using Optimum Cluster Number as 9. 

## <a id="9">9. Basic Feature Extraction</a>

Let us now contruct few features like:

* Same Cluster: 1 if both Question-1 & Question-2 belongs to same cluster, else 0
* freq_qid1: Frequency of qid1's
* freq_qid2: Frequency of qid2's
* q1len: Length of q1
* q2len: Length of q2
* q1_n_words: Number of words in Questions 1
* q2_n_words: Number of words in Questions 2
* word_Common: (Number of common unique words in Question 1 and Questions 2)
* word_Total: Total num of words in Question1 + Total num of words in Question 2
* **Dice’s coefficient** word_share : 2*(word_common)/(word_Total)
* freq_q1 + freq_q2: sum total of frequency of qid1 and qid2
* freq_q1 - freq_q2: absolute difference of frequency of qid1 and qid2

## <a id="10">10. Advanced Text Feature Extraction</a>
Definition:
- __Token__: You get a token by splitting sentence a space
- __Stop_Word__ : stop words as per NLTK.
- __Word__ : A token that is not a stop_word

Features:
- __cwc_min__ :  Ratio of common_word_count to min lenghth of word count of Q1 and Q2 <br>cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
- __cwc_max__ :  Ratio of common_word_count to max lenghth of word count of Q1 and Q2 <br>cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
- __csc_min__ :  Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2 <br> csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
- __csc_max__ :  Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2<br>csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
- __ctc_min__ :  Ratio of common_token_count to min lenghth of token count of Q1 and Q2<br>ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
- __ctc_max__ :  Ratio of common_token_count to max lenghth of token count of Q1 and Q2<br>ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
- __last_word_eq__ :  Check if last word of both questions is equal or not<br>last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
- __first_word_eq__ :  Check if First word of both questions is equal or not<br>first_word_eq = int(q1_tokens[0] == q2_tokens[0])
- __abs_len_diff__ :  Abs. length difference<br>abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
- __mean_len__ :  Average Token Length of both Questions<br>mean_len = (len(q1_tokens) + len(q2_tokens))/2
- __longest_substr_ratio__ :  Ratio of length longest common substring to min lenghth of token count of Q1 and Q2<br>longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))

## <a id="11">11. Fuzzy Text Similarity Feature Extraction</a>
#### String Similarity Measures:
* Fuzz Ratio
* fuzz.partial_ratio
* fuzz.token_sort_ratio
* fuzz.token_set_ratio
* fuzz.WRatio

## <a id="12">12. Finding Cosine Similarity from TF-IDF Text Vectors</a>
<img src="https://www.machinelearningplus.com/wp-content/uploads/2018/10/soft-cosine.png" width="400px" height="200px">

* Convert each Pre-processed Question1 and Question2 text into TF-IDF Weighted Vectors, then do Vector Dot-Product on those two vectors to get the cosine similarity.
* Here, we are using vectors where each single word represents a dimension and has a **value = TF * IDF**

<img src="https://miro.medium.com/max/1200/1*qQgnyPLDIkUmeZKN2_ZWbQ.png" width="500px" height="400px">

## <a id="13">13. Model Fitting</a>

## <a id="14">14. Logistic Model Performance</a>

## <a id="15">15. Different Classification Model Performance Comparison</a>

## <a id="16">16. Varying threshold to minimise False Duplicates</a>
Which error- FPR or FNR is important for us ?

When searching for duplicates, if the model predict a wrong question as duplicate, then that will be more costly rather than predicting a true duplicate question as non-duplicate

So,considering **Duplicate='Yes'** as Positive Class & **Duplicate='No'** as Negative Class, 
* **FPR** = True Non-Duplicate Questions that were predicted Duplicate by model
* **FNR** = True Duplicate Questions that were predicted Non-Duplicate by model
* **TPR** = True Duplicate Questions that were predicted Duplicate by Model
* **TNR** = True Non-Duplicate Questions that were predicted Non-Duplicate by model

## <a id="17">17. K-Fold Cross Validation</a>

## <a id="18">18. Confidence Interval & p-value of Coefficients of Logistic Regression</a>

## <a id="19">19. Importing Test data</a>

## <a id="20">20. Missing values in Test Dataset</a>
Imputing Nan values using "Unknown"

## <a id="21">21. Analysis of Unique & Repeated Questions in Test Set</a>

## <a id="22">22. Test Data: Pre-processing of Text</a>

## <a id="23">23. Clustering of Questions</a>

## <a id="24">24. Basic Text-Feature Engineering</a>

## <a id="25">25. Advanced Text-Feature Engineering in Test Data</a>

## <a id="26">26. Fuzzy Text-Similarity Ratios Extracted from Test Data</a>

## <a id="27">27. Cosine Similarity Extracted for Test data</a>

## <a id="28">28. Training Random Forrest Classifier Model on entire Train dataset</a>

## <a id="29">29. Prediction on Test set & Submission</a>
