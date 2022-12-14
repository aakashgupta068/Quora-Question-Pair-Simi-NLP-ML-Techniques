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
