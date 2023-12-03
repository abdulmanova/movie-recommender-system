# Introduction
In this assignment, we delve into the creation of a movie recommender system using the MovieLens 100K dataset. 
The objective is to leverage machine learning to offer personalized movie suggestions based on demographic information such as age, gender, occupation, as well as users' rated movies. 
The report navigates through data exploration, model implementation, and benchmarking processes to evaluate the system's recommendation quality. 
Ultimately, the goal is to construct an effective recommender system, shedding light on the complexities and insights involved in tailoring 
recommendations to individual user preferences.

# Data analysis

Rating Distribution: The uneven distribution of ratings implies that the dataset may have a bias towards positive ratings (5) and less representation of neutral or negative sentiments. To address this, stratified sampling during the training and evaluation phases can ensure that each rating category is adequately represented.

Age Distribution: The non-normal distribution of ages suggests a demographic skew in the dataset. This could impact the model's ability to generalize across age groups. Applying feature scaling or normalizing age values could help mitigate this issue.

Movie Genre Distribution: The skewed distribution of movie genres, with drama dominating, may result in a bias towards this genre in the recommendations. Consideration should be given to strategies like oversampling underrepresented genres or using techniques like SMOTE to balance the genre distribution.

Average Ratings by Gender and Age Group: The equality in average ratings across gender and age groups might indicate a lack of discriminatory power in these features for predicting preferences. Feature engineering or interaction terms could be explored to enhance the model's ability to capture nuanced preferences.

Movie Preferences by Gender: While a correlation between gender and genre preferences is observed, the potential bias due to the dominance of drama needs careful consideration. Techniques like oversampling or generating synthetic samples for underrepresented genres could help balance the dataset.

Movie Release Year Distribution: The right-skewed distribution of movie release years is logical but could potentially introduce a time-related bias. Techniques like temporal splitting of the dataset or applying time-based weighting during training could be considered to account for temporal shifts in user preferences.

Suggested Preprocessing Techniques:

Stratified Sampling: Ensure balanced representation of different rating categories during training and evaluation.
Feature Scaling/Normalization: Normalize age values to address the skewed age distribution.
Data Balancing Techniques: Apply oversampling or synthetic sample generation techniques to balance the distribution of movie genres.

Feature exclusion:
The exclusion of the zip_code column is justified due to its categorical nature, representing the user's location, which is deemed improbable to exert a substantial influence on the movie recommendation process. The incorporation of such a variable may introduce superfluous intricacies and noise to the model, yielding a heightened level of complexity without a commensurate enhancement in predictive efficacy.

Similarly, the removal of the movie_title column, functioning as a unique identifier for each movie, is grounded in the consideration that, despite its utility in human interpretation, its inclusion might not significantly contribute to the machine learning model's discernment of user preferences. The model is directed towards more informative features, encompassing genres, ratings, and user demographics, for its training regimen.

Moreover,  the inclusion of external links ('IMDb_URL' column) holds limited relevance to the recommendation process, as the core factors influencing recommendations lie in user demographics and movie preferences. Moreover, the dataset's size and complexity may render non-essential columns burdensome, with simpler models often proving more effective and interpretable.

In adapting the dataset for recommender system implementation, the focus shifts from predicting explicit movie ratings to forecasting implicit user interactions. The dataset is binarized, categorizing ratings surpassing a specified threshold (e.g., > 3) as positive interactions, and treating the rest, including instances where a user has not interacted with a film, as negative interactions. This modification aligns with the central goal of predicting user engagement with movies, emphasizing the recommendation of films with heightened interaction probabilities. Addressing the uniform categorization of every sample into the positive class post-binarization, negative samples are introduced for model training. Negative instances are determined by considering movies a user has not interacted with, assuming such movies signify a lack of user interest. The inclusion of negative samples diversifies the training dataset, enhancing the model's ability to discern between positive and negative interactions, with the quantity of introduced negative samples contingent upon the absence of user interactions with specific movies, ensuring a comprehensive representation of potential disinterest.

# Model Implementation

Chosen model, Neural Collaborative Filtering (NCF), offers a streamlined and comprehensible architecture suitable for tutorial purposes. Embarking on the understanding of embeddings is pivotal for delving into the model's intricacies. Embeddings, in essence, represent users in a lower-dimensional space, capturing nuanced relationships from a higher-dimensional space. Illustratively, users with akin preferences are proximate in this embedding space. The dimensions in this representation signify traits, and we opt for an 8-dimensional embedding for user and item representations, striking a balance between precision and model complexity. The learning of these embeddings is facilitated through Collaborative Filtering, leveraging the ratings dataset to discern similarities among users and movies. Transitioning to the model architecture, our inputs comprise one-hot encoded user and item vectors for a given interaction. These vectors undergo embedding layers, resulting in condensed representations. Concatenating these embeddings, subsequent fully connected layers map them to a prediction vector. The Sigmoid function is then applied for binary classification, determining the probability of interaction. In sum, our NCF model adeptly navigates the complexities of collaborative filtering, providing a robust foundation for recommendation systems.

# Model Advantages and Disadvantages

### Advantages:

- Expressiveness: NCF, being a deep learning-based model, exhibits high expressiveness in capturing complex patterns and non-linear relationships within user-item interactions. This enables the model to discern intricate user preferences.

- Embedding Learning: The model autonomously learns user and item embeddings, mitigating the need for manual feature engineering. Through collaborative filtering, it gleans latent features, contributing to a more nuanced and data-driven representation.

- Scalability: NCF is scalable to large datasets, allowing it to effectively handle extensive user-item interactions. This scalability is crucial for real-world applications with substantial amounts of data.

- Flexibility: The model can accommodate different types of interactions and feedback, making it versatile for various recommendation scenarios. It can seamlessly adapt to implicit feedback and binary interactions.

### Disadvantages:

- Cold Start Problem: NCF faces challenges when dealing with new users or items (cold start problem) since it relies on historical interactions for learning embeddings. Recommending to users with limited or no historical data can be less accurate.

- Data Sparsity: In scenarios where user-item interactions are sparse, the model may struggle to learn accurate embeddings, leading to potential information loss. This is a common challenge in collaborative filtering-based models.

- Interpretability: Deep learning models, including NCF, are often considered as "black-box" models, making it challenging to interpret how the model arrives at specific recommendations. Interpretability is crucial in certain domains for user trust and understanding.

- Computational Intensity: Training deep learning models, especially on large datasets, can be computationally intensive and may require substantial resources. This can be a limitation in resource-constrained environments.

- Dependency on Ratings: NCF relies heavily on user ratings for training. In scenarios where explicit ratings are not available or unreliable, the model's effectiveness may diminish.

- Balancing these advantages and disadvantages is essential when considering the adoption of NCF in a recommendation system, taking into account the specific characteristics and requirements of the application.

# Training Process

Initialization:

The model is initialized with two embedding layers for users and items, each with a specified number of embeddings (num_users and num_items) and embedding dimensions (8 in this case).
Two fully connected layers (fc1 and fc2) are added, along with the output layer for prediction.
Forward Pass:

During a forward pass, the user_input and item_input are passed through their respective embedding layers (user_embedding and item_embedding).
The resulting embeddings are concatenated to form a vector representation of the user-item interaction.
Dense Layers:

The concatenated vector is then passed through two fully connected (dense) layers (fc1 and fc2), each followed by a ReLU activation function. These layers capture non-linear relationships and learn complex patterns from the input embeddings.
Output Layer:

The output of the second dense layer is passed through the output layer, which produces a single value for prediction.
A Sigmoid activation function is applied to squash the output between 0 and 1, representing the probability of a positive interaction.
Loss Calculation:

In the training_step method, the predicted_labels are compared with the actual labels (interacted or not) from the training batch.
The binary cross-entropy loss (BCELoss) is computed, measuring the dissimilarity between predicted and true labels. This loss quantifies how well the model is performing on the training data.
Optimization:

The configure_optimizers method specifies the optimization algorithm, in this case, Adam optimizer, which adjusts the model's parameters to minimize the computed loss.
Data Loading:

The train_dataloader method defines how the training data is loaded into the model. It utilizes a DataLoader with a specified batch size and number of workers.
Training Loop:

The model is then trained through multiple epochs, where each epoch involves iterating through batches of training data.
The optimizer adjusts the model parameters based on the computed loss, gradually improving the model's ability to make accurate predictions.
This training process iterates until a predefined number of epochs are completed, resulting in a trained NCF model ready for making movie recommendations.

# Evaluation

Certainly, let's break down the evaluation approach for the test loop:

Initialization:

The test_user_item_set contains pairs of user-item interactions from the test set.
user_interacted_items is a dictionary mapping each user to a list of items they have interacted with in the training data.
Evaluation Loop:

For each (u, i) pair in the test set:
interacted_items is the list of items that the user (u) has interacted with in the training data.
not_interacted_items is the set of all items subtracted by the interacted_items.
selected_not_interacted randomly selects 99 items from the not_interacted_items set.
test_items is created by combining the selected_not_interacted items with the current test item (i).
Prediction:

The model is then used to predict the likelihood of interaction for the user (u) with the test_items.
predicted_labels contain the predicted probabilities of interaction for each item in the test_items.
Top-10 Recommendations:

The top 10 items with the highest predicted probabilities are selected from the test_items.
top10_items contain the recommended items for the user (u).
Hit Ratio Calculation:

The Hit Ratio @ 10 is calculated by checking if the actual test item (i) is present in the top10_items.
If the actual test item is in the top 10 recommendations, a hit is recorded (1), otherwise, a miss is recorded (0).
Final Hit Ratio Calculation:

The average hit ratio is calculated by taking the mean of all the recorded hits.
The final result is printed, indicating the percentage of users for whom the actual test item was among the top 10 recommended items.
This approach evaluates the model's performance by assessing how often the actual test item appears in the top 10 recommendations, providing insights into the effectiveness of the recommendation system.

# Results

The Hit Ratio @ 10 of 0.37 indicates that, on average, approximately 37% of the recommended items are among the top 10 items that the user actually interacted with. Here are some conclusions and results based on this metric:

Model Performance: The Hit Ratio @ 10 serves as a measure of how well the recommendation model is able to identify items that a user is likely to interact with. In this case, achieving a Hit Ratio of 0.37 suggests a moderate level of success.

Room for Improvement: While a Hit Ratio of 0.37 is a positive outcome, there is room for improvement. Enhancing the model's performance could involve fine-tuning hyperparameters, experimenting with different architectures, or considering more advanced recommendation algorithms.