---
layout: default
---

Welcome to my personal website! Shoot me email _jchen1105@hotmail.com_ if you want any of my source codes shared on my site. You can also visit [my LinkedIn](https://www.linkedin.com/in/jiaming-chen-data-analyst/) page.


# [](#header-6)Music Player Churn Prediction & Recommendation System
* Data set: 14.19 GB play log file (named by the date of creation)
* Possible tools for processing: PySpark, Pandas, MLlib, sklearn etc.
* Update: added date info to the log file, concat all log files into one file, load all_play_log into a spark session:
```python
def parseLine(line):
    fields = line.split('\t')
    if len(fields) == 10:
        uid = int(fields[0])
        device = str(fields[1])
        song_id = int(fields[2])
        song_type = int(fields[3])
        song_name = str(fields[4])
        singer = str(fields[5])
        play_time = int(fields[6])
        song_length = int(fields[7])
        paid_flag = int(fields[8])
        fn = str(fields[9])
        return Row(uid, device, song_id, song_type, song_name, singer, play_time, song_length, paid_flag, fn)
    else:
        return Row(None)
        
songs = lines.map(parseLine).filter(lambda x: len(x) == len(schema))

songDataset = spark.createDataFrame(songs).cache()
songDataset.show()

songDataset.groupBy('uid').count().orderBy('count', ascending = False).show(truncate=False)
```


# [](#header-5)Yelp dataset challenge: NLP & sentiment analysis
This project can be divided into 5 stages:
* [Data Preprocessing](https://github.com/JMistral/Yelp_Challenge/blob/master/Yelp_Dataset_-_Data_Preprocessing.ipynb)
* [Natural Language Processing](https://github.com/JMistral/Yelp_Challenge/blob/master/Yelp_Dataset_-_NLP.ipynb):
* [Clustering and PCA](https://github.com/JMistral/Yelp_Challenge/blob/master/Yelp_Dataset_-_Clustering_and_PCA.ipynb):
Let's take a look at the most frequent words for each cluster of reviews:
```python
from wordcloud import WordCloud
fig, ax = plt.subplots(kmeans.n_clusters,1, figsize=(1*10,kmeans.n_clusters*5))
for i in range(kmeans.n_clusters):
    wordcloud = WordCloud().generate(cluster_txt[i])
    ax[i].imshow(wordcloud, interpolation='bilinear')
    ax[i].axis("off")
    ax[i].set_title('cluster#'+str(i)+" review word clouds")
plt.savefig("word_clouds"+'_'+str(kmeans.n_clusters) + "clusters")
```
![wordcloud](/images/cluster3word.png)
_Note: "Winner winner chicken dinner" the reviews in this cluster must come from some chicken lovers_
* [Restaurant Recommender](https://github.com/JMistral/Yelp_Challenge/blob/master/Yelp_Dataset_-_Restaurant_Recommender.ipynb)



# [](#header-1)Data Scientist Internship (June 2017 ~ August 2017)
This internship with a startup company gave me experience with using AWS and Python for data wrangling, data visualization, clustering,and also **data story telling**. Here are some results

1. Data Visualization(heatmap):
Given a log file of one electrical vehicle's battery State of Charge (**SOC**) over one month, how would you explore the driver's behavior?
![ev_behave](/images/ev_behave.png)

_Note: the color bar stands for the battery status (in %), and we can clearly see the pattern of this driver:_
 * Start his/her day at around 7 a.m.
 * Finish his/her day at 12 p.m.
 
 
2. Feature Engineering:
By taking the derivative of battery usage, we generated a new categorical feature: _Status_, with three categories: _driving, parked, charging_. Using the same heatmap visualization, we grouped the status by day and time. Here is one of the results:
![ev_status](/images/ev_status.png)






# [](#header-2)Statistical Computing Using R
In this course we follow the book [_Introduction to Statistic Learning_ ](http://www-bcf.usc.edu/~gareth/ISL/). A great book for learning R and basic EDA skills as well. And most importantly, our professor Dr.Sonderegger shares **lots of** wonderful and well explained R source code on [his personal page](https://dereksonderegger.github.io/578/) 👏

Here's my homework code:

> [Final Exam: bias-variance tradeoff, interpretability flexibility tradeoff AND model selection using cross validation](Final)

> [exam 2: feature engineering and feature selection/regularization using lasso and ridge](exam2.pdf)

> [Choosing degree of freedom? (examples with KNN regression and linear regression)](HW4_JCHEN)

> [Resampling(LOOCV, bootstrapping, K-fold CrossValidation)](HW5_STA578)

> [Lasso and Ridge Regression](HW6_STA578)

> [Smoothing Splines(with lidar data)](HW7_STA578.pdf)




# [](#header-3)Data Analyst with Udacity
[my GitHub repo](https://github.com/JMistral/DataAnalyst_Udacity) for this program

_new update!_ I just got some cool data from this [website](http://www.gapminder.org/data/) about sugar and food consumption all over the world.

It's really exciting to explore some new sweet secrets!

Which country is the most rich one regarding food available per capita per day in the year 1961?

**Switzerland!**, it has on average 3500 kcal food available per person in 1961, it's almost the same level as Canada in the year of 2001.

[R sourcecode](sweetie)

I will keep updating this project as I move on.



# [](#header-4)Image Processing Using MATLAB and Python

We're using the famous _Digital Image Processing (Gonzalez 4th edition)_ for this course. Some of my homework code in MATLAB can be found in [my GitHub repo](https://github.com/JMistral/ImageProcessingEE542).

>_Update_: Kaggle Iceberg Challenge: RGB component extraction, Image Augmentation, and Convolutional Neural Network! (I didn't get a very high rank in the Kaggle Challenge, but it was a fun project anyway!)[_link_](https://github.com/JMistral/Iceberg)

> Have you tried to apply **Butterworth Filter** in spatial domain? It's usually applied in spatial frequency domain, but I tried to use inverse FFT to find the spatial convolution kernel of Butterworth and then convolute image and the filter kernel in spatial domain. The code can be found [here](https://github.com/JMistral/ImageProcessingEE542/blob/master/HW6main.m). Moving average's frequency response is also included.

>  **Laplacian of Gaussian (LoG)**, examples of different kernel size. [MATLAB code](https://github.com/JMistral/ImageProcessingEE542/blob/master/HW5main.m)


> Kaggle Project: Image Augmentation and Deep Learning. Some useful tutorial can be found [here](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/)

