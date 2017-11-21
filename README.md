# Samsung-SMS-Classification-Part-2

Classfication Classes - Appointment, PickUp, Delivery, Hotel, Payment, Train, Bus, Movie

First I performed feature engineering on the dataset,

-Replaced DATE,TIME,DIGITS,ALPHANUMERNIC and Hyperlink.
-Removed punctuation
-Peformed lemmatization and stemming on the dataset
-Removed Non-English words from the dataset.\
-Peformed Tokenization
-Finally used Support Vector Classifier for modelling on the processed data

I used NLTK and Sklearn packages of Python for this challenge. I got a accuracy of 95.33% on the final test data.
