import numpy as np
import pickle

class NewsClassifier:
    def __init__(self, pkl_path="pkl_files/", model_path='model.pkl', vectorizer_path='vectorizer.pkl', selection_path='selection.pkl', encoder_path='encoder.pkl'):
        self.model_path = pkl_path + model_path
        self.vectorizer_path = pkl_path + vectorizer_path
        self.selection_path = pkl_path + selection_path
        self.encoder_path = pkl_path + encoder_path
        
    def update_paths(self, model_path=None, vectorizer_path=None, selection_path=None, encoder_path=None):
        if model_path:
            self.model_path = model_path
        if vectorizer_path:
            self.vectorizer_path = vectorizer_path
        if selection_path:
            self.selection_path = selection_path
        if encoder_path:
            self.encoder_path = encoder_path
    
    def load(self):
        self.model = pickle.load(open(self.model_path, 'rb'))
        self.vectorizer = pickle.load(open(self.vectorizer_path, 'rb'))
        self.selection = pickle.load(open(self.selection_path, 'rb'))
        self.encoder = pickle.load(open(self.encoder_path, 'rb'))
        
    def save(self, model_path=None, vectorizer_path=None, selection_path=None, encoder_path=None):
        if not model_path:
            model_path = self.model_path
        if not vectorizer_path:
            vectorizer_path = self.vectorizer_path
        if not selection_path:
            selection_path = self.selection_path
        if not encoder_path:
            encoder_path = self.encoder_path
        
        pickle.dump(self.model, open(model_path, 'wb'))
        pickle.dump(self.vectorizer, open(vectorizer_path, 'wb'))
        pickle.dump(self.selection, open(selection_path, 'wb'))
        pickle.dump(self.encoder, open(encoder_path, 'wb'))
    
    def predict(self, input_iterable):
        X = self.vectorizer.transform(iter(np.array(input_iterable)))
        X = self.selection.transform(X)
        pred = self.model.predict(X)
        return self.encoder.inverse_transform(pred)

if __name__ == '__main__':
    print("Testing News Classifier")
    classifier = NewsClassifier()
    classifier.load()
    input_test = ["Salad's covid relief bill passes the house and heads on to the senate", "Notorious pirate Tristan brown apprehended by police on tuesday morning", "top scientist neil griffith discovers cure for cancer", "Stefan leads the LA lakers with 40 points to win the NBA finals"]
    res = classifier.predict(input_test)

    # res = categorize(loaded_model, vectorizer, selection, encoder, input_test)
    zipped = zip(input_test, res)
    for pair in zipped:
        print(pair)