
class BaseInference:
    def load_model(self):
        """ Load pretrain model """
        pass
    
    def inference(self):
        """ Execute the main process """
        pass
    
    def batch_inference(self):
        """ Execute the main process by batch """
        pass
    
    def preprocess(self):
        """ Pre-processing image/embedding """
        pass
    
    def posprocess(self):
        """ Post-processing image/embedding """
        pass