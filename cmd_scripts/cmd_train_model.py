import argparse
from model_train.training_model import ModelTrain


class ModelTrainerCommand:
    """Class to execute training of model based on provided parameters."""

    def __init__(self):
        self._parser = argparse.ArgumentParser(prog = 'model_training',
                                               usage = '%(prog)s [options]')
        self.install_flags()

    def execute(self, arg_list=None):
        print('9087988685')
        args = self._parser.parse_args(arg_list)

        model_trainer = ModelTrain(args.data_filepath,
                                   args.model_type,
                                   args.output_filepath,
                                   args.run_id)
        
        print('Here we are !')
        model_trainer.execute()

    def install_flags(self):
        """Setup command line arguments."""
        self._parser.add_argument('--data-filepath',
                                  type=str,
                                  action='store',
                                  required=True,
                                  help='Provide path of training data')
        
        self._parser.add_argument('--model-type',
                                  type=str,
                                  choices=['bow','tfidf'],
                                  action='store',
                                  required=True,
                                  help='Type of model to be trained')
        
        self._parser.add_argument('--output-filepath',
                                  type=str,
                                  action='store',
                                  required=True,
                                  help='Provide path of output')
        
        self._parser.add_argument('--run-id',
                                  type=str,
                                  action='store',
                                  required=False,
                                  help='Optional run id')
        
if __name__ == "__main__":

    mtc = ModelTrainerCommand()
    mtc.execute()