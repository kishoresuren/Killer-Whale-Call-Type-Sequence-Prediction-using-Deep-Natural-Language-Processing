Welcome to the Readme file for Orca Sequence processing!

Note: All parameters are optional unless 'required' is mentioned.

There are 4 main scripts to run :

1. splitdata.py - This script is required to split the input corpus into a training and test set.

Input parameters : a. trainPercentage (Any value between 0 and 1. Default 0.8)
                   b. windowWidth (Specifies the length of the input sequence. Default 5)
		
Outputs : Four numpy arrays - XTrainSet.npy, YTrainSet.npy, XTestSet.npy, YTestSet.npy


2. buildModel.py - This script is used to train one of the four neural networks upto 100 epochs and save the model. It uses the XTrain and YTrain numpy arrays created in splitdata.py. 

Input parameters : a. numClusters (The number of unique sound patterns in the cluster file. Default 21)
                   b. modelType (Can be 'Uni' or 'Bi', for unidirectional and bidirectional respectively. Default 'Bi')
	           c. modelName (Can be 'LSTM' or 'GRU', to specify the RNN type. Default LSTM)
		
Outputs : Model_name.h5 file (Eg:BiGRU.h5 for a bidirectional GRU network), 'orcaTokenizer' tokenizer in pickle format, plot of accuracies vs epochs as a png file.


3. predictOrcaSequence - This script is to predict an 'n' length orca sequence from the model built in buildModel.py. XTest created in splitdata.py is used as the input to predict 'n' length sequences.

Input parameters:  a. modelType (Can be 'Uni' or 'Bi', for unidirectional and bidirectional respectively. Default 'Bi')
		   b. modelName (Can be 'LSTM' or 'GRU', to specify the RNN type. Default LSTM)
	           c. numGenWords (Can be any valid integer 'n')
				 
Output : yGuess.npy file, containing the predicted sequences of length 'n'.


4. fetchInformationFromSequence.py - This script is used to input a sequence and fetch all/specific information about the occurrence of this sequence like the year, tape and timestamp of occurrences.

Input parameters : a. inputSequence (Comma separated sequence of valid sound patterns, such as '5,4,2'. This is a required parameter.)
                   b. thresholdTime (Time gap between the start of the next sound pattern and end of the previous sound pattern within the input sequence, in seconds. This parameter is not required if the user is interested in seeing all occurrences.)
				   
Output : A text file 'information_<sequence>.txt', containing the desired information about when the input sequence occurred.