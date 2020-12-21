from os import path
import shutil
import time
import datetime
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import AutoModel, BertTokenizerFast, AdamW


class BERT_Classifier(nn.Module):
    """Defines extended model architecture"""

    def __init__(self):

        super(BERT_Classifier, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.relu =  nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,16)
        self.fc3 = nn.Linear(16,2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):

        _, cls_hs = self.bert(sent_id, attention_mask=mask)

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x


class Returnability:
    """Use BERT to extract NLP features related to product return"""

    def __init__(self):

        self.curr_path = path.abspath(__file__) # Full path to current class-definition script
        self.root_path = path.dirname(path.dirname(path.dirname(self.curr_path)))

        # Use GPU if available (strongly recommended)
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.device("cuda")
            print("There are %d GPU(s) available" % torch.cuda.device_count())
            print("We will use the GPU:", torch.cuda.get_device_name(0))
        else:
            print("No GPU available, using the CPU instead")
            print("Work may take extremely long time; consider using GPU")
            self.device = torch.device("cpu")

        self.model = BERT_Classifier()
        self.model_loaded = False
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.token_max_length = 50 # Most reviews contain less than 50 tokens
        self.train_ratio = 0.9 # For train/validation split
        self.random_state = 297
        self.batch_size = 32
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=2e-5, # Default is 5e-5
            eps=1e-8, # Default value
        )
        self.epochs = 3 # For model training
        self.train_losses = []
        self.valid_losses = []


    def _preprocess_train_data(self, reviews, labels):
        """
        Args:
            reviews (list[str]): Review texts
            labels (list[bool]): Indicator for whether a review resulted in a product return

        Returns:
            train_dataloader (torch.utils.data.DataLoader): Training data
            val_dataloader (torch.utils.data.DataLoader): Validation data
        """
        # Tokenize all of the sentences and map the tokens to thier word IDs
        encoded_dict = self.tokenizer.batch_encode_plus(
            reviews,
            add_special_tokens=True,
            max_length=self.token_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt', # Return pytorch tensors
        )

        # Get separate components
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        labels = torch.tensor(labels)

        # Combine the training inputs into a TensorDataset
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Split data into train/validation sets
        train_size = int(self.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_state), # For reproducibility
        )

        # Create data loaders for training and validation sets
        train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = self.batch_size,
        )
        val_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = self.batch_size,
        )

        return train_dataloader, val_dataloader


    def _format_time(self, elapsed):
        """Helper function for formatting elapsed times

        Args:
            elapsed (float): Elapsed time in seconds

        Returns:
            (str): Elapsed time in "hh:mm:ss" format
        """
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))


    def _train(self, dataloader):
        """
        Args:
            dataloader (torch.utils.data.DataLoader): Training data

        Returns:
            avg_loss (float): Average training loss
            epoch_time (str): Elapsed time in "hh:mm:ss" format
            total_probs (numpy.array[float]): Predicted probabilities of product return
        """

        print("")
        print("Training...")

        # Set the model to train mode
        self.model.train()

        # Initialize variables to track
        total_loss, total_accuracy = 0, 0

        # Initiate empty list to save model predictions
        total_preds = []

        # Measure how long the training epoch takes
        t0 = time.time()

        # Iterate over batches
        for step, batch in enumerate(dataloader):

            # Progress update after every 50 batches
            if step % 50 == 0 and not step == 0:

                # Calculate elapsed time in minutes
                elapsed = self._format_time(time.time() - t0)

                # Report progress
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

            # Push the batch to GPU/CPU
            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch

            # Clear previously calculated gradients
            self.model.zero_grad()

            # Get model predictions for the current batch
            preds = self.model(sent_id, mask)

            # Compute the loss between actual and predicted values
            loss = self.lossfunc(preds, labels)

            # Add on to the total loss
            total_loss = total_loss + loss.item()

            # Backward pass to calculate the gradients
            loss.backward()

            # Clip the the gradients to 1.0 to prevent the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update parameters
            self.optimizer.step()

            # Model predictions are stored on GPU, so push them to CPU
            preds = preds.detach().cpu().numpy()

            # Append the model predictions
            total_preds.append(preds)

        # Measure how long this epoch took
        epoch_time = self._format_time(time.time() - t0)

        # Compute the average training loss of the epoch
        avg_loss = total_loss / len(dataloader)

        # Reshape predictions in form of (number of samples, number of classes)
        total_probs = np.exp(np.concatenate(total_preds, axis=0)) # Convert to probabilities

        return avg_loss, epoch_time, total_probs


    def _evaluate(self, dataloader):
        """
        Args:
            dataloader (torch.utils.data.DataLoader): Validation data

        Returns:
            avg_loss (float): Average validation loss
            epoch_time (str): Elapsed time in "hh:mm:ss" format
            total_probs (numpy.array[float]): Predicted probabilities of product return
        """

        print("")
        print("Evaluating...")

        # Deactivate dropout layers
        self.model.eval()

        # Initialize variables to track
        total_loss, total_accuracy = 0, 0

        # Initiate empty list to save model predictions
        total_preds = []

        # Measure how long the training epoch takes
        t0 = time.time()

        # Iterate over batches
        for step, batch in enumerate(dataloader):

            # Progress update after every 50 batches
            if step % 50 == 0 and not step == 0:

                # Calculate elapsed time in minutes
                elapsed = self._format_time(time.time() - t0)

                # Report progress
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

            # Push the batch to GPU/CPU
            batch = [t.to(self.device) for t in batch]
            sent_id, mask, labels = batch

            # Deactivate autograd
            with torch.no_grad():

                # Get model predictions for the current batch
                preds = self.model(sent_id, mask)

                # Compute the validation loss between actual and predicted values
                loss = self.lossfunc(preds, labels)

                # Add on to the total loss
                total_loss = total_loss + loss.item()

                # Model predictions are stored on GPU, so push them to CPU
                preds = preds.detach().cpu().numpy()

                # Append the model predictions
                total_preds.append(preds)

        # Measure how long this epoch took
        epoch_time = self._format_time(time.time() - t0)

        # Compute the average training loss of the epoch
        avg_loss = total_loss / len(dataloader)

        # Reshape predictions in form of (number of samples, number of classes)
        total_probs = np.exp(np.concatenate(total_preds, axis=0)) # Convert to probabilities

        return avg_loss, epoch_time, total_probs


    def train(self, reviews, labels, save_filename):
        """Train model with input data and save best model weights in the specified location

        Args:
            reviews (list[str]): Review texts
            labels (list[bool]): Indicator for whether a review resulted in a product return
            save_filename (str): Path to file to save trained model weights

        Returns:
            (None)
        """
        self.model = self.model.to(self.device) # Push to GPU/CPU

        # Pre-process input data
        reviews, labels = list(reviews), list(labels) # Ensure correct type
        train_dataloader, val_dataloader = self._preprocess_train_data(reviews, labels)

        # Apply class weights to loss function
        train_labels = np.array(labels)[train_dataloader.dataset.indices]
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels,
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        class_weights = class_weights.to(self.device) # Push to GPU/CPU
        self.lossfunc = nn.NLLLoss(weight=class_weights)

        # Set initial loss to infinity
        best_valid_loss = float('inf')

        # Specify temporary file name to save model weights
        temp_filename = 'BERT_classifier_weights.pt'

        # Iterate over epochs
        total_t0 = time.time() # To measure the total training time for the whole run
        for epoch in range(self.epochs):

            print("")
            print("======== Epoch {:} / {:} ========".format(epoch + 1, self.epochs))

            # Train model
            train_loss, train_time, _ = self._train(train_dataloader)
            print("")
            print("  Average training loss: {0:.2f}".format(train_loss))
            print("  Training epoch took: {:}".format(train_time))

            # Evaluate model
            valid_loss, valid_time, _ = self._evaluate(val_dataloader)
            print("")
            print("  Average validation loss: {0:.2f}".format(valid_loss))
            print("  Validation epoch took: {:}".format(valid_time))

            # Save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), temp_filename)

            # Append training and validation loss
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self._format_time(time.time()-total_t0)))

        # Load weights of best model
        self.model.load_state_dict(torch.load(temp_filename, map_location=self.device))
        self.model_loaded = True

        # Move saved file to desired location
        shutil.move(temp_filename, save_filename)


    def load_model(self, model_filename=None):
        """Load pre-saved model weights

        Args:
            model_filename (str): Path to `.pt` file storing model weights

        Returns:
            (None)
        """
        if not model_filename:
            model_filename = path.join(self.root_path, 'models', 'BERT_classifier_weights_sprint8.pt')

        self.model = self.model.to(self.device) # Push to GPU/CPU
        self.model.load_state_dict(torch.load(model_filename, map_location=self.device))
        self.model_loaded = True
        print("Model loaded successfully")


    def _make_dataloader(self, reviews, batch_size):
        """
        Args:
            reviews (list[str]): Review texts
            batch_size (int): Batch size to use

        Returns:
            dataloader (torch.utils.data.DataLoader): Data to extract features from
        """

        # Tokenize and encode sequences in the input data set
        encoded_dict = self.tokenizer.batch_encode_plus(
            reviews,
            add_special_tokens=True,
            max_length=self.token_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt', # Return pytorch tensors
        )

        # Combine the results into a TensorDataset
        dataset = TensorDataset(encoded_dict['input_ids'], encoded_dict['attention_mask'])

        # Create data loader
        dataloader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset), # Pull out batches sequentially.
            batch_size=batch_size,
        )

        return dataloader


    def _extract(self, dataloader):
        """
        Args:
            dataloader (torch.utils.data.DataLoader): Data to extract features from

        Returns:
            epoch_time (str): Elapsed time in "hh:mm:ss" format
            model_outputs (dict[str, numpy.array[float]]): Model outputs including
                desired activation values and predicted probabilities of product return
        """

        print("")
        print("Extracting features...")

        # Deactivate dropout layers
        self.model.eval()

        # Set up the hook to extract activations from the desired intermidate layer
        total_activs = []
        def get_activation(save_lst):
            def hook(model, input, output):
                save_lst.append(output.detach().cpu().numpy())
            return hook
        self.model.fc2.register_forward_hook(get_activation(total_activs))

        # Initiate empty list to save model predictions
        total_preds = []

        # Measure how long the training epoch takes
        t0 = time.time()

        # Iterate over batches
        for step, batch in enumerate(dataloader):

            # Progress update after every 50 batches
            if step % 50 == 0 and not step == 0:

                # Calculate elapsed time in minutes
                elapsed = self._format_time(time.time() - t0)

                # Report progress
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

            # Push the batch to GPU/CPU
            batch = [t.to(self.device) for t in batch]
            sent_id, mask = batch

            # Deactivate autograd
            with torch.no_grad():

                # Get model predictions for the current batch
                preds = self.model(sent_id, mask)

                # Model predictions are stored on GPU, so push them to CPU
                preds = preds.detach().cpu().numpy()

                # Append the model predictions
                total_preds.append(preds)

        # Measure how long this epoch took
        epoch_time = self._format_time(time.time() - t0)

        # Collect all desired model outputs
        activs = np.concatenate(total_activs, axis=0)
        probs = np.exp(np.concatenate(total_preds, axis=0)) # Convert to probabilities
        model_outputs = {'activ' : activs, 'prob' : probs}

        return epoch_time, model_outputs


    def extract(self, reviews, batch_size=200):
        """
        Args:
            reviews (list[str]): Review texts
            batch_size (int): Batch size to use (default set to 200)

        Returns:
            outputs_df (pandas.DataFrame): Table with NLP feature columns
        """

        if not self.model_loaded:
            self.load_model() # Use pre-trained model weights

        if len(reviews) > 100000:
            print("WARNING: May take a long time; Consider dividing the work")

        # Create data loader
        reviews = list(reviews) # Ensure correct type
        dataloader = self._make_dataloader(reviews, batch_size)

        # Extract features
        pred_time, model_outputs = self._extract(dataloader)

        # Arrange results into a table
        outputs_df = pd.DataFrame(model_outputs['activ'])
        outputs_df.columns = [f'emb{i+1}' for i in outputs_df.columns]
        outputs_df['p_return'] = model_outputs['prob'][:,1]

        return outputs_df


def main():
    print("\n\n")

    # Load toy data
    CURR_PATH = path.abspath(__file__) # Full path to current script
    ROOT_PATH = path.dirname(path.dirname(path.dirname(CURR_PATH)))
    file_path = path.join(ROOT_PATH, "data", "demo", "reviews_toydata.csv")
    df = pd.read_csv(file_path)

    # Test feature extraction
    extractor = Returnability()
    features_df = extractor.extract(df["rvprcomments"])

    print("\n\n")
    print(features_df.head().round(2))


if __name__ == "__main__":
    main()
